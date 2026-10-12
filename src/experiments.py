"""
Runs every experiment, writes results/results.json and figures/*.png.

    python src/experiments.py          (~1-2 minutes on a laptop CPU)
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(__file__))
from simulate import simulate, channel_of, MIN_PER_DAY
from detect import account_features, train_supervised, graph_scores, FEATURES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
RES = os.path.join(ROOT, "results"); os.makedirs(RES, exist_ok=True)

WINDOW_H = 6         # hours per feed window
N0 = 1.0             # noise floor (posts per window) in the SINR denominator
TOPK = 3             # "trending" slots in the feed

# --- reference palette (validated categorical slots, fixed order) ---
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 12,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "lines.linewidth": 2, "lines.markersize": 6, "font.size": 10, "legend.frameon": False,
    "figure.dpi": 150,
})


# ---------------------------------------------------------------- metrics

def window_counts(s, weight=None):
    """(channels x windows) volume, split into authentic and coordinated parts."""
    w = np.ones(len(s.t)) if weight is None else weight
    ch = channel_of(s)
    win = (s.t // (WINDOW_H * 60)).astype(int)
    nW = s.days * 24 // WINDOW_H
    K = len(s.centers)
    auth = np.zeros((K, nW)); coord = np.zeros((K, nW))
    np.add.at(auth, (ch[~s.is_bot_post], win[~s.is_bot_post]), w[~s.is_bot_post])
    np.add.at(coord, (ch[s.is_bot_post], win[s.is_bot_post]), w[s.is_bot_post])
    return auth, coord


def discourse_sinr(s, weight=None):
    """Median SINR (dB) of the authentic signal in the campaigns' target channels,
    measured over *attacked* windows only (as RF SINR is measured while the
    jammer is on). Which windows count as attacked is fixed by ground truth, so
    mitigation cannot 'improve' SINR by changing the window set."""
    auth, coord = window_counts(s, weight)
    _, coord0 = window_counts(s)
    S, I = auth[s.targets], coord[s.targets]
    on = coord0[s.targets] > 0
    if not on.any():
        return float("nan")
    return float(np.median(10 * np.log10((S[on] + 1e-9) / (I[on] + N0))))


def displacement(s, weight=None):
    """Share of authentic 'trending' slots (top-K by authentic volume)
    that are lost once coordinated volume is added to the feed."""
    auth, coord = window_counts(s, weight)
    auth0, _ = window_counts(s)            # unweighted authentic baseline
    lost = tot = 0
    for w in range(auth.shape[1]):
        true_top = set(np.argsort(-auth0[:, w])[:TOPK])
        seen_top = set(np.argsort(-(auth[:, w] + coord[:, w]))[:TOPK])
        lost += len(true_top - seen_top); tot += TOPK
    return lost / tot


def recall_at_fpr(y, p, fpr_max=0.01):
    fpr, tpr, _ = roc_curve(y, p)
    return float(tpr[fpr <= fpr_max].max())


# ---------------------------------------------------------------- experiments

def exp_drowning(seeds=(11, 12, 13)):
    sizes = [0, 5, 10, 20, 30, 45, 60, 90, 130, 180]
    rows = []
    for b in sizes:
        sv, dv, share = [], [], []
        for sd in seeds:
            s = simulate(seed=sd, bots_per_campaign=b)
            sv.append(discourse_sinr(s) if b else np.nan)
            dv.append(displacement(s))
            share.append(s.is_bot.mean())
        rows.append(dict(bots_per_campaign=b, bot_account_share=float(np.mean(share)),
                         sinr_db=float(np.nanmean(sv)) if b else None,
                         displacement=float(np.mean(dv))))
    # interpolate the 0 dB crossing
    xs = np.array([r["bots_per_campaign"] for r in rows[1:]])
    ys = np.array([r["sinr_db"] for r in rows[1:]])
    i = np.argmax(ys < 0)
    cross = float(xs[i - 1] + (xs[i] - xs[i - 1]) * ys[i - 1] / (ys[i - 1] - ys[i])) if i > 0 else None
    return rows, cross


def exp_robustness():
    levels = np.round(np.linspace(0, 1, 9), 3)
    # clean training: the defender has only seen naive campaigns
    clean = [simulate(seed=sd, adapt=0.0) for sd in (100, 101, 102)]
    # adversarial training: defender has seen some evasion, but never a > 0.6
    adv = [simulate(seed=sd, adapt=a) for sd, a in zip(range(110, 119), [0, .3, .6] * 3)]
    fc = [account_features(s)[0] for s in clean]
    fa = [account_features(s)[0] for s in adv]
    clf_clean = train_supervised(fc, [s.is_bot for s in clean])
    clf_adv = train_supervised(fc + fa, [s.is_bot for s in clean] + [s.is_bot for s in adv])

    rows, single = [], {}
    cache = {}
    for a in levels:
        r = {"adapt": float(a)}
        acc = {k: [] for k in ["auc_clean", "auc_adv", "auc_graph", "r1_clean", "r1_adv",
                               "target_share", "bot_volume"]}
        for sd in (200, 201, 202):
            s = simulate(seed=sd, adapt=float(a)); X, G = account_features(s)
            y = s.is_bot.astype(int)
            pc, pa, pg = clf_clean.predict_proba(X)[:, 1], clf_adv.predict_proba(X)[:, 1], graph_scores(G)
            acc["auc_clean"].append(roc_auc_score(y, pc)); acc["auc_adv"].append(roc_auc_score(y, pa))
            acc["auc_graph"].append(roc_auc_score(y, pg))
            acc["r1_clean"].append(recall_at_fpr(y, pc)); acc["r1_adv"].append(recall_at_fpr(y, pa))
            ch = channel_of(s)
            acc["target_share"].append(np.isin(ch[s.is_bot_post], s.targets).sum() / s.is_bot.sum())
            acc["bot_volume"].append(s.is_bot_post.sum() / s.is_bot.sum())
            if sd == 200:
                cache[float(a)] = (s, X, G, pa)
        r.update({k: float(np.mean(v)) for k, v in acc.items()})
        rows.append(r)
    # which single signals survive evasion?
    for a in (0.0, 1.0):
        s, X, _, _ = cache[a]
        single[str(a)] = {f: float(roc_auc_score(s.is_bot, X[:, j])) for j, f in enumerate(FEATURES)}
    return rows, single, cache


def exp_tradeoff(cache):
    out = {}
    for a in (0.0, 0.5, 1.0):
        s, X, G, p = cache[a]
        post_p = p[s.acct]
        base = discourse_sinr(s)
        curve = []
        for tau in [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02]:
            w = (post_p < tau).astype(float)
            curve.append(dict(tau=tau, sinr_gain_db=discourse_sinr(s, w) - base,
                              auth_silenced=float((p[~s.is_bot] >= tau).mean()),
                              bots_removed=float((p[s.is_bot] >= tau).mean())))
        soft = 1 - post_p    # downweight instead of delete
        out[str(a)] = dict(base_sinr_db=base, curve=curve,
                           soft=dict(sinr_gain_db=discourse_sinr(s, soft) - base,
                                     auth_weight_lost=float(1 - soft[~s.is_bot_post].mean()),
                                     displacement_before=displacement(s),
                                     displacement_after=displacement(s, soft)))
    return out


# ---------------------------------------------------------------- figures

def fig_spectrogram(s, p):
    """Narrative channels x time, like an RF waterfall: authentic-only,
    observed (with campaigns), and after soft mitigation."""
    ch = channel_of(s); hr = (s.t // 60).astype(int); K, H = len(s.centers), s.days * 24
    def grid(mask, w=None):
        g = np.zeros((K, H)); np.add.at(g, (ch[mask], hr[mask]), 1 if w is None else w[mask]); return g
    w = 1 - p[s.acct]
    panels = [("Authentic voices only (ground truth)", grid(~s.is_bot_post)),
              ("What the feed sees: authentic + coordinated", grid(np.ones(len(ch), bool))),
              ("After soft mitigation (posts down-weighted by detector score)", grid(np.ones(len(ch), bool), w))]
    vmax = np.percentile(panels[1][1], 99.5)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.2), sharex=True, constrained_layout=True)
    for ax, (title, g) in zip(axes, panels):
        im = ax.imshow(g, aspect="auto", cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10.5); ax.grid(False)
        ax.set_yticks(range(K)); ax.set_yticklabels([f"N{k}" + ("  ◂ target" if k in s.targets else "") for k in range(K)], fontsize=7)
        for sp in ax.spines.values(): sp.set_visible(False)
    axes[-1].set_xticks(range(0, H + 1, 24)); axes[-1].set_xticklabels([f"day {d}" for d in range(s.days + 1)])
    axes[-1].set_xlabel("time (1-hour bins)")
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.01); cb.set_label("posts per hour"); cb.outline.set_visible(False)
    fig.suptitle("Discourse spectrogram: narrative channels over time", x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.savefig(os.path.join(FIG, "fig1_discourse_spectrogram.png")); plt.close(fig)


def fig_drowning(rows, cross):
    b = [r["bots_per_campaign"] for r in rows[1:]]
    sinr = [r["sinr_db"] for r in rows[1:]]
    disp = [100 * r["displacement"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    a1.plot(b, sinr, "-o", color=S1)
    a1.axhline(0, color=INK2, lw=1, ls="--"); a1.text(b[-1], 0.6, "0 dB: authentic = coordinated", ha="right", color=INK2, fontsize=8)
    if cross:
        a1.axvline(cross, color=INK2, lw=1, ls=":")
        a1.annotate(f"≈{cross:.0f} accounts per campaign\ndrown the channel", (cross, 0), (cross + 15, 8),
                    fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    a1.set_title("Discourse SINR in the targeted channel"); a1.set_xlabel("coordinated accounts per campaign"); a1.set_ylabel("median SINR (dB)")
    a2.plot([r["bots_per_campaign"] for r in rows], disp, "-o", color=S1)
    a2.set_title("Authentic trending slots displaced"); a2.set_xlabel("coordinated accounts per campaign")
    a2.set_ylabel(f"% of top-{TOPK} slots lost"); a2.set_ylim(0, None)
    fig.savefig(os.path.join(FIG, "fig2_drowning_curve.png")); plt.close(fig)


def fig_robustness(rows):
    a = [r["adapt"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9), constrained_layout=True)
    for key, lab, c in [("auc_clean", "naive-trained", S1),
                        ("auc_adv", "evasion-trained (≤ 0.6)", S2),
                        ("auc_graph", "co-action graph", S3)]:
        y = [r[key] for r in rows]
        a1.plot(a, y, "-o", color=c, label=lab)
    a1.axvspan(0.6, 1.0, color=GRID, alpha=0.5, lw=0); a1.text(0.8, 0.93, "unseen evasion levels", ha="center", color=INK2, fontsize=8)
    a1.set_ylim(0.45, 1.02); a1.set_xlabel("adversary adaptation (timing jitter, paraphrase, decoys)")
    a1.set_ylabel("ROC-AUC"); a1.set_title("Detector reliability as jammers adapt"); a1.legend(loc="lower left", bbox_to_anchor=(0.0, 0.1), fontsize=8)
    a2.plot(a, [r["target_share"] for r in rows], "-o", color=S1)
    a2.set_xlabel("adversary adaptation"); a2.set_ylabel("on-target posts per bot")
    a2.set_title("The price of evasion: jamming power per bot"); a2.set_ylim(0, None)
    fig.savefig(os.path.join(FIG, "fig3_robustness.png")); plt.close(fig)


def fig_tradeoff(tr):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for (a, d), c in zip(tr.items(), [S1, S2, S3]):
        x = [100 * r["auth_silenced"] for r in d["curve"]]; y = [r["sinr_gain_db"] for r in d["curve"]]
        ax.plot(x, y, "-o", color=c, label=f"adapt = {float(a):.1f}")
        ax.text(x[-1] + 0.05, y[-1], f"adapt {float(a):.1f}", color=INK, fontsize=8, va="center")
    ax.set_xlabel("authentic accounts silenced (%)"); ax.set_ylabel("SINR gain in target channels (dB)")
    ax.set_title("Mitigation vs. collateral: every dB costs voices"); ax.set_xlim(-0.05, 2.4); ax.legend(fontsize=8, loc="lower right")
    fig.savefig(os.path.join(FIG, "fig4_mitigation_tradeoff.png")); plt.close(fig)


def fig_graph(cache):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    for ax, a in zip(axes, (0.0, 0.5)):
        s, _, G, _ = cache[a]
        H = G.subgraph([n for n in G if G.degree(n) > 0])
        pos = nx.spring_layout(H, seed=1, k=0.25)
        nx.draw_networkx_edges(H, pos, ax=ax, edge_color=GRID, width=0.4, alpha=0.6)
        col = [S2 if s.is_bot[n] else S1 for n in H]
        nx.draw_networkx_nodes(H, pos, ax=ax, node_color=col, node_size=14, linewidths=0.6, edgecolors=SURF)
        nb = int(sum(s.is_bot[n] for n in H)); na = H.number_of_nodes() - nb
        ax.set_title(f"Co-action graph, adapt = {a}  ·  {nb} coordinated, {na} authentic nodes", fontsize=10)
        ax.axis("off")
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], marker="o", ls="", color=S2), Line2D([], [], marker="o", ls="", color=S1)],
               ["coordinated account", "authentic account"], loc="lower center", ncol=2, fontsize=9)
    fig.savefig(os.path.join(FIG, "fig5_coaction_graph.png")); plt.close(fig)


if __name__ == "__main__":
    print("1/3 drowning curve ...");  drown, cross = exp_drowning()
    print("2/3 robustness ...");      rob, single, cache = exp_robustness()
    print("3/3 mitigation ...");      trade = exp_tradeoff(cache)
    s, _, _, p = cache[0.5]
    fig_spectrogram(s, p); fig_drowning(drown, cross); fig_robustness(rob); fig_tradeoff(trade); fig_graph(cache)
    out = dict(drowning=drown, zero_db_crossing_bots_per_campaign=cross, robustness=rob,
               single_feature_auc=single, mitigation=trade)
    json.dump(out, open(os.path.join(RES, "results.json"), "w"), indent=2)
    print(json.dumps({"cross": cross, "robustness": rob, "single": single,
                      "soft": {k: v["soft"] | {"base": v["base_sinr_db"]} for k, v in trade.items()}}, indent=1))
