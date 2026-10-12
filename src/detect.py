"""
RF-inspired account features and three detectors.

Feature families (each borrows an idea from jamming detection):
  timing     inter-post CV, burstiness        ~ pulse / duty-cycle statistics
  spectral   periodicity of the activity       ~ spectral peak-to-average ratio
  rhythm     entropy of posting hour           ~ deviation from the band's normal occupancy
  content    self-similarity of own posts      ~ narrowband (low-entropy) emitter
  co-action  near-identical posts from other   ~ several transmitters locked to
             accounts within minutes             the same carrier at the same time
"""
from collections import defaultdict
import numpy as np
import networkx as nx
from sklearn.ensemble import GradientBoostingClassifier

WIN = 15            # minutes: co-action window
SIM = 0.80          # cosine threshold for "near-identical" posts

FEATURES = ["log_posts", "interval_cv", "burstiness", "spectral_par",
            "hour_entropy", "self_similarity", "twin_rate", "coaction_degree",
            "coaction_strength"]


def coaction_pairs(s):
    """Count, for each account pair, how often they posted near-identical
    content within WIN minutes of each other. Blocked by time bin -> O(n)."""
    order = np.argsort(s.t)
    t, a, e = s.t[order], s.acct[order], s.emb[order]
    b = (t // WIN).astype(int)
    starts = np.searchsorted(b, np.arange(b.max() + 2))
    pairs = defaultdict(int)
    has_twin = np.zeros(len(t), bool)
    for k in range(b.max() + 1):
        i0, i1, i2 = starts[k], starts[k + 1], starts[min(k + 2, len(starts) - 1)]
        if i1 == i0:
            continue
        S = e[i0:i1] @ e[i0:i2].T
        dt = np.abs(t[i0:i1, None] - t[None, i0:i2])
        hit = (S > SIM) & (dt <= WIN)
        ii, jj = np.nonzero(hit)
        jj = jj + i0; ii = ii + i0
        m = (a[ii] != a[jj]) & (jj > ii)
        for x, y in zip(a[ii[m]], a[jj[m]]):
            pairs[(min(x, y), max(x, y))] += 1
        has_twin[ii[m]] = True; has_twin[jj[m]] = True
    twin = np.zeros(len(t), bool); twin[order] = has_twin
    return pairs, twin


def account_features(s):
    n_acc = len(s.is_bot)
    pairs, twin = coaction_pairs(s)
    G = nx.Graph()
    G.add_nodes_from(range(n_acc))
    for (x, y), w in pairs.items():
        if w >= 2:
            G.add_edge(x, y, weight=w)

    X = np.zeros((n_acc, len(FEATURES)))
    order = np.argsort(s.acct, kind="stable")
    acct_sorted = s.acct[order]
    bounds = np.searchsorted(acct_sorted, np.arange(n_acc + 1))
    horizon = s.days * 24 * 60
    for i in range(n_acc):
        idx = order[bounds[i]:bounds[i + 1]]
        n = len(idx)
        if n == 0:
            continue
        t = np.sort(s.t[idx])
        X[i, 0] = np.log1p(n)
        if n >= 3:
            d = np.diff(t)
            mu, sd = d.mean(), d.std()
            X[i, 1] = sd / (mu + 1e-9)
            X[i, 2] = (sd - mu) / (sd + mu + 1e-9)
        # spectral peak-to-average ratio of the hourly activity series
        series = np.bincount((t // 60).astype(int), minlength=horizon // 60)
        P = np.abs(np.fft.rfft(series - series.mean())) ** 2
        X[i, 3] = P[1:].max() / (P[1:].mean() + 1e-9) if n >= 3 else 0
        hp = np.bincount(((t % 1440) // 60).astype(int), minlength=24) / n
        H = -(hp[hp > 0] * np.log(hp[hp > 0])).sum()
        X[i, 4] = H / np.log(min(n, 24)) if n >= 2 else 0   # normalised: count-free
        if n >= 2:
            e = s.emb[idx]
            C = e @ e.T
            X[i, 5] = (C.sum() - n) / (n * (n - 1))
        X[i, 6] = twin[idx].mean()
        X[i, 7] = G.degree(i)
        X[i, 8] = G.degree(i, weight="weight") / max(n, 1)
    return X, G


# ---------------- detectors ----------------

def train_supervised(Xs, ys, seed=0):
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                     learning_rate=0.05, random_state=seed)
    clf.fit(np.vstack(Xs), np.concatenate(ys))
    return clf


def graph_scores(G, min_comm=5):
    """Label-free detector: coordinated communities in the co-action graph.
    Score = weighted degree, zeroed outside communities of >= min_comm."""
    score = np.zeros(G.number_of_nodes())
    H = G.subgraph([n for n in G if G.degree(n) > 0])
    if H.number_of_edges() == 0:
        return score
    for comm in nx.community.louvain_communities(H, weight="weight", seed=0):
        if len(comm) >= min_comm:
            for n in comm:
                score[n] = H.degree(n, weight="weight")
    return score
