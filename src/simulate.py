"""
Synthetic social stream with hidden coordinated campaigns.

Posts live in a latent "narrative space" (think: sentence embeddings).
Authentic users post with human rhythms and personal interests.
Coordinated campaigns behave like jammers: many accounts push one
message vector in synchronized bursts on a target narrative channel.

`adapt` in [0, 1] is the adversary's evasion budget (the social-media
analogue of frequency hopping / power control):
  - timing jitter grows      (bursts become smeared)
  - content noise grows      (templates become paraphrases)
  - decoy posting grows      (bots also post ordinary, human-like content)
  - wave times shift toward human daytime hours
"""
from dataclasses import dataclass
import numpy as np

MIN_PER_DAY = 24 * 60


@dataclass
class Stream:
    t: np.ndarray          # post time in minutes
    acct: np.ndarray       # account id per post
    emb: np.ndarray        # unit-norm content vector per post
    is_bot_post: np.ndarray
    is_bot: np.ndarray     # per account
    camp_of: np.ndarray    # per account, -1 for authentic
    centers: np.ndarray    # narrative centres (K x D)
    targets: np.ndarray    # target channel per campaign
    days: int


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _noise(rng, shape, level):
    """Isotropic noise whose expected norm is `level` (dimension-independent)."""
    return rng.normal(size=shape) * level / np.sqrt(shape[-1])


def _diurnal_minutes(rng, n, days):
    """Human posting times: low at night, peaks late morning and evening."""
    hours = np.arange(24)
    pdf = 0.15 + np.exp(-0.5 * ((hours - 11) / 2.5) ** 2) + 1.2 * np.exp(-0.5 * ((hours - 20) / 2.0) ** 2)
    pdf /= pdf.sum()
    h = rng.choice(24, size=n, p=pdf)
    d = rng.integers(0, days, size=n)
    return d * MIN_PER_DAY + h * 60 + rng.uniform(0, 60, size=n)


def simulate(seed=0, n_auth=1500, n_campaigns=3, bots_per_campaign=60,
             adapt=0.0, days=7, K=12, D=32):
    rng = np.random.default_rng(seed)
    centers = _unit(rng.normal(size=(K, D)))

    T, A, E, B = [], [], [], []

    # ---------------- authentic accounts ----------------
    for a in range(n_auth):
        interest = rng.dirichlet(np.full(K, 0.3))
        n_posts = 1 + rng.negative_binomial(1.2, 0.13)          # heavy-tailed, mean ~9
        style = _noise(rng, (D,), 0.35)
        # sessions: humans post in clumps
        n_sess = max(1, rng.poisson(n_posts / 3))
        sess_t = _diurnal_minutes(rng, n_sess, days)
        t = sess_t[rng.integers(0, n_sess, n_posts)] + rng.exponential(25, n_posts)
        ch = rng.choice(K, size=n_posts, p=interest)
        e = _unit(centers[ch] + style + _noise(rng, (n_posts, D), 0.7))
        T.append(t); A.append(np.full(n_posts, a)); E.append(e); B.append(np.zeros(n_posts, bool))

    # ---------------- coordinated campaigns ----------------
    targets = rng.choice(K, size=n_campaigns, replace=False)
    n_acc = n_auth
    camp_of = [-1] * n_auth
    sigma_t = 3 + 60 * adapt                 # minutes of timing jitter
    sigma_c = 0.15 + 0.85 * adapt            # content noise norm (paraphrasing)
    for c in range(n_campaigns):
        # message close to the target narrative, but with its own slant
        msg = _unit(centers[targets[c]] + 0.6 * _unit(rng.normal(size=D)))
        n_waves = 2 * days
        if adapt > 0:   # shift waves toward daytime as evasion increases
            human = _diurnal_minutes(rng, n_waves, days)
            uniform = rng.uniform(0, days * MIN_PER_DAY, n_waves)
            pick = rng.random(n_waves) < adapt
            waves = np.where(pick, human, uniform)
        else:
            waves = rng.uniform(0, days * MIN_PER_DAY, n_waves)
        p_join = 0.75 * (1 - 0.6 * adapt)     # adaptive bots skip waves to look human
        for _ in range(bots_per_campaign):
            joined = waves[rng.random(n_waves) < p_join]
            t = joined + rng.normal(0, sigma_t, joined.size)
            e = _unit(msg + _noise(rng, (joined.size, D), sigma_c))
            # decoys: human-looking posts on random channels
            n_dec = rng.poisson(6 * adapt)        # ...and fill the gap with decoys
            if n_dec:
                td = _diurnal_minutes(rng, n_dec, days)
                ed = _unit(centers[rng.integers(0, K, n_dec)] + _noise(rng, (n_dec, D), 0.7))
                t = np.concatenate([t, td]); e = np.vstack([e, ed])
            T.append(t); A.append(np.full(t.size, n_acc)); E.append(e)
            B.append(np.ones(t.size, bool))
            camp_of.append(c); n_acc += 1

    t = np.concatenate(T); acct = np.concatenate(A); emb = np.vstack(E); bp = np.concatenate(B)
    keep = (t >= 0) & (t < days * MIN_PER_DAY)
    camp_of = np.array(camp_of)
    return Stream(t[keep], acct[keep], emb[keep], bp[keep], camp_of >= 0, camp_of,
                  centers, targets, days)


def channel_of(stream):
    """Assign each post to its nearest narrative channel (like a frequency bin)."""
    return np.argmax(stream.emb @ stream.centers.T, axis=1)
