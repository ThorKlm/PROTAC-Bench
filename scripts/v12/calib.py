"""Temperature and Platt scaling under a leave-one-target-out calibrator fit,
from stored LOTO predictions. Settles the sign of the temperature effect; the
absolute ECE differs from Appendix M's protocol, which refits the base model."""
import json, numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
EPS = 1e-6
d = json.load(open('results/v12/loto_predictions.json'))
def logit(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))
def sig(z): return 1.0 / (1.0 + np.exp(-z))
def ece(p, y, bins=10):
    e, n = 0.0, len(y)
    edge = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edge[:-1], edge[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.sum(): e += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return e
def fit_T(z, y):
    f = lambda T: -np.mean(y * np.log(sig(z / T) + EPS) + (1 - y) * np.log(1 - sig(z / T) + EPS))
    return minimize_scalar(f, bounds=(0.05, 20.0), method='bounded').x
rows = []
for s, per in d.items():
    tgts = sorted(per)
    P = {t: np.array(per[t]['p']) for t in tgts}
    Y = {t: np.array(per[t]['y']) for t in tgts}
    raw, tmp, plt = [], [], []
    for t in tgts:
        oth = [u for u in tgts if u != t]
        zc = logit(np.concatenate([P[u] for u in oth])); yc = np.concatenate([Y[u] for u in oth])
        zt, yt = logit(P[t]), Y[t]
        T = fit_T(zc, yc)
        lr = LogisticRegression(C=1e6, max_iter=2000).fit(zc.reshape(-1, 1), yc)
        raw.append((sig(zt), yt)); tmp.append((sig(zt / T), yt))
        plt.append((lr.predict_proba(zt.reshape(-1, 1))[:, 1], yt))
    cat = lambda L: (np.concatenate([a for a, _ in L]), np.concatenate([b for _, b in L]))
    rows.append([ece(*cat(raw)), ece(*cat(tmp)), ece(*cat(plt))])
r = np.array(rows)
for i, name in enumerate(('raw', 'temperature', 'platt')):
    print(f'ECE-10 {name:12s} {r[:,i].mean():.4f} sd {r[:,i].std(ddof=1):.4f}')
print(f'\ntemperature effect {r[:,1].mean()-r[:,0].mean():+.4f}  '
      f'platt effect {r[:,2].mean()-r[:,0].mean():+.4f}  (n_seeds {len(rows)})')
