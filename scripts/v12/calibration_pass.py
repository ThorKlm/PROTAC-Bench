#!/usr/bin/env python
"""Appendix M metrics from stored LOTO predictions at the canonical seeds.

Raw-output Brier and ECE-10, reliability bins, risk-coverage, and temperature
and Platt rescaling under a leave-one-target-out calibrator. Baseline condition
only: the stored predictions carry one condition per target.
"""
import json, sys, os
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
EPS = 1e-6
import sys as _s
SRC = _s.argv[1] if len(_s.argv) > 1 else 'results/v12/loto_predictions.json'
OUT = _s.argv[2] if len(_s.argv) > 2 else 'results/calibration.json'

def logit(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))
def sig(z): return 1.0 / (1.0 + np.exp(-z))

def ece(p, y, bins=10):
    e, n, edge = 0.0, len(y), np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edge[:-1], edge[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.sum(): e += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(e)

def reliability(p, y, nb=10):
    """Per-bin mean prediction and observed frequency, binned exactly as ece()
    bins, so a reliability plot and the reported ECE cannot disagree."""
    edge, out = np.linspace(0, 1, nb + 1), []
    for j, (lo, hi) in enumerate(zip(edge[:-1], edge[1:])):
        m = (p >= lo) & (p <= hi) if j == 0 else (p > lo) & (p <= hi)
        out.append({'lo': float(lo), 'hi': float(hi), 'n': int(m.sum()),
                    'mean_p': float(p[m].mean()) if m.sum() else None,
                    'obs': float(y[m].mean()) if m.sum() else None})
    return out

def fit_T(z, y):
    f = lambda T: -np.mean(y * np.log(sig(z / T) + EPS) + (1 - y) * np.log(1 - sig(z / T) + EPS))
    return float(minimize_scalar(f, bounds=(0.05, 20.0), method='bounded').x)

def coverage_curve(p, y, covs=(1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 0.05)):
    """Metrics on the most-confident fraction, confidence being distance from
    0.5. AUROC is reported because the earlier pass did, and error rate and
    Brier beside it because discarding uncertain cases removes the hard pairs
    AUROC penalises, so AUROC alone rises partly by construction."""
    from sklearn.metrics import roc_auc_score
    conf = np.maximum(p, 1 - p); order = np.argsort(-conf)
    out = {}
    for c in covs:
        k = max(2, int(round(c * len(y))))
        idx = order[:k]
        pr = float(y[idx].mean())
        out[str(c)] = {'n': int(k), 'pos_rate': pr,
                       'majority_error': float(1 - max(pr, 1 - pr)),
                       'auroc': (float(roc_auc_score(y[idx], p[idx]))
                                 if len(np.unique(y[idx])) > 1 else None),
                       'brier': float(np.mean((p[idx] - y[idx]) ** 2)),
                       'error_rate': float(np.mean((p[idx] > 0.5) != (y[idx] > 0.5)))}
    return out

def main():
    d = json.load(open(SRC))
    seeds = sorted(d, key=lambda s: (s != '42', int(s)))
    per_seed = {}
    for s in seeds:
        per = d[s]; tg = sorted(per)
        P = {t: np.asarray(per[t]['p'], float) for t in tg}
        Y = {t: np.asarray(per[t]['y'], float) for t in tg}
        p_all = np.concatenate([P[t] for t in tg]); y_all = np.concatenate([Y[t] for t in tg])
        pt, pp = np.zeros_like(p_all), np.zeros_like(p_all)
        i = 0
        for t in tg:
            oth = [u for u in tg if u != t]
            zo = logit(np.concatenate([P[u] for u in oth])); yo = np.concatenate([Y[u] for u in oth])
            T = fit_T(zo, yo)
            lr = LogisticRegression(solver='lbfgs').fit(zo.reshape(-1, 1), yo)
            n = len(P[t]); z = logit(P[t])
            pt[i:i + n] = sig(z / T)
            pp[i:i + n] = lr.predict_proba(z.reshape(-1, 1))[:, 1]
            i += n
        per_seed[s] = {'n': int(len(y_all)), 'n_targets': len(tg),
                       'brier_raw': float(np.mean((p_all - y_all) ** 2)),
                       'ece10_raw': ece(p_all, y_all), 'ece10_temperature': ece(pt, y_all),
                       'ece10_platt': ece(pp, y_all), 'coverage_raw': coverage_curve(p_all, y_all),
                       'reliability_raw': reliability(p_all, y_all),
                       'reliability_platt': reliability(pp, y_all)}
        print(s, {k: (round(v, 4) if isinstance(v, float) else v)
                  for k, v in per_seed[s].items() if not isinstance(v, (dict, list))}, flush=True)
    import statistics as st
    agg = {}
    for k in ('brier_raw', 'ece10_raw', 'ece10_temperature', 'ece10_platt'):
        v = [per_seed[s][k] for s in seeds]
        agg[k] = {'mean': sum(v) / len(v), 'sample_sd': st.stdev(v) if len(v) > 1 else 0.0}
    json.dump({'condition': 'baseline (Morgan) raw LOTO predictions',
               'protocol': 'leave-one-target-out calibrator over stored predictions; '
                           'base model not refit, unlike the discarded exp17 protocol',
               'source': SRC, 'seeds': seeds, 'aggregate': agg, 'per_seed': per_seed},
              open(OUT, 'w'), indent=1)
    for k, v in agg.items(): print(k, round(v['mean'], 4), round(v['sample_sd'], 4))
    print('wrote', OUT)

if __name__ == '__main__':
    main()
