"""Figures for the reproduced suite. Every value is read from results/."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R, OUT = 'results', os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({'figure.dpi': 150, 'font.size': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25,
                     'grid.linewidth': 0.5, 'legend.frameon': False,
                     'savefig.bbox': 'tight'})
C = ['#1f4e79', '#c0504d', '#4f81bd', '#9bbb59', '#8064a2']

def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None

def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p); plt.close(fig); print('wrote', p)

def protocol_collapse():
    pr, dd, rt = load('protocols.json'), load('cell_1000.json'), load('retained_65.json')
    if not (pr and dd and rt): return
    rows = [(k, pr['aggregate'][k]['pooled_mean'], pr['aggregate'][k]['across_seed_sd'])
            for k in sorted(pr['aggregate'])]
    rows += [('loto_retained', rt['macro_auroc'], rt['seed_sd']),
             ('loto_dedup', dd['macro_auroc'], dd['seed_sd'])]
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    x = np.arange(len(rows))
    ax.bar(x, [r[1] for r in rows], yerr=[r[2] for r in rows], capsize=2,
           color=[C[0]] * (len(rows) - 2) + [C[1], C[1]])
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], rotation=35, ha='right')
    ax.axhline(0.5, color='k', lw=0.6, ls=':')
    ax.set_ylabel('AUROC')
    save(fig, 'protocol_collapse.pdf')

def factorial():
    fs = sorted(glob.glob(f'{R}/cell_[01][01][01][01].json'))
    if not fs: return
    ds = [(os.path.basename(p)[5:9], json.load(open(p))) for p in fs]
    ds.sort(key=lambda z: z[1]['macro_auroc'])
    lab = ['morgan', 'warhead', 'admet', 'shots']
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    x = np.arange(len(ds))
    ax.bar(x, [d['macro_auroc'] for _, d in ds], yerr=[d['seed_sd'] for _, d in ds],
           capsize=2, color=C[2])
    ax.set_xticks(x)
    ax.set_xticklabels(['+'.join(l for l, b in zip(lab, c) if b == '1') or 'none'
                        for c, _ in ds], rotation=40, ha='right', fontsize=6)
    ax.set_ylabel('macro AUROC')
    save(fig, 'factorial.pdf')

def baselines():
    ds = [(n, load(f'{n}.json')) for n in ('cell_1000', 'xgb_morgan', 'gin', 'dmpnn')]
    ds = [(n, d) for n, d in ds if d]
    if not ds: return
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    x = np.arange(len(ds))
    ax.bar(x, [d['macro_auroc'] for _, d in ds], yerr=[d['target_se'] for _, d in ds],
           capsize=2, color=C[0])
    ax.set_xticks(x)
    ax.set_xticklabels([(d.get('_meta') or {}).get('model')
                        or (d.get('spec') or {}).get('architecture') or n
                        for n, d in ds], rotation=20, ha='right')
    ax.axhline(0.5, color='k', lw=0.6, ls=':')
    ax.set_ylabel('macro AUROC')
    save(fig, 'baselines.pdf')

def fewshot():
    pts = [(0, 'cell_1110'), (1, 'fewshot_k1'), (3, 'fewshot_k3'),
           (5, 'cell_1111'), (10, 'fewshot_k10')]
    ds = [(k, load(f'{f}.json')) for k, f in pts]
    ds = [(k, d) for k, d in ds if d]
    if not ds: return
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for key, sd, lb, col in (('macro_auroc', 'seed_sd', 'macro AUROC', C[0]),
                             ('mean_ap', 'ap_seed_sd', 'average precision', C[1])):
        e = [d[sd] for _, d in ds]
        if any(x is None for x in e):
            raise ValueError(f'few-shot figure: {sd} absent in an arm')
        ax.errorbar([k for k, _ in ds], [d[key] for _, d in ds],
                    yerr=e, marker='o', ms=3, lw=1, capsize=2, color=col, label=lb)
    rnd = [(k, load(f'fewshot_k{k}_random.json')) for k in (5, 10)]
    rnd = [(k, d) for k, d in rnd if d]
    if rnd:
        e = [d['seed_sd'] for _, d in rnd]
        if any(x is None for x in e):
            raise ValueError('few-shot figure: seed_sd absent in a random-shot arm')
        ax.errorbar([k for k, _ in rnd], [d['macro_auroc'] for _, d in rnd],
                    yerr=e, marker='s', ms=4,
                    lw=0, capsize=2, color=C[2], label='macro AUROC, random shots')
    ax.set_xlabel('labelled shots per target'); ax.set_ylabel('score')
    ax.legend(); save(fig, 'fewshot.pdf')

def sweep():
    ref = load('cell_1000.json')
    fs = sorted(glob.glob(f'{R}/oat_*.json'))
    if not (ref and fs): return
    ds = [(os.path.basename(p)[4:-5], json.load(open(p))) for p in fs]
    ds.sort(key=lambda z: z[1]['macro_auroc'])
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    y = np.arange(len(ds))
    ax.barh(y, [d['macro_auroc'] - ref['macro_auroc'] for _, d in ds],
            xerr=[d['seed_sd'] for _, d in ds], capsize=2, color=C[3])
    ax.set_yticks(y); ax.set_yticklabels([n for n, _ in ds], fontsize=7)
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('macro AUROC minus reference')
    save(fig, 'sweep.pdf')

def label_noise():
    fs = sorted(glob.glob(f'{R}/noise_*.json'))
    if not fs: return
    ds = [json.load(open(p)) for p in fs]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.errorbar([d['spec'].get('flip', 0.0) for d in ds], [d['macro_auroc'] for d in ds],
                yerr=[d['seed_sd'] for d in ds], marker='o', ms=3, lw=1, capsize=2,
                color=C[1])
    ax.axhline(0.5, color='k', lw=0.6, ls=':')
    ax.set_xlabel('fraction of training labels flipped')
    ax.set_ylabel('macro AUROC')
    save(fig, 'label_noise.pdf')

def calibration():
    c = load('calibration.json')
    if not c: return
    ps = c['per_seed']; s0 = list(ps)[0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.6, 2.5))
    nb = len(ps[s0]['reliability_raw'])
    xs, ys = [], []
    for i in range(nb):
        mp = [ps[s]['reliability_raw'][i]['mean_p'] for s in ps
              if ps[s]['reliability_raw'][i]['n']]
        ob = [ps[s]['reliability_raw'][i]['obs'] for s in ps
              if ps[s]['reliability_raw'][i]['n']]
        if mp: xs.append(np.mean(mp)); ys.append(np.mean(ob))
    a1.plot([0, 1], [0, 1], color='k', lw=0.6, ls=':')
    a1.plot(xs, ys, marker='o', ms=3, lw=1, color=C[0])
    a1.set_xlabel('predicted probability'); a1.set_ylabel('observed rate')
    cov = sorted(ps[s0]['coverage_raw'], key=float)
    a2.plot([float(k) for k in cov],
            [np.mean([ps[s]['coverage_raw'][k]['error_rate'] for s in ps]) for k in cov],
            marker='o', ms=3, lw=1, color=C[1])
    a2.set_xlabel('coverage'); a2.set_ylabel('error rate')
    save(fig, 'calibration.pdf')

def similarity():
    s = load('similarity.json')
    if not s: return
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    for i, k in enumerate([x for x in sorted(s) if not x.startswith('_')]):
        e, h = np.asarray(s[k]['hist_edges']), np.asarray(s[k]['hist_counts'])
        ax.step(e[:-1], h / h.sum(), where='post', lw=1, color=C[i % len(C)], label=k)
    ax.set_xlabel('maximum Tanimoto to nearest training compound')
    ax.set_ylabel('fraction of rows')
    ax.legend(fontsize=6); save(fig, 'similarity.pdf')

def overlap():
    o, rt = load('overlap.json'), load('retained_65.json')
    if not (o and rt): return
    rows = [('full', rt['macro_auroc'], rt['seed_sd']),
            ('novel', o['novel']['macro'], o['novel']['seed_sd']),
            ('seen', o['seen']['macro'], o['seen']['seed_sd'])]
    fig, ax = plt.subplots(figsize=(2.8, 2.5))
    x = np.arange(len(rows))
    ax.bar(x, [r[1] for r in rows], yerr=[r[2] for r in rows], capsize=2,
           color=[C[0], C[1], C[3]])
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows])
    ax.axhline(0.5, color='k', lw=0.6, ls=':')
    ax.set_ylabel('macro AUROC')
    save(fig, 'overlap.pdf')

def cohort_variants():
    d = load('cohort_cells.json')
    if not d: return
    ks = [k for k in d if not k.startswith('_')]
    ks.sort(key=lambda k: d[k]['macro'])
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    x = np.arange(len(ks))
    ax.bar(x, [d[k]['macro'] for k in ks], yerr=[d[k]['seed_sd'] for k in ks],
           capsize=2, color=[C[1] if d[k]['dedup'] else C[0] for k in ks])
    ax.set_xticks(x); ax.set_xticklabels(ks, rotation=30, ha='right', fontsize=7)
    ax.axhline(0.5, color='k', lw=0.6, ls=':')
    ax.set_ylabel('macro AUROC')
    save(fig, 'cohort_variants.pdf')

def splits():
    lf, e3 = load('lofo.json'), load('cross_e3.json')
    if not lf: return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.2, 4.0),
                                 gridspec_kw={'width_ratios': [2, 1]})
    pt = sorted(lf['per_target'].items(), key=lambda z: z[1])
    y = np.arange(len(pt))
    a1.barh(y, [b for _, b in pt], color=C[2])
    a1.set_yticks(y); a1.set_yticklabels([a for a, _ in pt], fontsize=6)
    a1.axvline(lf['macro_auroc'], color=C[1], lw=1)
    a1.axvline(0.5, color='k', lw=0.6, ls=':')
    a1.set_xlabel('AUROC, held-out family'); a1.set_xlim(0, 1)
    if e3:
        pe = sorted(e3['per_target'].items(), key=lambda z: z[1])
        ye = np.arange(len(pe))
        a2.barh(ye, [b for _, b in pe], color=C[4])
        a2.set_yticks(ye); a2.set_yticklabels([a for a, _ in pe], fontsize=6)
        a2.axvline(e3['macro_auroc'], color=C[1], lw=1)
        a2.axvline(0.5, color='k', lw=0.6, ls=':')
        a2.set_xlabel('AUROC, held-out ligase'); a2.set_xlim(0, 1)
    save(fig, 'splits.pdf')

def representations():
    names = ('cell_1000', 'chemberta', 'morgan_chemberta', 'morgan_esm',
             'morgan_chemberta_esm', 'chemberta_p64', 'morgan_chemberta_p64',
             'morgan_esm_p64', 'morgan_chemberta_esm_p64')
    ds = [(n, load(f'{n}.json')) for n in names]
    ds = [(n, d) for n, d in ds if d]
    if len(ds) < 2: return
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    x = np.arange(len(ds))
    ax.bar(x, [d['macro_auroc'] for _, d in ds],
           yerr=[d['target_se'] for _, d in ds], capsize=2,
           color=[C[0]] + [C[2]] * (len(ds) - 1))
    ax.set_xticks(x); ax.set_xticklabels([n for n, _ in ds], rotation=30,
                                         ha='right', fontsize=7)
    ax.axhline(0.5, color='k', lw=0.6, ls=':')
    ax.set_ylabel('macro AUROC')
    save(fig, 'representations.pdf')

FIGURES = (protocol_collapse, factorial, baselines, representations,
           fewshot, sweep, label_noise,
           cohort_variants, calibration, similarity, overlap, splits)

def main():
    for f in FIGURES: f()

if __name__ == '__main__':
    main()
