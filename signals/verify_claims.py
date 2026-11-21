#!/usr/bin/env python
"""One-line verification of the manuscript's artefact-backed claims."""
import json, os, glob, csv, statistics as st

PAIRS = {
    'W': [('1000', '1100'), ('1001', '1101'), ('1010', '1110'), ('1011', '1111')],
    'A': [('1000', '1010'), ('1001', '1011'), ('1100', '1110'), ('1101', '1111')],
    'K': [('1000', '1001'), ('1010', '1011'), ('1100', '1101'), ('1110', '1111')],
    'M': [('0010', '1010'), ('0011', '1011'), ('0110', '1110'), ('0111', '1111')],
}

def g(p):
    try: return json.load(open(p))
    except Exception as e: return {'__err': str(e)[:40]}

def s(n, *a): print('%02d' % n, *a)

def r4(x): return round(x, 4) if isinstance(x, (int, float)) else x

d = g('results/confidence_intervals.json')
c0 = d.get('C0_morgan_only', {})
s(1, 'C0 anchor', c0.get('mean_auroc'), c0.get('std'), c0.get('ci95_low'), c0.get('ci95_high'))
s(2, 'cumulative', [r4(d[k]['mean_auroc']) for k in
     ('C1_morgan+warhead', 'C2_morgan+warhead+admet7', 'C3_morgan+warhead+admet7+k5') if k in d])
d = g('results/v12/metrics_pass.json')
s(3, 'mAP prior lift med npos', d.get('mean_average_precision'), d.get('macro_per_target_prior'),
     d.get('mean_lift'), d.get('median_lift'), d.get('n_positive_lift'))
d = g('results/v12/protocol_rcv_scaffold.json')
s(4, 'rf rcv/scaffold pooled', d.get('random_cv_summary', {}).get('pooled_mean'),
     d.get('scaffold_cv_summary', {}).get('pooled_mean'))
d = g('results/v12/xgb_protocols.json')
s(5, 'xgb rcv/scaf/loto', d.get('random_cv', {}).get('pooled_mean'),
     d.get('scaffold_cv', {}).get('pooled_mean'), d.get('loto', {}).get('macro_mean'))
d = g('results/v12/oat.json')
s(6, 'oat', {k: r4(v.get('mean_auroc')) for k, v in d.items() if isinstance(v, dict) and 'mean_auroc' in v})
d = g('results/plm_scaling.json')
b = d.get('morgan_only', {}).get('mean_auroc')
s(7, 'plm deltas', {k: r4(v['mean_auroc'] - b) for k, v in d.items()
                    if isinstance(v, dict) and 'mean_auroc' in v and k != 'morgan_only'})
d = g('results/lofo.json')
s(8, 'lofo', d.get('n_targets'), d.get('n_families'), d.get('lofo_mean_auroc'),
     d.get('delta_lofo_minus_loto'), d.get('wilcoxon_pval'))
want = ('Bromodomain', 'HDAC', 'Phosphatase', 'Kinase', 'STAT', 'PARP', 'IAP', 'GPX4', 'HCFC1')
s(9, 'lofo families', {f['family']: r4(f['mean_lofo']) for f in d.get('per_family_summary', [])
                       if f['family'] in want})
d = g('results/nonkinase.json').get('conditions', {})
s(10, 'nonkinase', {k: (v['n_targets'], v['morgan_auroc'], v['full_auroc'], v['wilcoxon_p'])
                    for k, v in d.items()})
d = g('results/cross_e3.json').get('conditions', {})
s(11, 'cross_e3 means', {k: v.get('mean_auroc') for k, v in d.items()})
s(12, 'cross_e3 sds', {k: r4(st.stdev(list(v['aurocs_by_seed'].values())))
                       for k, v in d.items() if 'aurocs_by_seed' in v})
d = g('results/temporal_prospective.json')
s(13, 'year split', d.get('macro'), d.get('target_se'), d.get('n_train_rows'),
     d.get('n_test_rows'), d.get('n_test_targets_evaluable'), d.get('n_test_targets_total'))
d = g('results/temporal.json')
s(14, 'temporal.json cross-source', d.get('split_info', {}).get('train_source'),
     d.get('split_info', {}).get('test_source'), len(d.get('C0_morgan_only', {}).get('aurocs', [])))
d = g('results/single_source.json')
s(15, 'single_source', d.get('_meta', {}).get('seeds'),
     {k: (v.get('n_loto_targets'), v.get('mean_auroc')) for k, v in d.items()
      if isinstance(v, dict) and ('n_loto_targets' in v)})
s(16, 'warhead target p', g('results/warhead_transfer.json').get('p_value'))
d = g('results/warhead_ablation.json')
s(17, 'warhead ablation p', d.get('p_C2_vs_C0'), d.get('p_C2_vs_C3'))
s(18, 'warhead lofo p', g('results/warhead_lofo.json').get('p_value'))
d = g('results/gnn_baselines.json')
s(19, 'gnn', {k: (r4(v.get('mean_auroc')), r4(v.get('std_auroc')))
              for k, v in d.items() if isinstance(v, dict)})
d = g('results/pending/factorial_canonical.json')
s(20, 'factorial', d.get('meta', {}).get('seeds'), len(d.get('cells', [])))
cells = {e['id']: e['mean_auroc'] for e in d.get('cells', []) if e.get('mean_auroc') is not None}
for nm, pr in PAIRS.items():
    if all(a in cells and b in cells for a, b in pr):
        s(21, 'marginal ' + nm, r4(sum(cells[b] - cells[a] for a, b in pr) / len(pr)))
s(22, 'tiers', [r4(cells.get(k)) for k in ('1000', '1010', '1011')], 'cell 1111', r4(cells.get('1111')))
d = g('results/structure_ladder.json')
s(23, 'ladder baseline / smina', d.get('baseline_morgan'), str(d.get('smina_docking_scores'))[:60])
d = g('results/egnn_10seed.json').get('egnn_seeds', {})
s(24, 'egnn seeds', sorted(d), 'mean', r4(st.mean(list(d.values()))) if d else None)
d = g('results/pocket_shuffle_control.json')
s(25, 'pocket_shuffle seeds', d.get('seeds') or d.get('_meta', {}).get('seeds'))
d = g('results/stan_combined.json')
s(26, 'stan C0', d.get('conditions', {}).get('C0', {}).get('mean_auroc'), list(d.get('conditions', {})))
s(27, 'deepprotacs seeds', g('results/deepprotacs_protacdb_replication.json').get('seeds'))
s(28, 'dm_loto seeds', g('results/dm_loto.json').get('seeds'))
d = g('results/task7_confound_detection.json')
s(29, 'task7', d.get('seeds'), str(d.get('summary') or d.get('macro_ovr') or list(d))[:70])
d = g('results/task14_within_target_cross_lab.json')
s(30, 'task14', d.get('config', {}).get('seeds'), str(d.get('summary'))[:90])
s(31, 'task22 seeds', g('results/task22_gap_decomposition.json').get('seeds'))
for n, p in ((32, 'results/calibration.json'), (33, 'results/spectra_sweep.json'),
             (34, 'results/label_noise.json'), (35, 'results/exp40_hpo_v2/hpo_v2_validation.json')):
    s(n, 'exists', p, os.path.exists(p))
d = g('results/fewshot.json')
s(36, 'fewshot stub', d.get('seeds'), d.get('k_values'))
s(37, 'baseline_rf_morgan seeds', g('results/baseline_rf_morgan.json').get('seeds'))
prog = []
for p in sorted(glob.glob('results/pending/cascade_seed_*.json')):
    q = g(p); prog.append((q.get('seed'), len(q.get('folds', {})), q.get('status')))
s(38, 'cascade', prog)
rows = list(csv.reader(open('data/protac_bench.csv')))
s(39, 'csv', rows[0], len(rows) - 1)
s(40, 'dc50/dmax populated', sum(1 for x in rows[1:] if (x[4] or '').strip()),
     sum(1 for x in rows[1:] if (x[5] or '').strip()), 'source col', 'source' in rows[0])
d = g('data/lofo_folds.json').get('_meta', {})
s(41, 'lofo_folds', d.get('n_families'), d.get('n_targets'), d.get('unmapped_singletons_excluded'))
d = g('data/cross_lab_folds.json')
s(42, 'cross_lab', d['_meta'].get('n_targets'),
     sum(len(v['papers']) for k, v in d.items() if k != '_meta'),
     str(d['_meta'].get('fold_index_basis'))[:34])
d = g('data/temporal_prospective_folds.json').get('_meta', {})
s(43, 'temporal folds', d.get('n_rows_with_pub_year'), d.get('n_rows_excluded_year_2023_gap'),
     d.get('n_train'), d.get('n_test'))
d = g('data/croissant.json')
s(44, 'local croissant', d.get('conformsTo'),
     'plural' if 'rai:dataLimitations' in d else 'singular', str(d.get('citeAs'))[42:78])
