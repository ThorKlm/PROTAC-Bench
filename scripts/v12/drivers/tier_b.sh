cd "$(git rev-parse --show-toplevel)"
S=42,0,1,2,3,4,5,6,7,8
run () { echo "=== $1 ==="; shift; "$@"; echo "--- exit $? ---"; }
run "fewshot (MAML, ProtoNet, RF curves)" python3 -I signals/fewshot.py --seeds $S
run "rs_per_target_stan"                  python3 -I baselines/rs_per_target_stan.py
run "gnn_baselines (GIN, D-MPNN)"         python3 -I baselines/gnn_baselines.py --seeds $S
run "rs_per_target_gnn"                   python3 -I baselines/rs_per_target_gnn.py
run "rs_per_target_knn"                   python3 -I baselines/rs_per_target_knn.py
run "chemprop_hpo"                        python3 -I baselines/chemprop_hpo.py
echo "=== TIER B DONE ==="
