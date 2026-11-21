cd "$(git rev-parse --show-toplevel)"
S=42,0,1,2,3,4,5,6,7,8
run () { echo "=== $1 ==="; shift; "$@"; echo "--- exit $? ---"; }
run "lofo"             python3 -I robustness/lofo.py --seeds $S
run "nonkinase"        python3 -I robustness/nonkinase.py --seeds $S
run "cross_e3"         python3 -I robustness/cross_e3.py --seeds $S
run "temporal"         python3 -I robustness/temporal.py --seeds $S
run "warhead_transfer" python3 -I signals/warhead_transfer.py --seeds $S
run "warhead_ablation" python3 -I signals/warhead_ablation.py --seeds $S
run "warhead_lofo"     python3 -I signals/warhead_lofo.py --seeds $S
run "xgboost_morgan"   python3 -I baselines/xgboost_morgan.py --seeds $S
echo "=== TIER A DONE ==="
