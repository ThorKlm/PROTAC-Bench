cd "$(git rev-parse --show-toplevel)"
run () { echo "=== $1 ==="; shift; "$@"; echo "--- exit $? ---"; }
run "metrics_pass: anchor, AUPRC, predictions"  python3 -I scripts/v12/metrics_pass.py
run "calib: temperature vs platt"               python3 -I scripts/v12/calib.py
run "cohort_cells: the floor"                   python3 -I scripts/v12/cohort_cells.py
run "confidence_intervals: four-condition chain" python3 -I robustness/confidence_intervals.py --morgan-bits 2048 --trees 200
run "xgb_protocols: second model family"        python3 -I scripts/v12/xgb_protocols.py
run "oat: configuration insensitivity"          python3 -I scripts/v12/oat.py
run "admet_cascade"                             python3 -I signals/admet_cascade.py --seeds 42,0,1,2,3,4,5,6,7,8 --output results/admet_cascade.json
echo "=== ALL DONE ==="
