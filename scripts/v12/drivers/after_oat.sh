while kill -0 3334916 2>/dev/null; do sleep 60; done
cd "$(git rev-parse --show-toplevel)"
S=42,0,1,2,3,4,5,6,7,8
run () { echo "=== $1 ==="; shift; "$@"; echo "--- exit $? ---"; }
run "nonkinase" python3 -I robustness/nonkinase.py --seeds $S
run "cross_e3"  python3 -I robustness/cross_e3.py  --seeds $S
echo "=== AFTER OAT DONE ==="

cd "$(git rev-parse --show-toplevel)"
echo "=== temporal_prospective ==="
python3 -I scripts/v12/temporal_prospective.py
echo "--- exit $? ---"

cd "$(git rev-parse --show-toplevel)"
echo "=== oat with sklearn-default cell ==="
python3 -I scripts/v12/oat.py
echo "--- exit $? ---"
