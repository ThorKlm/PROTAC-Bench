cd "$(git rev-parse --show-toplevel)"
run () { echo "=== $1 ==="; shift; "$@"; echo "--- exit $? ---"; }
run "oat ten-seed, seven variants" python3 -I scripts/v12/oat.py
OMP_NUM_THREADS=1 bash scripts/v12/drivers/tier_a.sh
OMP_NUM_THREADS=1 bash scripts/v12/drivers/tier_b.sh
echo "=== TONIGHT DONE ==="
