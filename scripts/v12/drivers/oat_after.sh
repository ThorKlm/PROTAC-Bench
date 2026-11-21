while kill -0 2716909 2>/dev/null; do sleep 60; done
cd "$(git rev-parse --show-toplevel)"
echo "=== oat ten-seed, seven variants ==="
python3 -I scripts/v12/oat.py
echo "--- exit $? ---"
