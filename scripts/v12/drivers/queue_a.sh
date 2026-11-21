while kill -0 1963312 2>/dev/null; do sleep 60; done
OMP_NUM_THREADS=1 bash scripts/v12/drivers/tier_a.sh
