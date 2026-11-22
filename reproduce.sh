#!/bin/bash
# Regenerate PROTAC-Bench results and write results/REPRODUCTION.txt.
# Arms are selected by band: see reproduce.py --help.
set -e
if [ ! -f data/protac_bench.csv ]; then
  echo "data/protac_bench.csv not found. Fetch the release with:"
  echo "  huggingface-cli download ThorKl/protac-bench --repo-type dataset --local-dir ."
  exit 1
fi
exec "${PYTHON:-python3}" reproduce.py "$@"
