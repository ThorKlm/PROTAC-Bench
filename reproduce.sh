#!/bin/bash
# Reproduce core PROTAC-Bench experiments.
# Total runtime: ~2-3 hours on CPU.
set -e

echo '============================================='
echo '  PROTAC-Bench Reproduction'
echo '============================================='

mkdir -p results

echo ''
echo 'Step 1: Baseline RF+Morgan...'
python3 baselines/rf_morgan.py --seeds 42,43,44

echo ''
echo 'Step 2: Warhead transfer signal...'
python3 signals/warhead_transfer.py --seeds 42,43,44

echo ''
echo 'Step 3: ADMET cascade...'
python3 signals/admet_cascade.py

echo ''
echo 'Step 4: Few-shot (k=5)...'
python3 signals/fewshot.py --k 5

echo ''
echo 'Step 5: Full stack (best result)...'
python3 signals/full_stack.py --seeds 42,43,44

echo ''
echo 'Step 6: Robustness checks...'
python3 robustness/single_source.py
python3 robustness/nonkinase.py

echo ''
echo '============================================='
echo '  Done. Check results/ for outputs.'
echo '============================================='
