#!/bin/bash
declare -A f=( [cascade]=results/admet_cascade.json
               [plm]=results/plm_scaling.json
               [rcv]=results/v12/protocol_rcv_scaffold.json )
declare -A t0
for k in "${!f[@]}"; do t0[$k]=$( [ -f "${f[$k]}" ] && stat -c %Y "${f[$k]}" || echo 0 ); done
echo "baseline mtimes: $(for k in "${!t0[@]}"; do printf '%s=%s ' "$k" "${t0[$k]}"; done)"
while [ ${#f[@]} -gt 0 ]; do
  for k in "${!f[@]}"; do
    p=${f[$k]}
    if [ -f "$p" ] && [ "$(stat -c %Y "$p")" -gt "${t0[$k]}" ]; then
      echo "[$(date -u +%H:%M)] $k WROTE: $(stat -c %s "$p") B"
      unset 'f[$k]'
    fi
  done
  sleep 60
done
echo "[$(date -u +%H:%M)] all three written"
