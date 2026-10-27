#!/bin/bash
# collect_nccl_info.sh
#
# Run ON node-00 after an INFO run (NCCL_DEBUG=INFO with
# NCCL_DEBUG_FILE=/tmp/nccl_info/%h.%p.log). Pulls every node's
# /tmp/nccl_info/*.log back IN PARALLEL into one folder on node-00, then
# ranks all GPUs (worst first) with rank_gpus_from_info.py.
#
# The per-rank files stay where they are on each node: /tmp is cleared at
# reboot, and the next INFO run empties /tmp/nccl_info first, so nothing
# piles up across runs.
#
#   bash collect_nccl_info.sh                      # nodes from the IMEX config
#   bash collect_nccl_info.sh --nodes ip1,ip2,...  # explicit node list
#   bash collect_nccl_info.sh --src /tmp/nccl_info # different remote folder

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=8"
IMEX_CFG="/etc/nvidia-imex/nodes_config.cfg"
SRC="/tmp/nccl_info"
NODES_RAW=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nodes) NODES_RAW="$2"; shift 2 ;;
    --src)   SRC="$2"; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

NODES=()
if [[ -n "$NODES_RAW" ]]; then
  IFS=',' read -ra NODES <<< "$NODES_RAW"
else
  [[ -f "$IMEX_CFG" ]] || { echo "ERROR: $IMEX_CFG not found -- pass --nodes" >&2; exit 1; }
  mapfile -t NODES < <(awk 'NF{print $1}' "$IMEX_CFG")
fi

OUT_DIR="$SCRIPT_DIR/results/nccl_info_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"
echo "=== Collecting ${SRC}/*.log from ${#NODES[@]} node(s) -> $OUT_DIR ==="

declare -A PIDS
for ip in "${NODES[@]}"; do
  (
    mkdir -p "$OUT_DIR/$ip"
    ssh -o BatchMode=yes $SSH_OPTS "root@${ip}" \
      "cd '$SRC' 2>/dev/null && ls *.log >/dev/null 2>&1 && tar cf - *.log" \
      | tar xf - -C "$OUT_DIR/$ip" 2>/dev/null
  ) &
  PIDS["$ip"]=$!
done

MISSING=()
for ip in "${NODES[@]}"; do
  wait "${PIDS[$ip]}"
  n=$(find "$OUT_DIR/$ip" -name '*.log' | wc -l)
  if [[ $n -eq 0 ]]; then MISSING+=("$ip"); echo "  [$ip] no files"; else echo "  [$ip] $n file(s)"; fi
done
[[ ${#MISSING[@]} -eq 0 ]] || echo "WARNING: nothing collected from: ${MISSING[*]} (unreachable, or that node never ran with INFO)" >&2

echo ""
python3 "$SCRIPT_DIR/rank_gpus_from_info.py" "$OUT_DIR" --expect $(( ${#NODES[@]} * 4 )) --csv "$OUT_DIR/gpu_ranking.csv" \
  | tee "$OUT_DIR/gpu_ranking.txt"
