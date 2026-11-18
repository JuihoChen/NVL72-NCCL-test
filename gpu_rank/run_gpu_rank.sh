#!/bin/bash
# run_gpu_rank.sh
#
# Rank all GPUs of the rack by NCCL point-to-point bandwidth from ONE mpirun
# launch (nccl_pairwise_rank: N-1 rounds, every GPU paired with every other).
# Prints every GPU slowest -> fastest, with each GPU's worst partner.
#
# Run ON node-00, from the run/ folder deploy_rack_nccl_test.sh staged
# (next to run_nccl_test.sh), with nccl_pairwise_rank.c alongside it.
# Run run_nccl_test.sh once since the last boot first: its pre-flight repairs
# nvidia-imex/channel0 and writes the IMEX node list this script reads. This
# script only CHECKS that state; it repairs nothing. run_nccl_test.sh and
# deploy_rack_nccl_test.sh are not touched.
#
#   bash run_gpu_rank.sh               # full rack
#   bash run_gpu_rank.sh 9             # first 9 nodes only (same rule as run_nccl_test.sh)
#   bash run_gpu_rank.sh -- -b 2G -n 20  # pass options through to nccl_pairwise_rank
#
# First run: builds bin/nccl_pairwise_rank from nccl_pairwise_rank.c with gcc
# (no CUDA/NCCL/MPI headers needed -- it dlopen()s the pack's libnccl and
# libcudart), then copies the binary to the same path on every node.
#
# The mpirun flags and NCCL env below are copied from run_nccl_test.sh's
# run_one() on purpose -- keep the two in sync, or the numbers stop being
# comparable. NCCL_COMM_ID is the only addition: it replaces MPI for the
# NCCL bootstrap (rank 0 listens there).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACK_DIR="$(dirname "$SCRIPT_DIR")"
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=8"
IMEX_CFG="/etc/nvidia-imex/nodes_config.cfg"
GPUS_PER_NODE=4
SRC="$SCRIPT_DIR/nccl_pairwise_rank.c"
BIN="$PACK_DIR/bin/nccl_pairwise_rank"
MPIRUN_BIN="$PACK_DIR/bin/mpirun"
COMM_PORT=29517

NODES_REQUESTED=""
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) NODES_REQUESTED="$1"; shift ;;
  esac
done

[[ -f "$IMEX_CFG" ]] || { echo "ERROR: $IMEX_CFG not found -- run 'bash $SCRIPT_DIR/run_nccl_test.sh 1' once first" >&2; exit 1; }
mapfile -t ALL_NODES < <(awk 'NF{print $1}' "$IMEX_CFG")
NODES_REQUESTED="${NODES_REQUESTED:-${#ALL_NODES[@]}}"
if ! [[ "$NODES_REQUESTED" =~ ^[0-9]+$ ]] || [[ $NODES_REQUESTED -lt 1 || $NODES_REQUESTED -gt ${#ALL_NODES[@]} ]]; then
  echo "ERROR: node_count must be 1..${#ALL_NODES[@]} (got: $NODES_REQUESTED)" >&2; exit 1
fi
NODES=("${ALL_NODES[@]:0:$NODES_REQUESTED}")
RUN_GPUS=$(( NODES_REQUESTED * GPUS_PER_NODE ))

MNNVL_UUID=$(grep -oE 'NCCL_MNNVL_UUID=0x[0-9A-Fa-f]+' "$SCRIPT_DIR/run_nccl_test.sh" 2>/dev/null | head -1 | cut -d= -f2)
[[ -n "$MNNVL_UUID" ]] || { echo "ERROR: could not read NCCL_MNNVL_UUID from $SCRIPT_DIR/run_nccl_test.sh" >&2; exit 1; }
[[ -f "$MPIRUN_BIN" ]] || { echo "ERROR: $MPIRUN_BIN not found" >&2; exit 1; }

# --- Build (node-00 only, once; rebuilt if the source is newer) -------------
if [[ ! -x "$BIN" || "$SRC" -nt "$BIN" ]]; then
  [[ -f "$SRC" ]] || { echo "ERROR: $SRC not found -- copy it next to this script" >&2; exit 1; }
  command -v gcc >/dev/null || { echo "ERROR: gcc not found on node-00. Build on any aarch64 Linux box with:" >&2
    echo "  gcc -O2 -std=gnu11 -o nccl_pairwise_rank nccl_pairwise_rank.c -ldl" >&2
    echo "and copy the result to $BIN" >&2; exit 1; }
  echo "=== Building $BIN ==="
  gcc -O2 -std=gnu11 -o "$BIN" "$SRC" -ldl || { echo "ERROR: build failed" >&2; exit 1; }
fi

# --- Push the binary to every node (parallel, only where it differs) --------
SUM=$(md5sum "$BIN" | cut -d' ' -f1)
echo "=== Checking ${#NODES[@]} node(s): binary, nvidia-imex, channel0, stale GPU processes ==="
declare -A PIDS
for ip in "${NODES[@]}"; do
  (
    remote=$(ssh -o BatchMode=yes $SSH_OPTS "root@${ip}" "md5sum '$BIN' 2>/dev/null | cut -d' ' -f1")
    if [[ "$remote" != "$SUM" ]]; then
      scp -q $SSH_OPTS "$BIN" "root@${ip}:$BIN" || { echo "  [$ip] copy FAILED"; exit 1; }
    fi
    msg=$(ssh -o BatchMode=yes $SSH_OPTS "root@${ip}" bash -s 2>&1 <<'REMOTE'
systemctl is-active --quiet nvidia-imex || { echo "nvidia-imex not active"; exit 1; }
[[ -e /dev/nvidia-caps-imex-channels/channel0 ]] || { echo "channel0 missing"; exit 1; }
STALE=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -E 'all_reduce_perf|alltoall_perf|nccl_pairwise_rank|orted|mpirun')
[[ -z "$STALE" ]] || { echo "stale GPU process: $STALE"; exit 1; }
REMOTE
    ) || { echo "  [$ip] NOT READY: ${msg:-ssh failed}"; exit 1; }
  ) &
  PIDS["$ip"]=$!
done
BAD=0
for ip in "${NODES[@]}"; do wait "${PIDS[$ip]}" || BAD=1; done
if [[ $BAD -ne 0 ]]; then
  echo "ERROR: some node(s) not ready -- run 'bash $SCRIPT_DIR/run_nccl_test.sh 1' (its pre-flight" >&2
  echo "       repairs imex/channel0 and clears stale processes), then retry." >&2
  exit 1
fi
echo "    all ${#NODES[@]} node(s) ready"

# --- Run --------------------------------------------------------------------
export OPAL_PREFIX="$PACK_DIR"
export PATH="$PACK_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$PACK_DIR/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

RUN_HOSTFILE="$SCRIPT_DIR/hostfile_gpu_rank_${NODES_REQUESTED}node.cfg"
printf '%s\n' "${NODES[@]}" > "$RUN_HOSTFILE"
NEXT_HOP=$(awk 'NR==2{print $1}' "$IMEX_CFG")
IFACE=$(ip route get "$NEXT_HOP" 2>/dev/null | sed -E 's/.*?dev (\S+) .*/\1/;t;d')
# rank 0 runs on the first hostfile entry (node-00); NCCL's bootstrap root listens there
COMM_ID="${NODES[0]}:${COMM_PORT}"
mkdir -p "$SCRIPT_DIR/results"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$SCRIPT_DIR/results/${STAMP}_${NODES_REQUESTED}node_gpu_rank.log"
CSV="$SCRIPT_DIR/results/${STAMP}_${NODES_REQUESTED}node_gpu_rank_matrix.csv"
echo "Using interface: $IFACE | NCCL_COMM_ID=$COMM_ID | $RUN_GPUS GPUs"

{
  echo "$(date) : GPU pairwise rank | $NODES_REQUESTED node(s) | $RUN_GPUS GPUs | launch $(hostname)"
  "$MPIRUN_BIN" \
    --mca schizo ompi \
    --mca pml ob1 \
    --mca btl tcp,self \
    --mca btl_tcp_if_include $IFACE \
    --mca coll_hcoll_enable 0 \
    --mca mca_base_component_show_load_errors 0 \
    -np $RUN_GPUS \
    -N $GPUS_PER_NODE \
    --hostfile $RUN_HOSTFILE \
    --bind-to none \
    --oversubscribe \
    --allow-run-as-root \
    -x OPAL_PREFIX="$PACK_DIR" \
    -x PATH="$PACK_DIR/bin:$PATH" \
    -x LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    -x NCCL_DEBUG=WARN \
    -x NCCL_SOCKET_IFNAME=$IFACE \
    -x NCCL_MNNVL_ENABLE=2 \
    -x NCCL_NVLS_ENABLE=1 \
    -x NCCL_P2P_DISABLE=0 \
    -x NCCL_IB_DISABLE=1 \
    -x CUDA_IPC_HANDLE_SHARING_SUPPORT=1 \
    -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -x NCCL_MNNVL_UUID=$MNNVL_UUID \
    -x NCCL_MIN_CTAS=32 \
    -x NCCL_P2P_SCHEDULE_GROUP_SIZE=1 \
    -x NCCL_COMM_ID=$COMM_ID \
    "$BIN" -o "$CSV" "${EXTRA[@]}"
  echo "$(date) : Done."
} 2>&1 | tee "$LOG"
echo ""
echo "Log saved on this node: $LOG"
