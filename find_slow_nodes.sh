#!/bin/bash
# find_slow_nodes.sh
#
# Pinpoint which GPU node(s) in a GB300 NVL72 rack drag NCCL bandwidth
# down. Run ON node-00, from the run/ folder that deploy_rack_nccl_test.sh
# staged (next to run_nccl_test.sh), AFTER run_nccl_test.sh has run at
# least once since the last reboot -- this script reuses that pre-flight's
# work (nvidia-imex active, channel0 present, IMEX nodes_config written,
# SSH mesh OK) and only re-checks it cheaply; it never repairs anything.
#
# Why this exists: run_nccl_test.sh N always tests the FIRST N nodes of
# the hostfile, so a slow full-rack result (below ~927 GB/s All-Reduce /
# ~670 GB/s All-to-All out-of-place busbw at the top of the sweep) says
# the rack is slow but not WHICH tray is. This script runs many small,
# short jobs over chosen node subsets and compares them against each other:
#
#   single : one 4-GPU job per node (18 jobs on an NVL72). Catches a bad
#            GPU / tray-local NVLink problem.
#   pair   : 2-node (8-GPU) jobs in a ring: n0+n1, n1+n2, ..., n17+n0.
#            Every node is in exactly two pairs; a node whose BOTH pairs
#            are slow is the suspect. Catches cross-tray problems a single
#            node can't see.
#   loo    : leave-one-out, (N-1)-node jobs each dropping one node. Use when
#            the slowdown only shows at full scale. The job that recovers
#            to well above its peers names the culprit. Slowest phase.
#
# Scoring is RELATIVE (each job vs. the median of its phase), not against
# the 72-GPU 927/670 GB/s baseline -- 4- and 8-GPU jobs have different
# healthy numbers, and NVLS behaves differently at small rank counts. The
# metric is the peak OUT-OF-PLACE busbw over the tested sizes (same column
# NV's acceptance spec uses -- see the design doc). Any "#wrong" > 0, a
# non-zero exit, or a timeout is a hard FAIL regardless of bandwidth.
#
# Jobs run strictly one at a time: concurrent jobs would share the
# NVSwitch fabric and contaminate each other's numbers.
#
#   bash find_slow_nodes.sh                       # single + pair (default)
#   bash find_slow_nodes.sh --phases single       # quickest triage
#   bash find_slow_nodes.sh --phases loo          # full-scale-only slowdown
#   bash find_slow_nodes.sh --threshold 0.85      # flag below 85% of median
#   bash find_slow_nodes.sh --nodes 10.0.0.5,10.0.0.6,10.0.0.7   # subset
#   bash find_slow_nodes.sh --coll allreduce      # skip All-to-All
#
# mpirun flags and NCCL env are copied from run_nccl_test.sh's run_one()
# on purpose -- keep the two in sync, or results stop being comparable.

set -uo pipefail

SCRIPT_VERSION="0.1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=8"
IMEX_CFG="/etc/nvidia-imex/nodes_config.cfg"
GPUS_PER_NODE=4

PHASES="single,pair"
COLLS="allreduce,alltoall"
THRESHOLD="0.90"
MIN_SIZE="1G"
MAX_SIZE="8G"
ITERS=20
WARMUP=5
JOB_TIMEOUT=300
NODES_RAW=""
MNNVL_UUID=""
PACK_DIR=""
PREFLIGHT=1

usage() {
  cat <<EOF
Usage: $0 [options]
  --phases LIST     comma list of single,pair,loo        (default: ${PHASES})
  --coll LIST       comma list of allreduce,alltoall     (default: ${COLLS})
  --threshold F     flag a job below F x phase median     (default: ${THRESHOLD})
  --min-size SZ     smallest message size, nccl-tests -b  (default: ${MIN_SIZE})
  --max-size SZ     largest message size, nccl-tests -e   (default: ${MAX_SIZE})
  --iters N         timed iterations per size, -n         (default: ${ITERS})
  --timeout SEC     kill a single job after SEC seconds   (default: ${JOB_TIMEOUT})
  --nodes ip,ip,... test only these nodes                 (default: ${IMEX_CFG})
  --uuid 0xNN       NCCL_MNNVL_UUID (default: read from ./run_nccl_test.sh)
  --pack-dir PATH   pack location   (default: parent of this script's folder)
  --no-preflight    skip the per-node imex/channel0/stale-process check
  --version
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phases)       PHASES="$2"; shift 2 ;;
    --coll)         COLLS="$2"; shift 2 ;;
    --threshold)    THRESHOLD="$2"; shift 2 ;;
    --min-size)     MIN_SIZE="$2"; shift 2 ;;
    --max-size)     MAX_SIZE="$2"; shift 2 ;;
    --iters)        ITERS="$2"; shift 2 ;;
    --timeout)      JOB_TIMEOUT="$2"; shift 2 ;;
    --nodes)        NODES_RAW="$2"; shift 2 ;;
    --uuid)         MNNVL_UUID="$2"; shift 2 ;;
    --pack-dir)     PACK_DIR="$2"; shift 2 ;;
    --no-preflight) PREFLIGHT=0; shift ;;
    --version)      echo "find_slow_nodes.sh version ${SCRIPT_VERSION}"; exit 0 ;;
    -h|--help)      usage; exit 0 ;;
    *)              echo "Unknown arg: $1" >&2; usage >&2; exit 1 ;;
  esac
done

# --- Resolve pack, UUID and node list ---------------------------------------
# deploy_rack_nccl_test.sh stages run/ directly under the data dir, so the
# pack (bin/ + lib/) is this script's parent folder.
PACK_DIR="${PACK_DIR:-$(dirname "$SCRIPT_DIR")}"
PACK_DIR="${PACK_DIR%/}"
MPIRUN_BIN="$PACK_DIR/bin/mpirun"
for f in mpirun all_reduce_perf alltoall_perf; do
  [[ -f "$PACK_DIR/bin/$f" ]] || { echo "ERROR: $PACK_DIR/bin/$f not found (use --pack-dir)" >&2; exit 1; }
done

if [[ -z "$MNNVL_UUID" && -f "$SCRIPT_DIR/run_nccl_test.sh" ]]; then
  MNNVL_UUID=$(grep -oE 'NCCL_MNNVL_UUID=0x[0-9A-Fa-f]+' "$SCRIPT_DIR/run_nccl_test.sh" | head -1 | cut -d= -f2)
fi
[[ -n "$MNNVL_UUID" ]] || { echo "ERROR: could not read NCCL_MNNVL_UUID from $SCRIPT_DIR/run_nccl_test.sh -- pass --uuid" >&2; exit 1; }

NODES=()
if [[ -n "$NODES_RAW" ]]; then
  IFS=',' read -ra NODES <<< "$NODES_RAW"
else
  [[ -f "$IMEX_CFG" ]] || { echo "ERROR: $IMEX_CFG not found -- run run_nccl_test.sh once first, or pass --nodes" >&2; exit 1; }
  mapfile -t NODES < <(awk 'NF{print $1}' "$IMEX_CFG")
fi
N=${#NODES[@]}
[[ $N -ge 1 ]] || { echo "ERROR: empty node list" >&2; exit 1; }

IFS=',' read -ra PHASE_LIST <<< "$PHASES"
IFS=',' read -ra COLL_LIST <<< "$COLLS"
for p in "${PHASE_LIST[@]}"; do
  case "$p" in single|pair|loo) ;; *) echo "ERROR: unknown phase '$p'" >&2; exit 1 ;; esac
  [[ "$p" == pair && $N -lt 3 ]] && { echo "ERROR: pair phase needs >= 3 nodes" >&2; exit 1; }
  [[ "$p" == loo  && $N -lt 3 ]] && { echo "ERROR: loo phase needs >= 3 nodes" >&2; exit 1; }
done
for c in "${COLL_LIST[@]}"; do
  case "$c" in allreduce|alltoall) ;; *) echo "ERROR: unknown collective '$c'" >&2; exit 1 ;; esac
done

# Same interface selection as run_nccl_test.sh: route to the 2nd node.
NEXT_HOP="${NODES[1]:-${NODES[0]}}"
IFACE=$(ip route get "$NEXT_HOP" 2>/dev/null | sed -E 's/.*?dev (\S+) .*/\1/;t;d')
[[ -n "$IFACE" ]] || { echo "ERROR: could not determine network interface toward $NEXT_HOP" >&2; exit 1; }

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$SCRIPT_DIR/results/find_slow_nodes_${STAMP}"
mkdir -p "$OUT_DIR/jobs"
CSV="$OUT_DIR/results.csv"
echo "phase,collective,job,nodes,gpus,peak_busbw_gbs,status,log" > "$CSV"

export OPAL_PREFIX="$PACK_DIR"
export PATH="$PACK_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$PACK_DIR/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

echo "=== find_slow_nodes.sh v${SCRIPT_VERSION} | ${N} node(s) | phases=${PHASES} | coll=${COLLS} | threshold=${THRESHOLD} | sizes ${MIN_SIZE}..${MAX_SIZE} | UUID=${MNNVL_UUID} | iface=${IFACE} ==="
echo "    output: $OUT_DIR"

# --- Pre-flight: check only, never repair -----------------------------------
# A stale process or missing channel0 on one node would show up as a slow
# or failed job and get blamed on the wrong thing, so refuse to start.
if [[ $PREFLIGHT -eq 1 ]]; then
  echo "=== Pre-flight (check only): SSH, nvidia-imex, channel0, stale GPU processes ==="
  BAD=()
  for ip in "${NODES[@]}"; do
    if ! msg=$(ssh -o BatchMode=yes $SSH_OPTS "root@${ip}" bash -s 2>&1 <<'REMOTE'
systemctl is-active --quiet nvidia-imex || { echo "nvidia-imex not active"; exit 1; }
[[ -e /dev/nvidia-caps-imex-channels/channel0 ]] || { echo "channel0 missing"; exit 1; }
STALE=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -E 'all_reduce_perf|alltoall_perf|orted|mpirun')
[[ -z "$STALE" ]] || { echo "stale GPU process: $STALE"; exit 1; }
exit 0
REMOTE
    ); then
      echo "  [$ip] NOT READY: ${msg:-ssh failed}" >&2
      BAD+=("$ip")
    fi
  done
  if [[ ${#BAD[@]} -gt 0 ]]; then
    echo "ERROR: ${#BAD[@]} node(s) not ready. Run 'bash $SCRIPT_DIR/run_nccl_test.sh 1' once" >&2
    echo "       (its pre-flight repairs imex/channel0 and clears stale processes on" >&2
    echo "       every node), then retry. --no-preflight skips this check." >&2
    exit 1
  fi
  echo "    all ${N} node(s) ready"
fi

# --- One job ----------------------------------------------------------------
# Prints "<peak_busbw> <status>" where status is OK | WRONG | FAIL | TIMEOUT.
# Peak busbw = max of the out-of-place busbw column (field 8 of each size
# row in nccl-tests output), across all sizes tested.
run_job() {
  local coll="$1" log="$2"; shift 2
  local hosts=("$@")
  local np=$(( ${#hosts[@]} * GPUS_PER_NODE ))
  local hostfile="${log%.log}.hosts"
  printf '%s\n' "${hosts[@]}" > "$hostfile"

  local bin args
  if [[ "$coll" == allreduce ]]; then
    bin="$PACK_DIR/bin/all_reduce_perf"; args="-b $MIN_SIZE -e $MAX_SIZE -f 2 -g 1 -n $ITERS -w $WARMUP"
  else
    bin="$PACK_DIR/bin/alltoall_perf";   args="-d uint8 -b $MIN_SIZE -e $MAX_SIZE -f 2 -n $ITERS -w $WARMUP"
  fi

  # shellcheck disable=SC2086
  timeout --kill-after=15 "$JOB_TIMEOUT" "$MPIRUN_BIN" \
    --mca schizo ompi \
    --mca pml ob1 \
    --mca btl tcp,self \
    --mca btl_tcp_if_include "$IFACE" \
    --mca coll_hcoll_enable 0 \
    --mca mca_base_component_show_load_errors 0 \
    -np "$np" \
    -N "$GPUS_PER_NODE" \
    --hostfile "$hostfile" \
    --bind-to none \
    --oversubscribe \
    --allow-run-as-root \
    -x OPAL_PREFIX="$PACK_DIR" \
    -x PATH="$PATH" \
    -x LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    -x NCCL_DEBUG=WARN \
    -x NCCL_SOCKET_IFNAME="$IFACE" \
    -x NCCL_MNNVL_ENABLE=2 \
    -x NCCL_NVLS_ENABLE=1 \
    -x NCCL_P2P_DISABLE=0 \
    -x NCCL_IB_DISABLE=1 \
    -x CUDA_IPC_HANDLE_SHARING_SUPPORT=1 \
    -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -x NCCL_MNNVL_UUID="$MNNVL_UUID" \
    -x NCCL_MIN_CTAS=32 \
    -x NCCL_P2P_SCHEDULE_GROUP_SIZE=1 \
    "$bin" $args > "$log" 2>&1
  local rc=$?

  local parsed peak wrong
  parsed=$(awk '
    $1 ~ /^[0-9]+$/ && NF >= 13 && $8 ~ /^[0-9.]+$/ {
      if ($8 > peak) peak = $8
      if ($9 ~ /^[0-9]+$/ && $9 > 0) wrong = 1
      if ($13 ~ /^[0-9]+$/ && $13 > 0) wrong = 1
      rows++
    }
    END { printf "%.2f %d %d\n", peak + 0, wrong + 0, rows + 0 }' "$log")
  read -r peak wrong rows <<< "$parsed"

  if [[ $rc -eq 124 || $rc -eq 137 ]]; then echo "0.00 TIMEOUT"
  elif [[ $rc -ne 0 || $rows -eq 0 ]];  then echo "0.00 FAIL"
  elif [[ $wrong -ne 0 ]];              then echo "$peak WRONG"
  else                                       echo "$peak OK"
  fi
}

# --- Job lists per phase ----------------------------------------------------
# Each entry: "<job label>|<space-separated node indexes>"
jobs_for_phase() {
  local phase="$1" i j k idx
  case "$phase" in
    single)
      for ((i=0; i<N; i++)); do echo "n${i}|${i}"; done ;;
    pair)
      for ((i=0; i<N; i++)); do j=$(( (i+1) % N )); echo "n${i}+n${j}|${i} ${j}"; done ;;
    loo)
      for ((i=0; i<N; i++)); do
        idx=""
        for ((k=0; k<N; k++)); do [[ $k -ne $i ]] && idx+="$k "; done
        echo "without-n${i}|${idx% }"
      done ;;
  esac
}

# --- Run everything ---------------------------------------------------------
declare -A PEAK STATUS   # key: "<phase>,<coll>,<job label>"
declare -A MED           # key: "<phase>,<coll>" -> median of OK jobs
TOTAL_JOBS=0
for p in "${PHASE_LIST[@]}"; do TOTAL_JOBS=$(( TOTAL_JOBS + N * ${#COLL_LIST[@]} )); done
JOB_NO=0

for phase in "${PHASE_LIST[@]}"; do
  for coll in "${COLL_LIST[@]}"; do
    echo "=== Phase ${phase} / ${coll} ==="
    while IFS='|' read -r label idxs; do
      JOB_NO=$(( JOB_NO + 1 ))
      hosts=()
      for i in $idxs; do hosts+=("${NODES[$i]}"); done
      log="$OUT_DIR/jobs/${phase}_${coll}_${label}.log"
      read -r peak status < <(run_job "$coll" "$log" "${hosts[@]}")
      PEAK["$phase,$coll,$label"]=$peak
      STATUS["$phase,$coll,$label"]=$status
      printf '  [%3d/%d] %-14s %-10s %3d GPUs  peak busbw %8s GB/s  %s\n' \
        "$JOB_NO" "$TOTAL_JOBS" "$label" "$coll" "$(( ${#hosts[@]} * GPUS_PER_NODE ))" "$peak" "$status"
      echo "${phase},${coll},${label},${hosts[*]},$(( ${#hosts[@]} * GPUS_PER_NODE )),${peak},${status},${log}" >> "$CSV"
    done < <(jobs_for_phase "$phase")
  done
done

# --- Scoring ----------------------------------------------------------------
median() { printf '%s\n' "$@" | sort -g | awk '{a[NR]=$1} END{ if (NR%2) print a[(NR+1)/2]; else printf "%.2f\n", (a[NR/2]+a[NR/2+1])/2 }'; }
below() { awk -v v="$1" -v m="$2" -v t="$THRESHOLD" 'BEGIN{exit !(v < m*t)}'; }
above() { awk -v v="$1" -v m="$2" -v t="$THRESHOLD" 'BEGIN{exit !(v*t > m)}'; }
pct()   { awk -v v="$1" -v m="$2" 'BEGIN{ if (m > 0) printf "%d%%", v*100/m; else print "-" }'; }

declare -A FLAGS   # key: node index -> accumulated reasons
flag() { FLAGS[$1]="${FLAGS[$1]:-}${FLAGS[$1]:+; }$2"; }

for phase in "${PHASE_LIST[@]}"; do
  for coll in "${COLL_LIST[@]}"; do
    ok_vals=()
    for ((i=0; i<N; i++)); do
      label=$(jobs_for_phase "$phase" | sed -n "$((i+1))p" | cut -d'|' -f1)
      [[ "${STATUS[$phase,$coll,$label]}" == OK ]] && ok_vals+=("${PEAK[$phase,$coll,$label]}")
    done
    if [[ ${#ok_vals[@]} -eq 0 ]]; then
      echo "WARNING: phase ${phase}/${coll}: no job succeeded -- nothing to compare against" >&2
      med=0
    else
      med=$(median "${ok_vals[@]}")
    fi
    MED["$phase,$coll"]=$med

    case "$phase" in
      single)
        for ((i=0; i<N; i++)); do
          s="${STATUS[$phase,$coll,n$i]}"; v="${PEAK[$phase,$coll,n$i]}"
          if [[ "$s" != OK ]]; then flag "$i" "single ${coll} ${s}"
          elif below "$v" "$med"; then flag "$i" "single ${coll} ${v} GB/s ($(pct "$v" "$med") of median ${med})"
          fi
        done ;;
      pair)
        # A node is suspect only if BOTH of its pairs are bad; one bad pair
        # is blamed on whichever neighbour is also bad in its other pair.
        for ((i=0; i<N; i++)); do
          prev=$(( (i-1+N) % N )); next=$(( (i+1) % N ))
          lp="n${prev}+n${i}"; ln="n${i}+n${next}"
          bad_p=0; bad_n=0
          if [[ "${STATUS[$phase,$coll,$lp]}" != OK ]] || below "${PEAK[$phase,$coll,$lp]}" "$med"; then bad_p=1; fi
          if [[ "${STATUS[$phase,$coll,$ln]}" != OK ]] || below "${PEAK[$phase,$coll,$ln]}" "$med"; then bad_n=1; fi
          if [[ $bad_p -eq 1 && $bad_n -eq 1 ]]; then
            flag "$i" "pair ${coll}: ${lp}=${PEAK[$phase,$coll,$lp]} ${STATUS[$phase,$coll,$lp]}, ${ln}=${PEAK[$phase,$coll,$ln]} ${STATUS[$phase,$coll,$ln]} (median ${med})"
          fi
        done ;;
      loo)
        # Dropping the culprit should RAISE bandwidth above its peers; a
        # failed job without node i is also telling (the fault is elsewhere).
        for ((i=0; i<N; i++)); do
          s="${STATUS[$phase,$coll,without-n$i]}"; v="${PEAK[$phase,$coll,without-n$i]}"
          if [[ "$s" == OK ]] && above "$v" "$med"; then
            flag "$i" "loo ${coll}: run without it = ${v} GB/s ($(pct "$v" "$med") of median ${med})"
          fi
        done ;;
    esac
  done
done

# --- Report -----------------------------------------------------------------
REPORT="$OUT_DIR/summary.txt"
{
  echo ""
  echo "=== Per-node summary (threshold ${THRESHOLD} x phase median, out-of-place busbw peak over ${MIN_SIZE}..${MAX_SIZE}) ==="
  for phase in "${PHASE_LIST[@]}"; do
    for coll in "${COLL_LIST[@]}"; do
      printf '  median %-6s %-10s %8s GB/s\n' "$phase" "$coll" "${MED[$phase,$coll]}"
    done
  done
  echo ""
  if [[ " ${PHASE_LIST[*]} " == *" single "* ]]; then
    printf '  %-5s %-16s' "node" "ip"
    for coll in "${COLL_LIST[@]}"; do printf ' %14s' "single-${coll}"; done
    printf '  %s\n' "verdict"
  else
    printf '  %-5s %-16s  %s\n' "node" "ip" "verdict"
  fi
  SUSPECTS=0
  for ((i=0; i<N; i++)); do
    printf '  n%-4s %-16s' "$i" "${NODES[$i]}"
    if [[ " ${PHASE_LIST[*]} " == *" single "* ]]; then
      for coll in "${COLL_LIST[@]}"; do
        s="${STATUS[single,$coll,n$i]}"
        [[ "$s" == OK ]] && printf ' %14s' "${PEAK[single,$coll,n$i]}" || printf ' %14s' "$s"
      done
    fi
    if [[ -n "${FLAGS[$i]:-}" ]]; then
      printf '  SUSPECT  %s\n' "${FLAGS[$i]}"
      SUSPECTS=$(( SUSPECTS + 1 ))
    else
      printf '  PASS\n'
    fi
  done
  echo ""
  if [[ $SUSPECTS -eq 0 ]]; then
    echo "No node stands out at threshold ${THRESHOLD}. If the full-rack run is still slow,"
    echo "try --phases loo (full-scale-only issues) or a stricter --threshold."
  else
    echo "${SUSPECTS} suspect node(s). Per-job logs: $OUT_DIR/jobs/"
  fi
  echo "CSV: $CSV"
} > "$REPORT"
cat "$REPORT"

# Non-zero exit when anything is flagged, so a wrapper can gate on it.
[[ $SUSPECTS -eq 0 ]] || exit 2
exit 0
