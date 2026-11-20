#!/bin/bash
# run_vr_nccl_test.sh -- multi-node NCCL test for VR200 nodes, launched from node 1
#
# Derived from deploy_rack_nccl_test.sh v0.22 (its run_nccl_test.sh part),
# adapted to how these VR200 nodes are set up:
#   - no jumper, no test pack: every node uses its SYSTEM-installed driver
#     (620.62), CUDA 13.5 and NCCL (libnccl2 2.32.4gr100), and nccl-tests
#     built with MPI=1 at the SAME path on every node (default ~/nccl-tests)
#   - launcher is the system OpenMPI 5 (prterun) -- not a custom pack mpirun
#   - runs as root (like the rack script) or as a normal user (e.g. nvidia).
#     Either way it needs passwordless SSH AS THAT USER to every node, itself
#     included. As root, --allow-run-as-root is added and the IMEX/channel0
#     repairs run directly; as a normal user they go through "sudo -n".
#     nccl-tests is looked for in $NCCL_TESTS_DIR, else ~/nccl-tests, else
#     /home/nvidia/nccl-tests (so root uses the nvidia user's build).
#   - same NCCL settings as the rack script (MNNVL auto, NVLS on, IB off)
#
# Run it ON NODE 1 (the first host in the list):
#   bash run_vr_nccl_test.sh                 # all hosts (default: 2 nodes / 8 GPUs)
#   bash run_vr_nccl_test.sh 1               # first node only (4 GPUs, smoke test)
#   bash run_vr_nccl_test.sh --info          # NCCL INFO -> /tmp/nccl_info/ on each node
#   bash run_vr_nccl_test.sh --dry-run       # pre-flight only (no repairs), print mpirun
#   bash run_vr_nccl_test.sh --max-size 8G   # shorter sweep (default 32G)
#   bash run_vr_nccl_test.sh --hosts 10.0.0.1,10.0.0.2
#   bash run_vr_nccl_test.sh --require-mnnvl # fail instead of silently falling back
#                                            # if multi-node NVLink is unavailable
#   bash run_vr_nccl_test.sh --allow-ib      # let NCCL use IB/RoCE between nodes
#   bash run_vr_nccl_test.sh --uuid 0x01     # set NCCL_MNNVL_UUID (normally not needed)
#   bash run_vr_nccl_test.sh --no-nvls       # NCCL_NVLS_ENABLE=0: skip NVLink-switch multicast
#                                            # (isolates multicast problems from plain MNNVL)
#   bash run_vr_nccl_test.sh --no-mnnvl      # NCCL_MNNVL_ENABLE=0 (+ no NVLS): control run that
#                                            # avoids NVLink between nodes (TCP, slow) to prove
#                                            # MPI/NCCL/tests are fine apart from the fabric
#                                            # (default max size then 64M: TCP over the
#                                            # bootstrap NIC is ~0.1 GB/s)
#
# Overridable via environment:
#   NCCL_TESTS_DIR (default ~/nccl-tests, then /home/nvidia/nccl-tests)
#   RESULTS_DIR    (default ~/nccl-results, i.e. /root/nccl-results as root)
#   EXPECT_DRIVER  (default 620.62)         EXPECT_NCCL (default 2.32.4gr100)
#   CUDA_HOME      (default /usr/local/cuda)

set -euo pipefail

SCRIPT_VERSION="0.7-vr"
GPUS_PER_NODE=4
DEFAULT_HOSTS="192.168.101.136,192.168.101.137"
if [[ -z "${NCCL_TESTS_DIR:-}" ]]; then
  for d in "$HOME/nccl-tests" /home/nvidia/nccl-tests; do
    [[ -x "$d/build/all_reduce_perf" ]] && { NCCL_TESTS_DIR="$d"; break; }
  done
  NCCL_TESTS_DIR="${NCCL_TESTS_DIR:-$HOME/nccl-tests}"
fi
RESULTS_DIR="${RESULTS_DIR:-$HOME/nccl-results}"
EXPECT_DRIVER="${EXPECT_DRIVER:-620.62}"
EXPECT_NCCL="${EXPECT_NCCL:-2.32.4gr100}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
IMEX_CFG="/etc/nvidia-imex/nodes_config.cfg"
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8)

HOSTS_RAW="$DEFAULT_HOSTS"
INFO=0
DRY_RUN=0
MAX_SIZE=""
MNNVL_ENABLE=2      # 2 = auto (rack script default), 1 = required, 0 = off
IB_DISABLE=1
NVLS_ENABLE=1
MNNVL_UUID=""
NODES_REQUESTED=""

usage() { sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hosts)         HOSTS_RAW="$2"; shift 2 ;;
    --info)          INFO=1; shift ;;
    --dry-run)       DRY_RUN=1; shift ;;
    --max-size)      MAX_SIZE="$2"; shift 2 ;;
    --require-mnnvl) MNNVL_ENABLE=1; shift ;;
    --allow-ib)      IB_DISABLE=0; shift ;;
    --no-nvls)       NVLS_ENABLE=0; shift ;;
    --no-mnnvl)      MNNVL_ENABLE=0; NVLS_ENABLE=0; shift ;;
    --uuid)          MNNVL_UUID="$2"; shift 2 ;;
    --version)       echo "run_vr_nccl_test.sh version ${SCRIPT_VERSION}"; exit 0 ;;
    -h|--help)       usage; exit 0 ;;
    [0-9]*)          NODES_REQUESTED="$1"; shift ;;
    *)               echo "ERROR: unknown argument: $1 (see --help)" >&2; exit 1 ;;
  esac
done

# Default sweep: 32G, but only 64M for --no-mnnvl (TCP over a ~1 Gb/s NIC
# would take tens of minutes per size above that)
MAX_SIZE="${MAX_SIZE:-$([[ $MNNVL_ENABLE -eq 0 ]] && echo 64M || echo 32G)}"

IFS=',' read -ra NODES <<< "$HOSTS_RAW"
NODES_REQUESTED="${NODES_REQUESTED:-${#NODES[@]}}"
if ! [[ "$NODES_REQUESTED" =~ ^[0-9]+$ ]] || (( NODES_REQUESTED < 1 || NODES_REQUESTED > ${#NODES[@]} )); then
  echo "ERROR: node count must be 1..${#NODES[@]} (got: $NODES_REQUESTED)" >&2
  exit 1
fi
RUN_NODES=("${NODES[@]:0:$NODES_REQUESTED}")
RUN_GPUS=$(( NODES_REQUESTED * GPUS_PER_NODE ))
NODE00="${NODES[0]}"

echo "=== run_vr_nccl_test.sh ${SCRIPT_VERSION} | $(hostname) | user $(whoami) ==="
echo "Test scope: ${NODES_REQUESTED} node(s) / ${RUN_GPUS} GPUs: ${RUN_NODES[*]}"

# mpirun must start on node-00 (the first host): it is the launch node and
# the IMEX/hostfile order is built around it.
if ! hostname -I | tr ' ' '\n' | grep -qx "$NODE00"; then
  echo "WARNING: this machine does not own ${NODE00} (first host). Run this on node 1." >&2
fi

ROOT_FLAG=()
if [[ $EUID -eq 0 ]]; then
  echo "Running as root: adding --allow-run-as-root"
  ROOT_FLAG=(--allow-run-as-root)
  # also honoured by the remote prted daemons / ranks
  export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
fi
echo "nccl-tests: ${NCCL_TESTS_DIR} | results: ${RESULTS_DIR}"

# --- 1. SSH: mpirun reaches every node (itself included) over SSH ---------
echo "=== Checking passwordless SSH to ${#RUN_NODES[@]} node(s) ==="
SSH_BROKEN=()
for ip in "${RUN_NODES[@]}"; do
  ssh "${SSH_OPTS[@]}" "$ip" true 2>/dev/null || SSH_BROKEN+=("$ip")
done
if [[ ${#SSH_BROKEN[@]} -gt 0 ]]; then
  echo "ERROR: passwordless SSH as $(whoami) failed to: ${SSH_BROKEN[*]}" >&2
  echo "       Fix (as $(whoami)): ssh-keygen -t ed25519 (if no key yet), then ssh-copy-id $(whoami)@<ip>" >&2
  echo "       for each node, including this node's own IP." >&2
  echo "       If ssh itself errors, check /etc/ssh/ssh_config for server-only options" >&2
  echo "       (e.g. PermitRootLogin belongs in sshd_config, not ssh_config)." >&2
  exit 1
fi
echo "    OK"

# --- 2. Per-node pre-flight (runs ON each node over SSH) -------------------
# Prints its checks, then one machine-readable line:
#   PREFLIGHT|<rc>|<driver>|<nccl>|<fabric state>|<cluster uuid>
read -r -d '' REMOTE_PREFLIGHT <<'REMOTE' || true
set -uo pipefail
EXPECT_DRIVER="$1"; EXPECT_NCCL="$2"; TESTS_DIR="$3"; CUDA_HOME="$4"
IMEX_CFG="$5"; HOSTLIST="$6"; DRY_RUN="$7"; INFO="$8"; MULTI="$9"; MYIP="${10}"
TAG="[$(hostname)/$(hostname -I | awk '{print $1}')]"
RC=0
ok()   { echo "  $TAG [OK]   $*"; }
warn() { echo "  $TAG [WARN] $*"; }
bad()  { echo "  $TAG [FAIL] $*"; RC=1; }
priv() { if [[ $EUID -eq 0 ]]; then "$@"; else sudo -n "$@"; fi; }

# Driver: user space + kernel module must both match (the 615/620 mix-up)
DRV="?"; KMD="?"
if OUT=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1); then
  DRV=$(echo "$OUT" | head -1 | tr -d ' ')
  NGPU=$(echo "$OUT" | wc -l)
  KMD=$(grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' /proc/driver/nvidia/version 2>/dev/null | head -1)
  if [[ "$DRV" == "$EXPECT_DRIVER" && "$KMD" == "$EXPECT_DRIVER" ]]; then ok "driver $DRV, $NGPU GPUs"
  else bad "driver user-space=$DRV kernel=$KMD, expected $EXPECT_DRIVER"; fi
  [[ "$NGPU" -eq 4 ]] || bad "$NGPU GPUs visible, expected 4"
else
  bad "nvidia-smi failed: $(echo "$OUT" | head -1)  (check: journalctl -k -b | grep NVRM)"
fi

# NCCL package
NCCL=$(dpkg-query -W -f='${Version}' libnccl2 2>/dev/null || echo "none")
if [[ "$NCCL" == *"$EXPECT_NCCL"* ]]; then ok "libnccl2 $NCCL"
else bad "libnccl2 is '$NCCL', expected $EXPECT_NCCL"; fi

# Test binaries: present, built with MPI, linked to the system NCCL
for b in all_reduce_perf alltoall_perf; do
  BIN="$TESTS_DIR/build/$b"
  if [[ ! -x "$BIN" ]]; then bad "$BIN missing (build nccl-tests with MPI=1 on this node)"; continue; fi
  LDD=$(LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}" ldd "$BIN" 2>&1)
  echo "$LDD" | grep -q "not found" && bad "$b: missing libs: $(echo "$LDD" | awk '/not found/{print $1}' | xargs)"
  echo "$LDD" | grep -q "libmpi" || bad "$b: not built with MPI=1 (no libmpi) -- rebuild: make clean && make MPI=1 ..."
  LIB=$(echo "$LDD" | awk '/libnccl\.so/ {print $3}')
  [[ "$LIB" == "/usr/lib/aarch64-linux-gnu/libnccl.so.2" ]] || bad "$b loads NCCL from '${LIB:-nothing}', not the system copy"
done
[[ $RC -eq 0 ]] && ok "nccl-tests binaries OK"

# Bootstrap interface = the one holding this node's address from the host list.
# Its NAME differs per node (enx<MAC> etc.), so it is reported back, not assumed.
IFLINE=$(ip -o -4 addr show 2>/dev/null | awk -v ip="$MYIP" '{split($4,a,"/"); if (a[1]==ip) {print $2, $4; exit}}')
IFNAME="${IFLINE%% *}"; IFCIDR="${IFLINE#* }"
if [[ -n "$IFLINE" ]]; then ok "bootstrap interface $IFNAME ($IFCIDR)"
else bad "no local interface holds $MYIP"; IFNAME="?"; IFCIDR="?"; fi

FABRIC="n/a"; CLUSTER="n/a"
if [[ "$MULTI" -eq 1 ]]; then
  # IMEX node list must contain every node, in the same order, on every node
  WANT=$(echo "$HOSTLIST" | tr ',' '\n')
  HAVE=$(cat "$IMEX_CFG" 2>/dev/null || true)
  RESTART=0
  if [[ "$HAVE" != "$WANT" ]]; then
    if [[ "$DRY_RUN" -eq 1 ]]; then warn "$IMEX_CFG differs from host list (would rewrite + restart IMEX)"
    elif echo "$WANT" | priv tee "$IMEX_CFG" >/dev/null 2>&1; then ok "rewrote $IMEX_CFG"; RESTART=1
    else bad "$IMEX_CFG differs and sudo -n is not allowed. Fix as root: printf '%s\n' $(echo "$WANT" | xargs) > $IMEX_CFG && systemctl restart nvidia-imex"; fi
  fi
  if ! systemctl is-active --quiet nvidia-imex; then RESTART=1; fi
  if [[ $RESTART -eq 1 && "$DRY_RUN" -eq 0 ]]; then
    if priv systemctl restart nvidia-imex 2>/dev/null; then
      for _ in $(seq 1 15); do systemctl is-active --quiet nvidia-imex && break; sleep 2; done
    fi
  fi
  if systemctl is-active --quiet nvidia-imex; then ok "nvidia-imex active"
  elif [[ "$DRY_RUN" -eq 1 ]]; then warn "nvidia-imex not active (would restart)"
  else bad "nvidia-imex not active (check: journalctl -u nvidia-imex -n 50)"; fi

  # IMEX channel0: CUDA needs it for multi-node NVLink memory sharing
  if [[ ! -e /dev/nvidia-caps-imex-channels/channel0 ]]; then
    MAJOR=$(awk '/nvidia-caps-imex-channels/ {print $1}' /proc/devices)
    if [[ "$DRY_RUN" -eq 1 ]]; then warn "channel0 missing (would mknod)"
    elif [[ -n "$MAJOR" ]] && priv mkdir -p /dev/nvidia-caps-imex-channels \
         && priv mknod /dev/nvidia-caps-imex-channels/channel0 c "$MAJOR" 0 \
         && priv chmod 0666 /dev/nvidia-caps-imex-channels/channel0; then
      ok "channel0 recreated"
    else bad "channel0 missing and could not be created (needs root; major='${MAJOR}')"; fi
  else ok "channel0 present"; fi

  # NVLink fabric registration (needed for MNNVL between the nodes)
  FABRIC=$(nvidia-smi -q 2>/dev/null | awk '/^ *Fabric *$/{f=1;next} f&&/State/{print $NF;exit}')
  CLUSTER=$(nvidia-smi -q 2>/dev/null | awk -F': *' '/ClusterUUID/{print $2;exit}')
  FABRIC="${FABRIC:-unknown}"; CLUSTER="${CLUSTER:-unknown}"
  if [[ "$FABRIC" == "Completed" ]]; then ok "fabric Completed, ClusterUUID $CLUSTER"
  else warn "fabric state '$FABRIC' -- multi-node NVLink may not be available"; fi
fi

# Fabric HEALTH (not just State): Route Unhealthy=True disables CUDA P2P even
# inside one tray (seen with a GFM partition that still expected missing trays)
HSUM=$(nvidia-smi -q 2>/dev/null | awk -F': *' '/^ *Summary/ {print $2}' | sort -u | xargs)
ROUTE=$(nvidia-smi -q 2>/dev/null | awk -F': *' '/Route Unhealthy/ {print $2}' | sort -u | xargs)
if [[ -n "$ROUTE" ]]; then
  if [[ "$ROUTE" == "False" && "$HSUM" == "Healthy" ]]; then ok "fabric health Healthy, routes OK"
  else bad "fabric health '$HSUM', Route Unhealthy '$ROUTE' -- check the NVLink partition on the GFM switch (nv show sdn partition)"; fi
fi
P2P_NS=$(nvidia-smi topo -p2p r 2>/dev/null | awk '$1 ~ /^GPU[0-9]+$/ && $2 !~ /^GPU/ {for(i=2;i<=NF;i++) if($i!="X" && $i!="OK") n++} END{print n+0}')
[[ "$P2P_NS" -eq 0 ]] && ok "GPU P2P OK between all GPU pairs" || bad "GPU P2P not OK for $P2P_NS GPU pair entries (nvidia-smi topo -p2p r)"


# Stale test processes from an aborted run ("CUDA-capable device(s) is/are busy")
PIDS=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
  | grep -E 'all_reduce_perf|alltoall_perf|prterun|prted|mpirun' | awk -F, '{print $1}' | xargs)
if [[ -n "$PIDS" ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then warn "stale test processes on GPU: $PIDS (would kill)"
  else
    kill -9 $PIDS 2>/dev/null || priv kill -9 $PIDS 2>/dev/null || true
    sleep 2
    if nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null \
        | grep -qE 'all_reduce_perf|alltoall_perf|prterun|prted|mpirun'; then
      bad "stale test processes survived kill -9 -- check dmesg for Xid"
    else ok "cleared stale processes: $PIDS"; fi
  fi
fi

# PMIx "unable to find a usable compression library" startup noise (cosmetic)
mkdir -p "$HOME/.pmix"
grep -qxF 'pcompress_base_silence_warning = 1' "$HOME/.pmix/mca-params.conf" 2>/dev/null \
  || echo 'pcompress_base_silence_warning = 1' >> "$HOME/.pmix/mca-params.conf"

if [[ "$INFO" -eq 1 && "$DRY_RUN" -eq 0 ]]; then rm -rf /tmp/nccl_info && mkdir -p /tmp/nccl_info; fi

echo "PREFLIGHT|$RC|$DRV|$NCCL|$FABRIC|$CLUSTER|$IFNAME|$IFCIDR|$(hostname)"
REMOTE

MULTI=$(( NODES_REQUESTED > 1 ? 1 : 0 ))
echo "=== Pre-flight on ${#RUN_NODES[@]} node(s)$([[ $DRY_RUN -eq 1 ]] && echo ' (dry run: check only)') ==="
FAILED=()
declare -A P_DRV P_NCCL P_CLUSTER P_IF P_CIDR P_HOST
for ip in "${RUN_NODES[@]}"; do
  OUT=$(ssh "${SSH_OPTS[@]}" "$ip" bash -s -- \
          "$EXPECT_DRIVER" "$EXPECT_NCCL" "$NCCL_TESTS_DIR" "$CUDA_HOME" \
          "$IMEX_CFG" "$HOSTS_RAW" "$DRY_RUN" "$INFO" "$MULTI" "$ip" <<< "$REMOTE_PREFLIGHT" 2>&1) || true
  echo "$OUT" | grep -v '^PREFLIGHT|'
  LINE=$(echo "$OUT" | grep '^PREFLIGHT|' | tail -1)
  if [[ -z "$LINE" ]]; then FAILED+=("$ip"); echo "  [$ip] [FAIL] pre-flight did not complete"; continue; fi
  IFS='|' read -r _ rc drv nccl _fabric cluster ifname ifcidr hname <<< "$LINE"
  [[ "$rc" == "0" ]] || FAILED+=("$ip")
  P_DRV[$ip]="$drv"; P_NCCL[$ip]="$nccl"; P_CLUSTER[$ip]="$cluster"
  P_IF[$ip]="$ifname"; P_CIDR[$ip]="$ifcidr"; P_HOST[$ip]="$hname"
done

# All nodes must run the identical stack
if [[ ${#RUN_NODES[@]} -gt 1 ]]; then
  for map in P_DRV P_NCCL; do
    declare -n M="$map"
    if [[ $(printf '%s\n' "${M[@]}" | sort -u | wc -l) -gt 1 ]]; then
      echo "  [FAIL] ${map#P_} differs between nodes:"; for ip in "${RUN_NODES[@]}"; do echo "         $ip ${M[$ip]}"; done
      FAILED+=("mismatch")
    fi
    unset -n M
  done
  # Identical hostnames (e.g. all "localhost") made Open MPI hang before NCCL
  # even started -- it treats ranks with the same hostname as one machine.
  if [[ $(printf '%s\n' "${P_HOST[@]}" | sort -u | wc -l) -lt ${#RUN_NODES[@]} ]]; then
    echo "  [FAIL] nodes share a hostname: $(for ip in "${RUN_NODES[@]}"; do printf '%s=%s ' "$ip" "${P_HOST[$ip]}"; done)"
    echo "         give each node a unique name: sudo hostnamectl set-hostname <name> (and add it to /etc/hosts)"
    FAILED+=("hostname")
  fi
  if [[ $(printf '%s\n' "${P_CLUSTER[@]}" | sort -u | wc -l) -gt 1 ]]; then
    echo "  [WARN] ClusterUUID differs between nodes -> they are NOT in the same NVLink domain;"
    echo "         NCCL cannot use MNNVL between them (needs --allow-ib and working IB/RoCE)."
  fi
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "=== Pre-flight FAILED: ${FAILED[*]} -- not running the test ===" >&2
  [[ $DRY_RUN -eq 1 ]] || exit 1
else
  echo "=== Pre-flight complete ==="
fi

# --- 3. Launch -------------------------------------------------------------
mkdir -p "$RESULTS_DIR"
HOSTFILE="${RESULTS_DIR}/hostfile_${NODES_REQUESTED}node.cfg"
for ip in "${RUN_NODES[@]}"; do echo "$ip slots=${GPUS_PER_NODE}"; done > "$HOSTFILE"

# Bootstrap network (MPI launch + NCCL's socket bootstrap; data goes over
# NVLink). Interface NAMES differ per node, so:
#   - OpenMPI/PRRTE get the SUBNET(s) in CIDR form -> valid on every node
#   - NCCL gets every node's exact name ("=a,b"); each node uses the one it has
cidr_net() {  # 192.168.101.136/24 -> 192.168.101.0/24
  local ip="${1%/*}" p="${1#*/}" a b c d n m
  IFS=. read -r a b c d <<< "$ip"
  n=$(( (a << 24) | (b << 16) | (c << 8) | d ))
  m=$(( p == 0 ? 0 : (0xFFFFFFFF << (32 - p)) & 0xFFFFFFFF ))
  n=$(( n & m ))
  echo "$(( n >> 24 & 255 )).$(( n >> 16 & 255 )).$(( n >> 8 & 255 )).$(( n & 255 ))/$p"
}
NETS=(); IFNAMES=()
for ip in "${RUN_NODES[@]}"; do
  [[ "${P_CIDR[$ip]:-?}" == "?" ]] && continue
  NETS+=("$(cidr_net "${P_CIDR[$ip]}")"); IFNAMES+=("${P_IF[$ip]}")
done
MPI_NET=$(printf '%s\n' "${NETS[@]}" | sort -u | paste -sd, -)
# NCCL parses one leading "=" (exact match) for the WHOLE list: "=a,b"
NCCL_IFS="=$(printf '%s\n' "${IFNAMES[@]}" | sort -u | paste -sd, -)"
if [[ -z "$MPI_NET" || -z "$NCCL_IFS" ]]; then
  echo "ERROR: could not determine the bootstrap network from pre-flight" >&2
  exit 1
fi
for ip in "${RUN_NODES[@]}"; do echo "Bootstrap: $ip -> ${P_IF[$ip]:-?} (${P_CIDR[$ip]:-?})"; done
echo "MPI network: $MPI_NET | NCCL_SOCKET_IFNAME: $NCCL_IFS"

LOG="${RESULTS_DIR}/$(date +%Y%m%d_%H%M%S)_${NODES_REQUESTED}node.log"
NCCL_ENV=(
  "NCCL_SOCKET_IFNAME=${NCCL_IFS}"
  "NCCL_MNNVL_ENABLE=${MNNVL_ENABLE}"
  "NCCL_NVLS_ENABLE=${NVLS_ENABLE}"
  NCCL_P2P_DISABLE=0
  "NCCL_IB_DISABLE=${IB_DISABLE}"
  CUDA_IPC_HANDLE_SHARING_SUPPORT=1
  CUDA_DEVICE_MAX_CONNECTIONS=1
  NCCL_MIN_CTAS=32
  NCCL_P2P_SCHEDULE_GROUP_SIZE=1
)
[[ -n "$MNNVL_UUID" ]] && NCCL_ENV+=("NCCL_MNNVL_UUID=${MNNVL_UUID}")
if [[ $INFO -eq 1 ]]; then
  NCCL_ENV+=(NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,NVLS,ENV "NCCL_DEBUG_FILE=/tmp/nccl_info/%h.%p.log")
  LOG="${LOG%.log}_info.log"
else
  NCCL_ENV+=(NCCL_DEBUG=WARN)
fi
ENV_ARGS=(); for kv in "${NCCL_ENV[@]}"; do ENV_ARGS+=(-x "$kv"); done

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PATH="${CUDA_HOME}/bin:${PATH}"

run_one() {
  local label="$1" bin="$2"; shift 2
  local cmd=(mpirun "${ROOT_FLAG[@]}"
    --hostfile "$HOSTFILE" -np "$RUN_GPUS" --map-by "ppr:${GPUS_PER_NODE}:node" --bind-to none
    --mca pml ob1 --mca btl self,tcp --mca btl_tcp_if_include "$MPI_NET"
    --prtemca oob_tcp_if_include "$MPI_NET"
    --mca mca_base_component_show_load_errors 0
    -x PATH -x LD_LIBRARY_PATH "${ENV_ARGS[@]}"
    "$bin" "$@" -g 1)
  echo "--- ${label} ---"
  echo "+ ${cmd[*]}"
  [[ $DRY_RUN -eq 1 ]] && return 0
  "${cmd[@]}"
}

AR_ARGS=(-b 8 -e "$MAX_SIZE" -f 2)
A2A_ARGS=(-d uint8 -b 8 -e "$MAX_SIZE" -f 2)

if [[ $DRY_RUN -eq 1 ]]; then
  echo "Hostfile ${HOSTFILE}:"; sed 's/^/  /' "$HOSTFILE"
  run_one "All-Reduce (${RUN_GPUS} GPUs)" "$NCCL_TESTS_DIR/build/all_reduce_perf" "${AR_ARGS[@]}"
  run_one "All-to-All (${RUN_GPUS} GPUs)" "$NCCL_TESTS_DIR/build/alltoall_perf" "${A2A_ARGS[@]}"
  echo "=== Dry run: nothing executed ==="
  exit 0
fi

{
  echo "$(date) : ${NODES_REQUESTED} node(s) | ${RUN_GPUS} GPUs | ${RUN_NODES[*]} | driver ${P_DRV[$NODE00]:-?} | nccl ${P_NCCL[$NODE00]:-?} | launch $(hostname)"
  run_one "All-Reduce (${RUN_GPUS} GPUs)" "$NCCL_TESTS_DIR/build/all_reduce_perf" "${AR_ARGS[@]}" \
    || echo "!!! All-Reduce exited with error $?"
  run_one "All-to-All (${RUN_GPUS} GPUs)" "$NCCL_TESTS_DIR/build/alltoall_perf" "${A2A_ARGS[@]}" \
    || echo "!!! All-to-All exited with error $?"
  echo "$(date) : Done."
} 2>&1 | tee "$LOG"

# --- 4. Summary ------------------------------------------------------------
echo ""
echo "=== Summary ==="
# per-test lines always; repeated CUDA failure lines (one per rank) only once
grep -aE "^--- |Out of bounds|Avg bus bandwidth|!!! |Cuda failure" "$LOG" \
  | awk '/Cuda failure/ { k=$0; sub(/^.*\(/, "", k); if (seen[k]++) next } { print }' | head -20 || true
RC=0
grep -aq "!!! " "$LOG" && RC=1
grep -ac "Avg bus bandwidth" "$LOG" | grep -qx 2 || RC=1
grep -a "Out of bounds values" "$LOG" | grep -avq " 0 OK" && RC=1

if [[ $INFO -eq 1 ]]; then
  # Which transport carried the traffic (from this node's ranks)
  echo "Transports seen in /tmp/nccl_info on $(hostname):"
  cat /tmp/nccl_info/*.log 2>/dev/null \
    | grep -oE "MNNVL[^ ]*|NVLS multicast support is (available|not available)|via (P2P/[A-Za-z]+|NET/[A-Za-z]+|SHM)" \
    | sort | uniq -c | sed 's/^/  /' || true
  echo "  (MNNVL / NVLS = NVLink between nodes; NET/Socket = TCP fallback; NET/IB = network)"
fi
echo "Log: $LOG"
if [[ $RC -eq 0 ]]; then echo "RESULT: PASS"; else echo "RESULT: FAIL"; fi
exit $RC
