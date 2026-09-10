#!/bin/bash
# deploy_rack_nccl_test.sh
#
# Run from the jumper, from the folder containing rackXX.sh and the .sqsh
# image (e.g. ~/carlonext). One script does everything end to end:
#
#   1. scp's the .sqsh ONCE to node-00 (CM_IPS IP0), NFS-exports it, and
#      mounts it read-only on every sibling node -- no 18x copy of a
#      large image across the rack
#   2. on EVERY node, IN PARALLEL, idempotently `enroot create`s a local
#      container from that shared .sqsh. The nccl-tests binaries run
#      INSIDE this container via `enroot start` -- nothing is extracted
#      onto the bare host filesystem, so the node's own OS/environment is
#      never touched. The unpacked container itself lives on the
#      dedicated persistent data/scratch volume you point --data-dir at
#      (NOT the OS disk, and NOT tmpfs) -- this is the slow step
#      (unsquashing a multi-GB image), so it: (a) runs once, at deploy
#      time, in parallel across all nodes instead of one-at-a-time, and
#      (b) persists across reboots/shipping, so it never needs to be
#      redone at the customer's site
#   3. meshes root SSH key: node-00 -> siblings (mpirun launches node->node)
#   4. generates and stages run_nccl_test.sh directly on node-00 -- no
#      separate script needed, ready to run immediately
#   5. (optional, --auto) SSHes into node-00 and runs it for you
#
# --data-dir <path> is REQUIRED: must be a path on a persistent volume
# that is genuinely separate from the OS/root filesystem (e.g. a local
# scratch/data partition or disk). Every node is checked at runtime
# (`findmnt`) to confirm --data-dir is NOT on the same filesystem as /,
# and the script refuses to proceed on that node if it is -- this is a
# hard safety guard against accidentally writing container data onto the
# system disk, not just a naming convention.
#
# IMPORTANT: nvidia-imex health + /dev/nvidia-caps-imex-channels/channel0
# are NOT checked here at deploy time. Unlike the container (above),
# channel0 CANNOT be made to persist across reboot under any design --
# it's a kernel-driver-backed device node that is always recreated fresh
# on every boot, on every system. So that check (cheap: a service-active
# check + an mknod, not the slow part) lives INSIDE run_nccl_test.sh and
# runs fresh every single time the test is invoked, against every node,
# right before mpirun. See "IMEX pre-flight" below.
#
# Usage:
#   ./deploy_rack_nccl_test.sh rack17.sh --data-dir /raid/nccl-diag
#                                                 # --data-dir is REQUIRED: your
#                                                 # persistent scratch/data volume,
#                                                 # NOT the OS disk (sqsh defaults
#                                                 # to ./compiled-nccl-test-image+latest.sqsh)
#   ./deploy_rack_nccl_test.sh rack17.sh my-image.sqsh --data-dir /raid/nccl-diag
#   ./deploy_rack_nccl_test.sh rack17.sh --data-dir /raid/nccl-diag --uuid 0x1969
#   ./deploy_rack_nccl_test.sh rack17.sh --data-dir /raid/nccl-diag --auto
#   ./deploy_rack_nccl_test.sh rack17.sh --data-dir /raid/nccl-diag --dry-run
#   ./deploy_rack_nccl_test.sh rack17.sh --data-dir /raid/nccl-diag --version
#   ./deploy_rack_nccl_test.sh rack17.sh --data-dir /raid/nccl-diag --only 192.168.14.187,192.168.14.191
#                                                 # redeploy just these node(s) after a
#                                                 # hardware swap -- skips the NFS-mount
#                                                 # and enroot container rebuild on every
#                                                 # OTHER node in the rack
#
# NCCL_MNNVL_UUID is auto-derived from the rack filename if not given
# (rack17.sh -> 0x17, rack8.sh -> 0x08) -- a human-traceable tag, not a
# magic value; override with --uuid anytime.
#
# RE-RUN / NODE-SWAP SAFETY: this script is safe to run 2+ times on the
# same rack, including after the diag team physically swaps a node:
#   - stale SSH host keys for swapped IPs are cleared automatically
#     (new hardware = new host key, same IP -- otherwise SSH refuses to
#     reconnect)
#   - the sqsh copy to node-00 is skipped if the remote file already
#     matches the local one's size (cheap re-runs, no repeated 7GB+ scp)
#   - NFS export/mount, enroot container creation, and SSH key meshing
#     are all check-before-act and won't duplicate work
#   - use --only <ip1,ip2,...> to limit the NFS-mount + container-rebuild
#     steps to just the swapped node(s); if node-00 itself is in that
#     list, the script automatically falls back to a full rack pass
#     since node-00 is the image/NFS source for everyone else
#   - if a replacement node gets a NEW IP, just update CM_IPS in the
#     rackXX.sh file accordingly before re-running -- everything else is
#     driven off that array
#   - nvidia-imex/channel0 health doesn't need a redeploy at all after a
#     reboot -- run_nccl_test.sh re-checks and repairs it every time it runs
#     (this is unavoidable for channel0; it is NOT true of the container,
#     which persists on --data-dir across reboots without any rework)

set -euo pipefail

SCRIPT_VERSION="0.1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARE_DIR="/mnt/nccl-share"
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=8"
CONTAINER_NAME="nccltest"
CONTAINER_WORKDIR="/var/nccl-tests"          # nccl-tests build dir INSIDE the container
IMEX_CFG="/etc/nvidia-imex/nodes_config.cfg"
GPUS_PER_NODE=4
DEFAULT_SQSH_NAME="compiled-nccl-test-image+latest.sqsh"
IMEX_WAIT_ATTEMPTS=15
IMEX_WAIT_SLEEP=2

# --version short-circuits everything else, even with no rack file given
for arg in "$@"; do
  if [[ "$arg" == "--version" ]]; then
    echo "deploy_rack_nccl_test.sh version ${SCRIPT_VERSION}"
    exit 0
  fi
done

RACK_FILE=""
SQSH=""
DRY_RUN=0
AUTO=0
MNNVL_UUID=""
ONLY_RAW=""
DATA_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)  DRY_RUN=1; shift ;;
    --auto)     AUTO=1; shift ;;
    --uuid)     MNNVL_UUID="$2"; shift 2 ;;
    --only)     ONLY_RAW="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    *.sqsh)     SQSH="$1"; shift ;;
    *)          RACK_FILE="$1"; shift ;;
  esac
done
[[ -n "$RACK_FILE" ]] || { echo "Usage: $0 <rack_file.sh> --data-dir <path> [sqsh_path] [--uuid 0xNNNN] [--only ip1,ip2] [--auto] [--dry-run] [--version]" >&2; exit 1; }
[[ -n "$DATA_DIR" ]] || { echo "ERROR: --data-dir <path> is required -- point it at a persistent scratch/data volume on each node, NOT the OS disk (e.g. --data-dir /raid/nccl-diag)" >&2; exit 1; }
DATA_DIR="${DATA_DIR%/}"   # strip any trailing slash for clean path joins
ENROOT_DATA_PATH="${DATA_DIR}/enroot/data"
ENROOT_RUNTIME_PATH="${DATA_DIR}/enroot/runtime"
ENROOT_CACHE_PATH="${DATA_DIR}/enroot/cache"

[[ -f "$RACK_FILE" ]] || RACK_FILE="$SCRIPT_DIR/$RACK_FILE"
[[ -f "$RACK_FILE" ]] || { echo "ERROR: rack file not found (checked cwd and $SCRIPT_DIR)" >&2; exit 1; }

# Auto-derive NCCL_MNNVL_UUID from the rack filename if not explicitly given
if [[ -z "$MNNVL_UUID" ]]; then
  RACK_NUM=$(basename "$RACK_FILE" .sh | grep -oE '[0-9]+' | head -1)
  if [[ -n "$RACK_NUM" ]]; then
    MNNVL_UUID="0x$(printf '%02d' "$RACK_NUM")"
    echo "Auto-derived NCCL_MNNVL_UUID=${MNNVL_UUID} from rack filename (override with --uuid)"
  else
    MNNVL_UUID="0x0"
    echo "WARNING: could not parse a rack number from $(basename "$RACK_FILE") -- defaulting NCCL_MNNVL_UUID=0x0 (set --uuid explicitly)" >&2
  fi
fi

# Default sqsh: ./compiled-nccl-test-image+latest.sqsh next to this script
if [[ -z "$SQSH" ]]; then
  SQSH="$SCRIPT_DIR/$DEFAULT_SQSH_NAME"
else
  [[ -f "$SQSH" ]] || SQSH="$SCRIPT_DIR/$SQSH"
fi
[[ -f "$SQSH" ]] || { echo "ERROR: sqsh not found: $SQSH (default is ./${DEFAULT_SQSH_NAME})" >&2; exit 1; }

[[ $DRY_RUN -eq 1 ]] && echo ">>> DRY-RUN MODE: no SSH/SCP/mount commands will actually run <<<"

# shellcheck source=/dev/null
source "$RACK_FILE"
[[ -n "${CM_IPS:-}" ]] || { echo "ERROR: CM_IPS array not found in $RACK_FILE" >&2; exit 1; }

# CM_IPS is listed high-IP-first (IP17..IP0); reverse so index 0 = IP0 = node-00
NODES=()
for ((i=${#CM_IPS[@]}-1; i>=0; i--)); do NODES+=("${CM_IPS[$i]}"); done
NODE00="${NODES[0]}"
SUBNET=$(echo "$NODE00" | awk -F. '{print $1"."$2"."$3".0/24"}')
SQSH_BASENAME="$(basename "$SQSH")"
TOTAL_RANKS=$(( ${#NODES[@]} * GPUS_PER_NODE ))

echo "=== Rack: $(basename "$RACK_FILE") | ${#NODES[@]} nodes | node-00=$NODE00 | sqsh=$SQSH_BASENAME | UUID=$MNNVL_UUID ==="

# --- Resolve --only into TARGET_NODES (the nodes that get the NFS mount +
# enroot container rebuild re-applied). SSH mesh + staging always cover the
# full rack since those are cheap and idempotent anyway.
TARGET_NODES=("${NODES[@]}")
if [[ -n "$ONLY_RAW" ]]; then
  IFS=',' read -ra ONLY_IPS <<< "$ONLY_RAW"
  TARGET_NODES=()
  for want in "${ONLY_IPS[@]}"; do
    found=0
    for ip in "${NODES[@]}"; do [[ "$ip" == "$want" ]] && { found=1; break; }; done
    if [[ $found -eq 0 ]]; then
      echo "ERROR: --only IP '$want' is not in $(basename "$RACK_FILE")'s CM_IPS" >&2
      exit 1
    fi
    TARGET_NODES+=("$want")
  done
  if printf '%s\n' "${TARGET_NODES[@]}" | grep -qx "$NODE00"; then
    echo "NOTE: node-00 ($NODE00) is in --only -- it's the image/NFS source for the"
    echo "      whole rack, so falling back to a full pass instead of a partial one."
    TARGET_NODES=("${NODES[@]}")
  else
    echo "--- Targeted redeploy: container rebuild limited to: ${TARGET_NODES[*]} ---"
  fi
fi

# Swapped hardware on a reused IP means a NEW host key -- clear any stale
# cached entry for everything we're about to SSH into this run, or SSH
# will refuse to reconnect ("REMOTE HOST IDENTIFICATION HAS CHANGED").
if [[ $DRY_RUN -eq 0 ]]; then
  { echo "$NODE00"; printf '%s\n' "${TARGET_NODES[@]}"; } | sort -u | while read -r ip; do
    ssh-keygen -R "$ip" >/dev/null 2>&1 || true
  done
fi

# --- dry-run aware helpers -------------------------------------------------
ssh_do() {
  local host="$1"; shift
  if [[ $DRY_RUN -eq 1 ]]; then echo "[DRY-RUN] ssh root@${host} $*"; else ssh $SSH_OPTS "root@${host}" "$@"; fi
}
scp_do() {
  local src="$1" host="$2" dst="$3"
  if [[ $DRY_RUN -eq 1 ]]; then echo "[DRY-RUN] scp $src root@${host}:${dst}"
  else scp $SSH_OPTS "$src" "root@${host}:${dst}"; fi
}
ssh_script() {  # usage: ssh_script <host> <<'EOF' ... EOF
  local host="$1"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] ssh root@${host} bash -s <<EOF"
    cat
    echo "EOF"
  else
    ssh $SSH_OPTS "root@${host}" bash -s
  fi
}
# ---------------------------------------------------------------------------

echo "--- [1/5] Copying sqsh to node-00 ($NODE00), NFS-sharing to siblings ---"
ssh_do "$NODE00" "mkdir -p ${SHARE_DIR}"
if [[ $DRY_RUN -eq 1 ]]; then
  echo "[DRY-RUN] would skip copy if remote size already matches local, else: scp $SQSH root@${NODE00}:${SHARE_DIR}/${SQSH_BASENAME}"
else
  LOCAL_SIZE=$(stat -c%s "$SQSH")
  REMOTE_SIZE=$(ssh $SSH_OPTS "root@${NODE00}" "stat -c%s ${SHARE_DIR}/${SQSH_BASENAME} 2>/dev/null || echo 0")
  if [[ "$REMOTE_SIZE" == "$LOCAL_SIZE" ]]; then
    echo "  sqsh already present on node-00 with matching size (${LOCAL_SIZE} bytes) -- skipping copy"
  else
    scp $SSH_OPTS "$SQSH" "root@${NODE00}:${SHARE_DIR}/${SQSH_BASENAME}"
  fi
fi

ssh_script "$NODE00" <<EOF
set -e
command -v exportfs >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq nfs-kernel-server)
grep -q "^${SHARE_DIR} " /etc/exports 2>/dev/null || echo "${SHARE_DIR} ${SUBNET}(ro,sync,no_subtree_check,no_root_squash)" >> /etc/exports
exportfs -ra
systemctl enable --now nfs-kernel-server 2>/dev/null || systemctl restart nfs-kernel-server
EOF

for ip in "${TARGET_NODES[@]}"; do
  [[ "$ip" == "$NODE00" ]] && continue
  ssh_script "$ip" <<EOF
set -e
command -v mount.nfs >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq nfs-common)
mkdir -p ${SHARE_DIR}
grep -q "${SHARE_DIR} " /etc/fstab 2>/dev/null || echo "${NODE00}:${SHARE_DIR} ${SHARE_DIR} nfs ro,_netdev 0 0" >> /etc/fstab
mountpoint -q ${SHARE_DIR} || mount ${SHARE_DIR}
EOF
done

echo "--- [2/5] Creating local enroot container '${CONTAINER_NAME}' on target node(s) (parallel) ---"
echo "    (container only -- nothing extracted to the host OS filesystem;"
echo "     persisted under ${DATA_DIR} on each node, verified separate from /)"

container_create_node() {
  local ip="$1"
  ssh_script "$ip" <<EOF
set -e
command -v enroot >/dev/null || { echo "ERROR: enroot missing on ${ip}" >&2; exit 1; }

# Hard safety check: refuse to write container data onto the same
# filesystem as / (the OS disk), even if --data-dir was mistyped.
mkdir -p "${DATA_DIR}"
ROOT_DEV=\$(findmnt -n -o SOURCE / 2>/dev/null)
DATA_DEV=\$(findmnt -n -o SOURCE --target "${DATA_DIR}" 2>/dev/null)
if [[ -z "\$DATA_DEV" || "\$DATA_DEV" == "\$ROOT_DEV" ]]; then
  echo "ERROR: [${ip}] ${DATA_DIR} is on the same filesystem as / -- refusing to" >&2
  echo "       write enroot container data onto the OS/system disk. Point" >&2
  echo "       --data-dir at a real separate scratch/data volume." >&2
  exit 1
fi

export ENROOT_DATA_PATH="${ENROOT_DATA_PATH}"
export ENROOT_RUNTIME_PATH="${ENROOT_RUNTIME_PATH}"
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH}"
mkdir -p "\$ENROOT_DATA_PATH" "\$ENROOT_RUNTIME_PATH" "\$ENROOT_CACHE_PATH"

if enroot list | grep -qx "${CONTAINER_NAME}"; then
  echo "  [$ip] container '${CONTAINER_NAME}' already exists on ${DATA_DIR}, skipping"
else
  echo "  [$ip] creating container '${CONTAINER_NAME}' from ${SHARE_DIR}/${SQSH_BASENAME} (this is the slow step)"
  enroot create --name ${CONTAINER_NAME} "${SHARE_DIR}/${SQSH_BASENAME}"
  echo "  [$ip] container '${CONTAINER_NAME}' ready, persisted under ${DATA_DIR}"
fi
EOF
}

if [[ $DRY_RUN -eq 1 ]]; then
  for ip in "${TARGET_NODES[@]}"; do container_create_node "$ip"; done
else
  declare -A PIDS
  for ip in "${TARGET_NODES[@]}"; do
    ( container_create_node "$ip" > "/tmp/.container_create_${ip}.log" 2>&1 ) &
    PIDS["$ip"]=$!
  done

  FAILED=()
  for ip in "${!PIDS[@]}"; do
    wait "${PIDS[$ip]}" || FAILED+=("$ip")
    sed "s/^/[$ip] /" "/tmp/.container_create_${ip}.log"
    rm -f "/tmp/.container_create_${ip}.log"
  done

  if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "ERROR: container creation failed on: ${FAILED[*]}" >&2
    exit 1
  fi
fi

echo "--- [3/5] Meshing root SSH key: node-00 -> siblings (required for mpirun AND the IMEX pre-flight) ---"
ssh_do "$NODE00" "test -f /root/.ssh/id_ed25519 || ssh-keygen -t ed25519 -N '' -f /root/.ssh/id_ed25519 -q"
if [[ $DRY_RUN -eq 1 ]]; then
  echo "[DRY-RUN] would fetch node-00 pubkey and append to authorized_keys on: ${NODES[*]}"
else
  PUBKEY=$(ssh $SSH_OPTS "root@${NODE00}" "cat /root/.ssh/id_ed25519.pub")
  for ip in "${NODES[@]}"; do
    ssh_do "$ip" "mkdir -p /root/.ssh && grep -qF '${PUBKEY}' /root/.ssh/authorized_keys 2>/dev/null || echo '${PUBKEY}' >> /root/.ssh/authorized_keys"
  done
fi

echo "--- [4/5] Staging run_nccl_test.sh on node-00 ---"

# mpirun's hostfile IS the IMEX node list (same convention as the
# reference diag log: --hostfile /etc/nvidia-imex/nodes_config.cfg).
# Each rank runs INSIDE the enroot container via `enroot start`, so the
# host's own CUDA/NCCL/OpenMPI environment is never touched.
#
# IMEX pre-flight: every node's nvidia-imex health + channel0 is checked
# (and repaired if needed) EVERY time this script runs, not just once at
# deploy time -- a reboot wipes channel0 silently, so deploy-time checks
# alone aren't enough. node-00 SSHes out to itself + every sibling here.
build_run_script() {
cat <<INNER
#!/bin/bash
set -e

IMEX_CFG="${IMEX_CFG}"
SSH_OPTS="${SSH_OPTS}"
SHARE_DIR="${SHARE_DIR}"
SQSH_BASENAME="${SQSH_BASENAME}"
CONTAINER_NAME="${CONTAINER_NAME}"
export ENROOT_DATA_PATH="${ENROOT_DATA_PATH}"
export ENROOT_RUNTIME_PATH="${ENROOT_RUNTIME_PATH}"
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH}"
NODES=(
$(for ip in "${NODES[@]}"; do printf '  "%s"\n' "$ip"; done)
)
NODE_LIST_CONTENT="\$(printf '%s\\n' "\${NODES[@]}")"

echo "=== Pre-flight: checking nvidia-imex + channel0 + container on \${#NODES[@]} node(s) ==="
for ip in "\${NODES[@]}"; do
  ssh \$SSH_OPTS "root@\${ip}" "mkdir -p \$(dirname "${IMEX_CFG}") && cat > ${IMEX_CFG}" <<< "\$NODE_LIST_CONTENT"
  ssh \$SSH_OPTS "root@\${ip}" ENROOT_DATA_PATH="\$ENROOT_DATA_PATH" ENROOT_RUNTIME_PATH="\$ENROOT_RUNTIME_PATH" ENROOT_CACHE_PATH="\$ENROOT_CACHE_PATH" SHARE_DIR="\$SHARE_DIR" SQSH_BASENAME="\$SQSH_BASENAME" CONTAINER_NAME="\$CONTAINER_NAME" bash -s <<'REMOTE'
set -e
if ! systemctl is-active --quiet nvidia-imex; then
  echo "  [\$(hostname)] nvidia-imex not active -- restarting..."
  systemctl restart nvidia-imex
  ok=0
  for attempt in \$(seq 1 ${IMEX_WAIT_ATTEMPTS}); do
    systemctl is-active --quiet nvidia-imex && { ok=1; break; }
    sleep ${IMEX_WAIT_SLEEP}
  done
  if [[ \$ok -eq 0 ]]; then
    echo "ERROR: nvidia-imex did not reach 'active' state on \$(hostname)" >&2
    echo "       Check: systemctl status nvidia-imex ; journalctl -u nvidia-imex -n 50" >&2
    exit 1
  fi
fi

if [[ ! -e /dev/nvidia-caps-imex-channels/channel0 ]]; then
  echo "  [\$(hostname)] channel0 missing (lost on last reboot, as expected) -- recreating..."
  MAJOR=""
  for attempt in \$(seq 1 ${IMEX_WAIT_ATTEMPTS}); do
    MAJOR=\$(cat /proc/devices | grep nvidia-caps-imex-channels | awk '{print \$1}')
    [[ -n "\$MAJOR" ]] && break
    sleep ${IMEX_WAIT_SLEEP}
  done
  if [[ -z "\$MAJOR" ]]; then
    echo "ERROR: nvidia-caps-imex-channels major number never appeared in /proc/devices on \$(hostname)" >&2
    exit 1
  fi
  mkdir -p /dev/nvidia-caps-imex-channels
  rm -f /dev/nvidia-caps-imex-channels/channel0
  mknod /dev/nvidia-caps-imex-channels/channel0 c "\$MAJOR" 0
  chmod 0666 /dev/nvidia-caps-imex-channels/channel0
fi
test -e /dev/nvidia-caps-imex-channels/channel0 || { echo "ERROR: channel0 still missing on \$(hostname)" >&2; exit 1; }

# Safety net only -- the container persists on the data-dir volume across
# reboots, so this should normally be a no-op. Re-creates it (slow) only
# if it's genuinely gone (e.g. the data volume itself was replaced).
if ! enroot list | grep -qx "\$CONTAINER_NAME"; then
  echo "  [\$(hostname)] container '\$CONTAINER_NAME' missing on persistent storage -- recreating (this will be slow)..."
  enroot create --name "\$CONTAINER_NAME" "\$SHARE_DIR/\$SQSH_BASENAME"
fi

echo "  [\$(hostname)] nvidia-imex active, channel0 OK, container OK"
REMOTE
done
echo "=== Pre-flight complete ==="

NEXT_HOP=\$(awk 'NR==2{print \$1}' ${IMEX_CFG})
IFACE=\$(ip route get "\$NEXT_HOP" 2>/dev/null | sed -E 's/.*?dev (\S+) .*/\1/;t;d')
echo "Using interface: \$IFACE"
mkdir -p ${SHARE_DIR}/results 2>/dev/null || true
LOG="${SHARE_DIR}/results/\$(date +%Y%m%d_%H%M%S).log"

run_one() {
  local bin="\$1" args="\$2" label="\$3"
  echo "--- \$label ---"
  mpirun -np ${TOTAL_RANKS} -N ${GPUS_PER_NODE} --hostfile ${IMEX_CFG} \\
    --bind-to none --oversubscribe \\
    -x NCCL_DEBUG=WARN -x NCCL_MNNVL_ENABLE=2 -x NCCL_NVLS_ENABLE=1 \\
    -x NCCL_P2P_DISABLE=0 -x NCCL_IB_DISABLE=1 -x CUDA_IPC_HANDLE_SHARING_SUPPORT=1 \\
    -x CUDA_DEVICE_MAX_CONNECTIONS=1 -x NCCL_MNNVL_UUID=${MNNVL_UUID} -x NCCL_MIN_CTAS=32 \\
    -x ENROOT_DATA_PATH -x ENROOT_RUNTIME_PATH -x ENROOT_CACHE_PATH \\
    --mca btl tcp,self --mca btl_tcp_if_include \$IFACE --allow-run-as-root \\
    --mca coll_hcoll_enable 0 \\
    enroot start --root \\
      --mount /dev/nvidia-caps-imex-channels:/dev/nvidia-caps-imex-channels \\
      --mount /dev/nvidia-caps:/dev/nvidia-caps \\
      ${CONTAINER_NAME} -- ${CONTAINER_WORKDIR}/build/\$bin \$args
}

{
  echo "\$(date) : $(basename "$RACK_FILE" .sh) | ${#NODES[@]} nodes | ${TOTAL_RANKS} GPUs | launch \$(hostname)"
  run_one "all_reduce_perf" "-b 8 -e 32G -f 2 -g 1" "All-Reduce (${TOTAL_RANKS} GPUs)"
  run_one "alltoall_perf"  "-d uint8 -b 8 -e 32G -f 2" "All-to-All (${TOTAL_RANKS} GPUs)"
  echo "\$(date) : Done."
} 2>&1 | tee "\$LOG"
echo ""
echo "Log saved on this node: \$LOG"
INNER
}

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[DRY-RUN] would push run_nccl_test.sh (NCCL_MNNVL_UUID=${MNNVL_UUID}, container=${CONTAINER_NAME} persisted under ${DATA_DIR}, pre-flight over ${#NODES[@]} nodes) to root@${NODE00}:${SHARE_DIR}/run_nccl_test.sh"
else
  build_run_script | ssh $SSH_OPTS "root@${NODE00}" "cat > ${SHARE_DIR}/run_nccl_test.sh && chmod +x ${SHARE_DIR}/run_nccl_test.sh"
fi

echo ""
echo "=== Deploy complete: $(basename "$RACK_FILE") | node-00=${NODE00} | container=${CONTAINER_NAME} | data-dir=${DATA_DIR} ==="

if [[ $AUTO -eq 1 ]]; then
  echo "--- [5/5] --auto: executing run_nccl_test.sh on node-00 now (includes IMEX pre-flight) ---"
  mkdir -p results
  LOGFILE="results/$(basename "$RACK_FILE" .sh)_$(date +%Y%m%d_%H%M%S).log"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] would: ssh root@${NODE00} bash ${SHARE_DIR}/run_nccl_test.sh   (output -> $LOGFILE)"
  else
    ssh $SSH_OPTS "root@${NODE00}" "bash ${SHARE_DIR}/run_nccl_test.sh" | tee "$LOGFILE"
    echo "=== Log saved locally: $LOGFILE ==="
  fi
else
  cat <<MSG

Next: ssh root@${NODE00} then 'bash ${SHARE_DIR}/run_nccl_test.sh'
      (this re-checks/repairs nvidia-imex + channel0 on every node first --
      that's unavoidable on every boot. The container itself persists on
      ${DATA_DIR} across reboots, so it is NOT re-unpacked here)
      (or re-run this script with --auto to have it run for you)
MSG
fi
