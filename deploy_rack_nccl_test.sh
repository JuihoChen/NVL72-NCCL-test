#!/bin/bash
# deploy_rack_nccl_test.sh
#
# Run from the jumper, from the folder containing rackXX.sh and the
# pre-built test pack (e.g. ~/carlonext). One script does everything:
#
#   1. pushes the pack (a small tar.gz of just the *_perf binaries +
#      their .so deps, NOT the full container) to EVERY node IN PARALLEL,
#      and extracts it locally on each -- no NFS, no enroot/container at
#      test time at all. Both the cached tar.gz and the extracted folder
#      persist under --data-dir, so this never needs to be redone after a
#      reboot (only at the customer's site, potentially much later than
#      when this script ran on the production line). Pack layout is
#      assumed to be bin/ (the *_perf binaries, PLUS a custom-built
#      mpirun/orted/ompi_info/orterun -- built --without-slurm, since the
#      system/container's own mpirun is built --with-slurm and rejects
#      plain CLI options under its slurm-aware "schizo" personality) and
#      lib/ (the .so deps for both the NCCL test binaries and this pack's
#      private Open MPI build)
#   2. meshes root SSH key: node-00 -> siblings (mpirun launches node->node)
#   3. generates and stages run_nccl_test.sh directly on node-00 -- no
#      separate script needed, ready to run immediately
#   4. (optional, --auto) SSHes into node-00 and runs it for you
#
# --data-dir <path> defaults to /root/portable-nccl if not given. It's a
# plain local folder -- no separate-filesystem requirement is enforced
# (this used to refuse to share a filesystem with /, but that check has
# been removed since it doesn't apply here: nothing here modifies any
# system libraries or touches anything else on the OS image).
#
# IMPORTANT: nvidia-imex health + /dev/nvidia-caps-imex-channels/channel0
# CANNOT be made to persist across reboot under any design -- it's a
# kernel-driver-backed device node recreated fresh on every boot, on
# every system, unconditionally. So that check (cheap: a service-active
# check + an mknod) lives INSIDE run_nccl_test.sh and runs fresh every
# time the test is invoked, against every node, right before mpirun.
#
# Usage:
#   ./deploy_rack_nccl_test.sh rack17.sh
#                                                 # uses ./nccl-test-pack-arm64.tar.gz
#                                                 # and /root/portable-nccl by default
#   ./deploy_rack_nccl_test.sh rack17.sh my-pack.tar.gz --data-dir /opt/portable-nccl
#   ./deploy_rack_nccl_test.sh rack17.sh --uuid 0x1969
#   ./deploy_rack_nccl_test.sh rack17.sh --auto
#   ./deploy_rack_nccl_test.sh rack17.sh --dry-run
#   ./deploy_rack_nccl_test.sh rack17.sh --version
#   ./deploy_rack_nccl_test.sh rack17.sh --only 192.168.14.187,192.168.14.191
#                                                 # redeploy just these node(s) after a
#                                                 # hardware swap
#
# run_nccl_test.sh itself takes an optional NODE-count argument when run
# manually on node-00 -- this is intentionally a node-00-side decision,
# not a jumper flag, since it's the diag team member at the rack who
# decides how many nodes to test with for a given run (capped at however
# many nodes are in the rack, e.g. 18 for an NVL72):
#   bash run/run_nccl_test.sh        # full rack (default, e.g. 18 nodes / 72 GPUs)
#   bash run/run_nccl_test.sh 9      # half rack (9 nodes / 36 GPUs -- matches
#                                     # NVIDIA's published GB300 NVL72 spec table)
#   bash run/run_nccl_test.sh 1      # single-node smoke test (4 GPUs)
#
# NCCL_MNNVL_UUID is auto-derived from the rack filename if not given
# (rack17.sh -> 0x17, rack8.sh -> 0x08) -- a human-traceable tag, not a
# magic value; override with --uuid anytime.
#
# RE-RUN / NODE-SWAP SAFETY: this script is safe to run 2+ times on the
# same rack, including after the diag team physically swaps a node:
#   - stale SSH host keys for swapped IPs are cleared automatically
#   - the pack copy to each node is skipped if the remote file already
#     matches the local one's size (cheap re-runs)
#   - SSH key meshing is check-before-act and won't duplicate work
#   - use --only <ip1,ip2,...> to limit the pack copy+extract to just the
#     swapped node(s); if node-00 itself is in that list, the script
#     automatically falls back to a full rack pass
#   - if a replacement node gets a NEW IP, update CM_IPS in the rackXX.sh
#     file first -- everything else is driven off that array
#   - nvidia-imex/channel0 health doesn't need a redeploy at all after a
#     reboot -- run_nccl_test.sh re-checks and repairs it every time it
#     runs (this is unavoidable for channel0; it is NOT true of the pack,
#     which persists under --data-dir across reboots without any rework)

set -euo pipefail

SCRIPT_VERSION="0.9"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=8"
IMEX_CFG="/etc/nvidia-imex/nodes_config.cfg"
GPUS_PER_NODE=4
DEFAULT_PACK_NAME="nccl-test-pack-arm64.tar.gz"
DEFAULT_DATA_DIR="/root/portable-nccl"
IMEX_WAIT_ATTEMPTS=15
IMEX_WAIT_SLEEP=2

# --version short-circuits everything else, even with no rack file given
for arg in "$@"; do
  if [[ "$arg" == "--version" ]]; then
    echo "deploy_rack_nccl_test.sh version ${SCRIPT_VERSION}"
    exit 0
  fi
done

# Root-privilege check: SSH into remote nodes as root requires the caller's
# SSH identity to be trusted on the targets, which is only set up for root
# on this controller. If not already root, stop early with a clear hint.
if [[ "${EUID}" -ne 0 ]]; then
  echo "ERROR: this script must be run as root." >&2
  echo "       Please run 'sudo -s' first, then re-run this script." >&2
  exit 1
fi

RACK_FILE=""
PACK=""
DRY_RUN=0
AUTO=0
MNNVL_UUID=""
ONLY_RAW=""
DATA_DIR=""
BOOTSTRAP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bootstrap) BOOTSTRAP=1; shift ;;
    --dry-run)  DRY_RUN=1; shift ;;
    --auto)     AUTO=1; shift ;;
    --uuid)     MNNVL_UUID="$2"; shift 2 ;;
    --only)     ONLY_RAW="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    *.tar.gz)   PACK="$1"; shift ;;
    *)          RACK_FILE="$1"; shift ;;
  esac
done
[[ -n "$RACK_FILE" ]] || { echo "Usage: $0 <rack_file.sh> [pack.tar.gz] [--data-dir <path>] [--uuid 0xNNNN] [--only ip1,ip2] [--auto] [--dry-run] [--bootstrap] [--version]" >&2; exit 1; }
DATA_DIR="${DATA_DIR:-$DEFAULT_DATA_DIR}"
DATA_DIR="${DATA_DIR%/}"   # strip any trailing slash for clean path joins

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

# Default pack: ./nccl-test-pack-arm64.tar.gz next to this script
if [[ -z "$PACK" ]]; then
  PACK="$SCRIPT_DIR/$DEFAULT_PACK_NAME"
else
  [[ -f "$PACK" ]] || PACK="$SCRIPT_DIR/$PACK"
fi
[[ -f "$PACK" ]] || { echo "ERROR: pack not found: $PACK (default is ./${DEFAULT_PACK_NAME})" >&2; exit 1; }
PACK_BASENAME="$(basename "$PACK")"
PACK_CACHE="${DATA_DIR}/${PACK_BASENAME}"
PACK_DIR="${DATA_DIR}"          # pack's own build/ and lib/ land directly here
RUN_DIR="${DATA_DIR}/run"

[[ $DRY_RUN -eq 1 ]] && echo ">>> DRY-RUN MODE: no SSH/SCP/mount commands will actually run <<<"

# shellcheck source=/dev/null
source "$RACK_FILE"
[[ -n "${CM_IPS:-}" ]] || { echo "ERROR: CM_IPS array not found in $RACK_FILE" >&2; exit 1; }

# CM_IPS is listed high-IP-first (IP17..IP0); reverse so index 0 = IP0 = node-00
NODES=()
for ((i=${#CM_IPS[@]}-1; i>=0; i--)); do NODES+=("${CM_IPS[$i]}"); done
NODE00="${NODES[0]}"
TOTAL_RANKS=$(( ${#NODES[@]} * GPUS_PER_NODE ))

echo "=== Rack: $(basename "$RACK_FILE") | ${#NODES[@]} nodes | node-00=$NODE00 | pack=$PACK_BASENAME ($(stat -c%s "$PACK" 2>/dev/null || echo '?') bytes) | UUID=$MNNVL_UUID | data-dir=$DATA_DIR ==="

# --- Resolve --only into TARGET_NODES (the nodes that get the pack
# copy+extract re-applied). SSH mesh + staging always cover the full rack
# since those are cheap and idempotent anyway.
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
    echo "NOTE: node-00 ($NODE00) is in --only -- falling back to a full pass."
    TARGET_NODES=("${NODES[@]}")
  else
    echo "--- Targeted redeploy: pack copy+extract limited to: ${TARGET_NODES[*]} ---"
  fi
fi

# Swapped hardware on a reused IP means a NEW host key -- clear any stale
# cached entry for everything we're about to SSH into this run.
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

# --- [--bootstrap] First-time key seeding: push node-00's public key ------
# Use this once on a freshly imaged rack where some nodes don't yet have
# node-00's key in /root/.ssh/authorized_keys. Requires sshpass and the
# rack's root password. After this, the normal SSH mesh keeps keys in sync.
if [[ $BOOTSTRAP -eq 1 ]]; then
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] --bootstrap: would push node-00 pubkey to all ${#NODES[@]} nodes via password SSH"
  else
    if ! command -v sshpass &>/dev/null; then
      echo "ERROR: --bootstrap requires sshpass. Install it with: apt-get install -y sshpass" >&2
      exit 1
    fi
    PUBKEY_FILE="/root/.ssh/id_ed25519.pub"
    [[ -f "$PUBKEY_FILE" ]] || PUBKEY_FILE="/root/.ssh/id_rsa.pub"
    [[ -f "$PUBKEY_FILE" ]] || { echo "ERROR: no public key found at /root/.ssh/id_ed25519.pub or id_rsa.pub" >&2; exit 1; }
    PUBKEY="$(cat "$PUBKEY_FILE")"
    echo "--- [bootstrap] Seeding root SSH key to all ${#NODES[@]} node(s) ---"
    echo "    Key: $PUBKEY_FILE"
    read -rsp "    Root password for all rack nodes: " RACK_PASS; echo ""
    BOOTSTRAP_FAIL=()
    for ip in "${NODES[@]}"; do
      if sshpass -p "$RACK_PASS" ssh           -o StrictHostKeyChecking=no           -o ConnectTimeout=8           "root@${ip}"           "mkdir -p /root/.ssh && chmod 700 /root/.ssh &&            grep -qxF '${PUBKEY}' /root/.ssh/authorized_keys 2>/dev/null ||            echo '${PUBKEY}' >> /root/.ssh/authorized_keys &&            chmod 600 /root/.ssh/authorized_keys" 2>/dev/null; then
        echo "  [${ip}] key seeded OK"
      else
        echo "  [${ip}] FAILED -- wrong password or node unreachable?" >&2
        BOOTSTRAP_FAIL+=("$ip")
      fi
    done
    if [[ ${#BOOTSTRAP_FAIL[@]} -gt 0 ]]; then
      echo "" >&2
      echo "ERROR: bootstrap failed for ${#BOOTSTRAP_FAIL[@]} node(s):" >&2
      printf '  %s\n' "${BOOTSTRAP_FAIL[@]}" >&2
      echo "Check password or network reachability, then retry --bootstrap." >&2
      exit 1
    fi
    echo "--- [bootstrap] All nodes seeded. Continuing with normal deploy ---"
  fi
fi

# --- [0/4] Preflight: verify passwordless SSH to all target nodes ----------
if [[ $DRY_RUN -eq 0 ]]; then
  echo "--- [0/4] Preflight: verifying passwordless root SSH to all ${#NODES[@]} node(s) ---"
  MISSING_KEY=()
  for ip in "${NODES[@]}"; do
    if ! ssh -o BatchMode=yes -o ConnectTimeout=5 $SSH_OPTS "root@${ip}" true 2>/dev/null; then
      MISSING_KEY+=("$ip")
    fi
  done
  if [[ ${#MISSING_KEY[@]} -gt 0 ]]; then
    echo "" >&2
    echo "ERROR: passwordless SSH failed for ${#MISSING_KEY[@]} node(s):" >&2
    for ip in "${MISSING_KEY[@]}"; do
      echo "  root@${ip}" >&2
    done
    echo "" >&2
    echo "Fix: run 'ssh-copy-id root@<ip>' for each node above, then re-run this script." >&2
    exit 1
  fi
  echo "    all nodes reachable -- OK"
fi

echo "--- [1/4] Distributing + extracting pack on target node(s) (parallel) ---"
echo "    (no NFS, no container -- a plain tar.gz pushed to each node directly;"
echo "     extracted under ${DATA_DIR} on each node)"

deploy_pack_node() {
  local ip="$1"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] would copy ${PACK} to $ip if size differs, and extract to ${PACK_DIR}"
    return 0
  fi

  ssh $SSH_OPTS "root@${ip}" "mkdir -p ${PACK_DIR}"

  LOCAL_SIZE=$(stat -c%s "$PACK")
  REMOTE_SIZE=$(ssh $SSH_OPTS "root@${ip}" "stat -c%s ${PACK_CACHE} 2>/dev/null || echo 0")
  if [[ "$REMOTE_SIZE" == "$LOCAL_SIZE" ]]; then
    echo "  [$ip] pack already present with matching size (${LOCAL_SIZE} bytes) -- skipping copy"
  else
    echo "  [$ip] copying pack (~$((LOCAL_SIZE/1024/1024)) MB)..."
    scp $SSH_OPTS "$PACK" "root@${ip}:${PACK_CACHE}"
  fi

  echo "  [$ip] extracting pack to ${PACK_DIR}..."
  ssh $SSH_OPTS "root@${ip}" "mkdir -p ${PACK_DIR} && tar xzf ${PACK_CACHE} -C ${PACK_DIR}"
  echo "  [$ip] pack ready at ${PACK_DIR}"
}

if [[ $DRY_RUN -eq 1 ]]; then
  for ip in "${TARGET_NODES[@]}"; do deploy_pack_node "$ip"; done
else
  declare -A PIDS
  for ip in "${TARGET_NODES[@]}"; do
    ( deploy_pack_node "$ip" > "/tmp/.pack_deploy_${ip}.log" 2>&1 ) &
    PIDS["$ip"]=$!
  done

  FAILED=()
  for ip in "${!PIDS[@]}"; do
    wait "${PIDS[$ip]}" || FAILED+=("$ip")
    sed "s/^/[$ip] /" "/tmp/.pack_deploy_${ip}.log"
    rm -f "/tmp/.pack_deploy_${ip}.log"
  done

  if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "ERROR: pack deployment failed on: ${FAILED[*]}" >&2
    exit 1
  fi
fi

echo "--- [2/4] Meshing root SSH key: node-00 -> siblings (required for mpirun AND the pre-flight) ---"
ssh_do "$NODE00" "test -f /root/.ssh/id_ed25519 || ssh-keygen -t ed25519 -N '' -f /root/.ssh/id_ed25519 -q"
if [[ $DRY_RUN -eq 1 ]]; then
  echo "[DRY-RUN] would fetch node-00 pubkey and append to authorized_keys on: ${NODES[*]}"
else
  PUBKEY=$(ssh $SSH_OPTS "root@${NODE00}" "cat /root/.ssh/id_ed25519.pub")
  for ip in "${NODES[@]}"; do
    ssh_do "$ip" "mkdir -p /root/.ssh && grep -qF '${PUBKEY}' /root/.ssh/authorized_keys 2>/dev/null || echo '${PUBKEY}' >> /root/.ssh/authorized_keys"
  done
fi

echo "--- [3/4] Staging run_nccl_test.sh on node-00 ---"

# mpirun's hostfile IS the IMEX node list (same convention as the
# reference diag log: --hostfile /etc/nvidia-imex/nodes_config.cfg).
# Each rank runs the bare *_perf binary directly -- no container, no
# enroot at test time. Pack layout is build/ + lib/ at its top level, so
# binary/lib paths are static, not discovered.
#
# Pre-flight: every node's nvidia-imex health + channel0 is checked (and
# repaired) EVERY time this runs, since that can't persist across reboot.
# The pack itself is only verified present (cheap) -- it persists under
# --data-dir, so the slow re-extract step should essentially never fire.
build_run_script() {
cat <<INNER
#!/bin/bash
set -e

IMEX_CFG="${IMEX_CFG}"
SSH_OPTS="${SSH_OPTS}"
PACK_DIR="${PACK_DIR}"
PACK_CACHE="${PACK_CACHE}"
NODES=(
$(for ip in "${NODES[@]}"; do printf '  "%s"\n' "$ip"; done)
)
NODE_LIST_CONTENT="\$(printf '%s\\n' "\${NODES[@]}")"

# --- Node count selection -------------------------------------------------
# This is a node-00-side decision, made by whoever is running the test --
# not a jumper/deploy-time flag. Defaults to the full rack. Pass a smaller
# count of whole nodes to match NVIDIA's published GB300 NVL72 spec-sheet
# configurations (e.g. 9 nodes = 36 GPUs) or for a quick single-node
# smoke test:
#   bash run_nccl_test.sh        # full rack (default, \${#NODES[@]} nodes)
#   bash run_nccl_test.sh 9      # 9 nodes (36 GPUs, half rack)
#   bash run_nccl_test.sh 1      # 1 node (4 GPUs, smoke test)
NODES_REQUESTED="\${1:-\${#NODES[@]}}"
case "\$NODES_REQUESTED" in
  -h|--help)
    echo "Usage: \$0 [node_count]"
    echo "  node_count: number of WHOLE nodes to test with (each contributes"
    echo "  all ${GPUS_PER_NODE} of its GPUs). Default: \${#NODES[@]} (full rack)."
    echo "  Must be between 1 and \${#NODES[@]} for this rack."
    exit 0
    ;;
esac
if ! [[ "\$NODES_REQUESTED" =~ ^[0-9]+\$ ]] || [[ "\$NODES_REQUESTED" -le 0 ]]; then
  echo "ERROR: node_count must be a positive integer (got: \$NODES_REQUESTED)" >&2
  exit 1
fi
if [[ \$NODES_REQUESTED -gt \${#NODES[@]} ]]; then
  echo "ERROR: node_count (\$NODES_REQUESTED) exceeds this rack's \${#NODES[@]} nodes" >&2
  exit 1
fi
RUN_GPUS=\$(( NODES_REQUESTED * ${GPUS_PER_NODE} ))
RUN_HOSTFILE="${RUN_DIR}/hostfile_\${NODES_REQUESTED}node.cfg"
printf '%s\\n' "\${NODES[@]:0:\$NODES_REQUESTED}" > "\$RUN_HOSTFILE"
echo "Test scope: \$NODES_REQUESTED node(s) / \$RUN_GPUS GPUs (\$RUN_HOSTFILE)"

echo "=== Pre-flight: checking nvidia-imex + channel0 + pack on \${#NODES[@]} node(s) ==="
for ip in "\${NODES[@]}"; do
  ssh \$SSH_OPTS "root@\${ip}" "mkdir -p \$(dirname "${IMEX_CFG}") && cat > ${IMEX_CFG}" <<< "\$NODE_LIST_CONTENT"
  ssh \$SSH_OPTS "root@\${ip}" PACK_DIR="\$PACK_DIR" PACK_CACHE="\$PACK_CACHE" bash -s <<'REMOTE'
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

# Safety net only -- the pack persists under --data-dir across reboots,
# so this should normally be a no-op. Re-extracts from the LOCAL cached
# tar.gz (no network needed) only if it's genuinely gone.
if [[ ! -f "\$PACK_DIR/bin/all_reduce_perf" || ! -f "\$PACK_DIR/bin/alltoall_perf" || ! -f "\$PACK_DIR/bin/mpirun" || ! -f "\$PACK_DIR/bin/orted" ]]; then
  echo "  [\$(hostname)] nccl-test-pack missing/incomplete -- re-extracting from local cache..."
  if [[ ! -f "\$PACK_CACHE" ]]; then
    echo "ERROR: cached pack not found at \$PACK_CACHE on \$(hostname) -- redeploy needed" >&2
    exit 1
  fi
  mkdir -p "\$PACK_DIR"
  tar xzf "\$PACK_CACHE" -C "\$PACK_DIR"
fi

echo "  [\$(hostname)] nvidia-imex active, channel0 OK, pack OK"
REMOTE
done
echo "=== Pre-flight complete ==="

ALLREDUCE_BIN="\$PACK_DIR/bin/all_reduce_perf"
ALLTOALL_BIN="\$PACK_DIR/bin/alltoall_perf"
MPIRUN_BIN="\$PACK_DIR/bin/mpirun"
[[ -f "\$ALLREDUCE_BIN" ]] || { echo "ERROR: \$ALLREDUCE_BIN not found" >&2; exit 1; }
[[ -f "\$ALLTOALL_BIN" ]]  || { echo "ERROR: \$ALLTOALL_BIN not found" >&2; exit 1; }
[[ -f "\$MPIRUN_BIN" ]]    || { echo "ERROR: \$MPIRUN_BIN not found -- the pack must ship its own mpirun/orted built --without-slurm" >&2; exit 1; }

# This pack ships its OWN mpirun/orted/ompi_info (built --without-slurm),
# since the system/container's mpirun was built --with-slurm and rejects
# plain CLI options under its slurm-aware "schizo" personality. OPAL_PREFIX
# must point here so this relocatable Open MPI build finds its own orted
# and libs on every node (not the system's).
export OPAL_PREFIX="\$PACK_DIR"
export PATH="\$PACK_DIR/bin:\$PATH"
export LD_LIBRARY_PATH="\$PACK_DIR/lib:/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}"
echo "Binaries: \$ALLREDUCE_BIN | \$ALLTOALL_BIN | mpirun: \$MPIRUN_BIN"
echo "LD_LIBRARY_PATH=\$LD_LIBRARY_PATH"

NEXT_HOP=\$(awk 'NR==2{print \$1}' ${IMEX_CFG})
IFACE=\$(ip route get "\$NEXT_HOP" 2>/dev/null | sed -E 's/.*?dev (\S+) .*/\1/;t;d')
echo "Using interface: \$IFACE"
mkdir -p ${RUN_DIR}/results 2>/dev/null || true
LOG="${RUN_DIR}/results/\$(date +%Y%m%d_%H%M%S)_\${NODES_REQUESTED}node.log"

run_one() {
  local bin_path="\$1" args="\$2" label="\$3"
  echo "--- \$label ---"
  "\$MPIRUN_BIN" \\
    --mca schizo ompi \\
    --mca pml ob1 \\
    --mca btl tcp,self \\
    --mca btl_tcp_if_include \$IFACE \\
    --mca coll_hcoll_enable 0 \\
    -np \$RUN_GPUS \\
    -N ${GPUS_PER_NODE} \\
    --hostfile \$RUN_HOSTFILE \\
    --bind-to none \\
    --oversubscribe \\
    --allow-run-as-root \\
    -x OPAL_PREFIX="\$PACK_DIR" \\
    -x PATH="\$PACK_DIR/bin:\$PATH" \\
    -x LD_LIBRARY_PATH="\$PACK_DIR/lib:/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}" \\
    -x NCCL_DEBUG=WARN \\
    -x NCCL_MNNVL_ENABLE=2 \\
    -x NCCL_NVLS_ENABLE=1 \\
    -x NCCL_P2P_DISABLE=0 \\
    -x NCCL_IB_DISABLE=1 \\
    -x CUDA_IPC_HANDLE_SHARING_SUPPORT=1 \\
    -x CUDA_DEVICE_MAX_CONNECTIONS=1 \\
    -x NCCL_MNNVL_UUID=${MNNVL_UUID} \\
    -x NCCL_MIN_CTAS=32 \\
    "\$bin_path" \$args
}

{
  echo "\$(date) : $(basename "$RACK_FILE" .sh) | \$NODES_REQUESTED node(s) | \$RUN_GPUS GPUs | launch \$(hostname)"
  run_one "\$ALLREDUCE_BIN" "-b 8 -e 32G -f 2 -g 1" "All-Reduce (\$RUN_GPUS GPUs)"
  run_one "\$ALLTOALL_BIN"  "-d uint8 -b 8 -e 32G -f 2" "All-to-All (\$RUN_GPUS GPUs)"
  echo "\$(date) : Done."
} 2>&1 | tee "\$LOG"
echo ""
echo "Log saved on this node: \$LOG"
INNER
}

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[DRY-RUN] would push run_nccl_test.sh (NCCL_MNNVL_UUID=${MNNVL_UUID}, pack persisted under ${DATA_DIR}, pre-flight over ${#NODES[@]} nodes) to root@${NODE00}:${RUN_DIR}/run_nccl_test.sh"
else
  ssh $SSH_OPTS "root@${NODE00}" "mkdir -p ${RUN_DIR}"
  build_run_script | ssh $SSH_OPTS "root@${NODE00}" "cat > ${RUN_DIR}/run_nccl_test.sh && chmod +x ${RUN_DIR}/run_nccl_test.sh"
fi

echo ""
echo "=== Deploy complete: $(basename "$RACK_FILE") | node-00=${NODE00} | data-dir=${DATA_DIR} ==="

if [[ $AUTO -eq 1 ]]; then
  echo "--- [4/4] --auto: executing run_nccl_test.sh on node-00 now (includes pre-flight) ---"
  mkdir -p results
  LOGFILE="results/$(basename "$RACK_FILE" .sh)_$(date +%Y%m%d_%H%M%S).log"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] would: ssh root@${NODE00} bash ${RUN_DIR}/run_nccl_test.sh   (output -> $LOGFILE)"
  else
    ssh $SSH_OPTS "root@${NODE00}" "bash ${RUN_DIR}/run_nccl_test.sh" | tee "$LOGFILE"
    echo "=== Log saved locally: $LOGFILE ==="
  fi
else
  HALF_NODES=$(( ${#NODES[@]} / 2 ))
  cat <<MSG

Next: ssh root@${NODE00} then 'bash ${RUN_DIR}/run_nccl_test.sh'
      (this re-checks/repairs nvidia-imex + channel0 on every node first --
      that's unavoidable on every boot. The pack itself persists on
      ${DATA_DIR} across reboots, so it is NOT re-copied/re-extracted here)
      (or re-run this script with --auto to have it run for you)

      Optional node-count arg (whole nodes, default = full rack):
        bash ${RUN_DIR}/run_nccl_test.sh             # full rack (${#NODES[@]} nodes / ${TOTAL_RANKS} GPUs)
        bash ${RUN_DIR}/run_nccl_test.sh ${HALF_NODES}              # half rack (${HALF_NODES} nodes / $((HALF_NODES * GPUS_PER_NODE)) GPUs)
MSG
fi
