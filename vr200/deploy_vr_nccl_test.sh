#!/bin/bash
# deploy_vr_nccl_test.sh -- deploy the VR200 NCCL test pack to compute trays, from the jumper
#
# The VR200 counterpart of ../deploy_rack_nccl_test.sh, same flow:
#
#   [0/4] checks passwordless SSH jumper -> every tray
#         (--bootstrap [password] seeds the key first)
#   [1/4] pushes the pack (nccl-test-pack-vr-arm64.tar.gz, built by
#         pack/build_vr_pack.sh: *_perf binaries + a private Open MPI 4.1.6
#         --without-slurm + stock NCCL 2.32.6 + CUDA 13.5 runtime) to EVERY
#         tray IN PARALLEL and extracts it under --data-dir. A tray whose
#         cached copy has the same sha256 is not copied again. The pack
#         persists across reboots; nothing on the OS image is touched.
#   [2/4] meshes SSH tray 1 -> every tray, itself included (mpirun uses it),
#         and verifies it from tray 1
#   [3/4] stages run_vr_nccl_test.sh on node-00 with the rack settings
#         (hosts, pack paths, UUID) injected, like GB300's run_nccl_test.sh
#   [4/4] runs its pre-flight (run_vr_nccl_test.sh --check) on tray 1:
#         driver / IMEX / pack versions, IMEX + channel0, fabric health,
#         GPU P2P, pack binaries
#   --auto then runs the test itself.
#
# Usage (tray 1 = the launch node = node-00):
#   ./deploy_vr_nccl_test.sh rackNN.sh                  # CM_IPS from a GB300-style rack
#                                                       # file (high-IP first, reversed:
#                                                       # node-00 = lowest = LAST entry)
#   ./deploy_vr_nccl_test.sh --hosts ip1,ip2            # or an explicit list, tray 1 first
#   ./deploy_vr_nccl_test.sh rackNN.sh my-pack.tar.gz   # default: ./nccl-test-pack-vr-arm64.tar.gz
#   ./deploy_vr_nccl_test.sh rackNN.sh --only ip3,ip4   # re-push the pack to swapped tray(s) only
#   ./deploy_vr_nccl_test.sh rackNN.sh --bootstrap [password]
#   ./deploy_vr_nccl_test.sh rackNN.sh --auto [-- <run_vr_nccl_test.sh args>]
#   ./deploy_vr_nccl_test.sh rackNN.sh --dry-run        # print what would be done
#   ./deploy_vr_nccl_test.sh --version
#
# Other options:
#   --data-dir <path>    default /root/portable-nccl (same as GB300)
#   --user <name>        SSH and run as this user instead of root (repairs on
#                        the trays then go through sudo -n)
#   --uuid 0xNN          NCCL_MNNVL_UUID; default derived from the rack file
#                        name like GB300 (rack17.sh -> 0x17), unset with --hosts
#   --versions-warn      version mismatches in the pre-flight are warnings
#   --no-check           skip step [4/4]
#
# EXPECT_* variables set here (see the embedded run_vr_nccl_test.sh) are injected into
# the staged script and so pinned for later runs.
#
# After a reboot nothing needs redeploying: run_vr_nccl_test.sh repairs
# IMEX / channel0, clears stale processes and, if needed, re-extracts the
# pack from its local cache on every run. Re-deploy after a tray swap, a
# reimage or a new pack.

set -euo pipefail

SCRIPT_VERSION="0.4-vr"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PACK_NAME="nccl-test-pack-vr-arm64.tar.gz"
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8)

usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; }

RACK_FILE=""
PACK=""
HOSTS_RAW=""
ONLY_RAW=""
RUN_USER="root"
DATA_DIR=""
BOOTSTRAP=0
BOOTSTRAP_PASS=""
AUTO=0
DRY_RUN=0
CHECK=1
VERSIONS_WARN=0
MNNVL_UUID=""
RUN_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hosts)          HOSTS_RAW="$2"; shift 2 ;;
    --only)           ONLY_RAW="$2"; shift 2 ;;
    --user)           RUN_USER="$2"; shift 2 ;;
    --data-dir)       DATA_DIR="${2%/}"; shift 2 ;;
    --bootstrap)
      BOOTSTRAP=1; shift
      if [[ $# -gt 0 && "$1" != -* && "$1" != *.sh && "$1" != *.tar.gz ]]; then BOOTSTRAP_PASS="$1"; shift; fi ;;
    --auto)           AUTO=1; shift ;;
    --dry-run)        DRY_RUN=1; shift ;;
    --no-check)       CHECK=0; shift ;;
    --versions-warn)  VERSIONS_WARN=1; shift ;;
    --uuid)           MNNVL_UUID="$2"; shift 2 ;;
    --version)        echo "deploy_vr_nccl_test.sh version ${SCRIPT_VERSION}"; exit 0 ;;
    -h|--help)        usage; exit 0 ;;
    --)               shift; RUN_ARGS=("$@"); break ;;
    -*)               echo "ERROR: unknown option: $1 (see --help)" >&2; exit 1 ;;
    *.tar.gz)         PACK="$1"; shift ;;
    *)                RACK_FILE="$1"; shift ;;
  esac
done


# --- Pack -----------------------------------------------------------------------
if [[ -z "$PACK" ]]; then PACK="$SCRIPT_DIR/$DEFAULT_PACK_NAME"
elif [[ ! -f "$PACK" ]]; then PACK="$SCRIPT_DIR/$PACK"; fi
[[ -f "$PACK" ]] || { echo "ERROR: pack not found: $PACK (build it with pack/build_vr_pack.sh)" >&2; exit 1; }
PACK_INFO=$(tar -xzOf "$PACK" ./PACK_INFO 2>/dev/null || tar -xzOf "$PACK" PACK_INFO 2>/dev/null || true)
[[ -n "$PACK_INFO" ]] || { echo "ERROR: $PACK has no PACK_INFO -- not a VR200 pack from pack/build_vr_pack.sh?" >&2; exit 1; }
PACK_SHA=$(sha256sum "$PACK" | cut -d' ' -f1)
PACK_BASENAME=$(basename "$PACK")

# --- Tray list: tray 1 (index 0) is the launch node -----------------------
NODES=()
if [[ -n "$HOSTS_RAW" ]]; then
  [[ -z "$RACK_FILE" ]] || { echo "ERROR: give either a rack file or --hosts, not both" >&2; exit 1; }
  IFS=',' read -ra NODES <<< "$HOSTS_RAW"
elif [[ -n "$RACK_FILE" ]]; then
  [[ -f "$RACK_FILE" ]] || RACK_FILE="$SCRIPT_DIR/$RACK_FILE"
  [[ -f "$RACK_FILE" ]] || { echo "ERROR: rack file not found: $RACK_FILE" >&2; exit 1; }
  # shellcheck source=/dev/null
  source "$RACK_FILE"
  [[ -n "${CM_IPS:-}" ]] || { echo "ERROR: CM_IPS array not found in $RACK_FILE" >&2; exit 1; }
  # Same convention as deploy_rack_nccl_test.sh: CM_IPS is high-IP first, reversed here
  for ((i=${#CM_IPS[@]}-1; i>=0; i--)); do NODES+=("${CM_IPS[$i]}"); done
  # Same UUID rule as GB300: rack17.sh -> 0x17 (base 10, leading zeros are not octal)
  if [[ -z "$MNNVL_UUID" ]]; then
    RACK_NUM=$(basename "$RACK_FILE" .sh | grep -oE '[0-9]+' | head -1 || true)
    [[ -n "$RACK_NUM" ]] && MNNVL_UUID="0x$(printf '%02d' "$((10#$RACK_NUM))")"
  fi
else
  echo "Usage: $0 <rack_file.sh> | --hosts ip1,ip2,...  [pack.tar.gz] [options]  (see --help)" >&2
  exit 1
fi
for ip in "${NODES[@]}"; do
  [[ "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "ERROR: not an IPv4 address: '$ip'" >&2; exit 1; }
done
if [[ $(printf '%s\n' "${NODES[@]}" | sort -u | wc -l) -ne ${#NODES[@]} ]]; then
  echo "ERROR: the tray list has duplicate IPs: ${NODES[*]}" >&2; exit 1
fi
NODE00="${NODES[0]}"
HOSTS_CSV="$(IFS=,; echo "${NODES[*]}")"

# --only: limit the pack copy to these trays (mesh + staging always cover all)
TARGET_NODES=("${NODES[@]}")
if [[ -n "$ONLY_RAW" ]]; then
  IFS=',' read -ra TARGET_NODES <<< "$ONLY_RAW"
  for want in "${TARGET_NODES[@]}"; do
    printf '%s\n' "${NODES[@]}" | grep -qxF "$want" || { echo "ERROR: --only IP '$want' is not in the tray list" >&2; exit 1; }
  done
fi

ssh_to() { local h="$1"; shift; ssh "${SSH_OPTS[@]}" "${RUN_USER}@${h}" "$@"; }
# ssh joins its arguments into one remote command line: quote each one
# (printf '%q ' with no arguments would print '' , an empty argument)
qargs() { [[ $# -eq 0 ]] || printf '%q ' "$@"; }

echo "=== deploy_vr_nccl_test.sh ${SCRIPT_VERSION} | ${#NODES[@]} tray(s) | node-00=${NODE00} | user=${RUN_USER} | UUID=${MNNVL_UUID:-auto} ==="
echo "    trays: ${NODES[*]}"
echo "    pack:  ${PACK_BASENAME} ($(du -h "$PACK" | cut -f1), sha256 ${PACK_SHA:0:16}...)"
echo "$PACK_INFO" | grep -E '^(nccl|cuda|openmpi|nccl_tests|built)=' | sed 's/^/           /'
[[ $DRY_RUN -eq 1 ]] && echo ">>> DRY-RUN: nothing is changed on any tray <<<"

# --- [bootstrap] seed the jumper's key with a password, once --------------
if [[ $BOOTSTRAP -eq 1 ]]; then
  PUBKEY_FILE="$HOME/.ssh/id_ed25519.pub"; [[ -f "$PUBKEY_FILE" ]] || PUBKEY_FILE="$HOME/.ssh/id_rsa.pub"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] --bootstrap: would append $PUBKEY_FILE to ${RUN_USER}'s authorized_keys on ${#NODES[@]} tray(s)"
  else
    command -v sshpass >/dev/null || { echo "ERROR: --bootstrap needs sshpass (apt-get install -y sshpass)" >&2; exit 1; }
    [[ -f "$PUBKEY_FILE" ]] || { echo "ERROR: no ~/.ssh/id_ed25519.pub or id_rsa.pub on the jumper (ssh-keygen -t ed25519)" >&2; exit 1; }
    PUBKEY="$(cat "$PUBKEY_FILE")"
    if [[ -n "$BOOTSTRAP_PASS" ]]; then PASS="$BOOTSTRAP_PASS"
    else read -rsp "    Password for ${RUN_USER} on all trays: " PASS; echo ""; fi
    echo "--- [bootstrap] seeding $PUBKEY_FILE for ${RUN_USER} ---"
    BOOT_FAIL=()
    for ip in "${NODES[@]}"; do
      if timeout 20 sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
           -o LogLevel=ERROR -o NumberOfPasswordPrompts=1 -o PreferredAuthentications=password \
           -o PubkeyAuthentication=no -o ConnectTimeout=8 "${RUN_USER}@${ip}" \
           "umask 077; mkdir -p ~/.ssh && touch ~/.ssh/authorized_keys && { grep -qxF '${PUBKEY}' ~/.ssh/authorized_keys || echo '${PUBKEY}' >> ~/.ssh/authorized_keys; }" 2>/dev/null; then
        echo "  [$ip] key seeded"
      else echo "  [$ip] FAILED (wrong password, unreachable, or password login disabled?)" >&2; BOOT_FAIL+=("$ip"); fi
    done
    [[ ${#BOOT_FAIL[@]} -eq 0 ]] || { echo "ERROR: bootstrap failed for: ${BOOT_FAIL[*]}" >&2; exit 1; }
  fi
fi

# --- [0/4] jumper -> trays ---------------------------------------------------
echo "--- [0/4] Passwordless SSH jumper -> ${#NODES[@]} tray(s) as ${RUN_USER} ---"
if [[ $DRY_RUN -eq 1 ]]; then
  echo "[DRY-RUN] would clear stale known_hosts entries and run: ssh ${RUN_USER}@<ip> true"
else
  # A swapped tray on a reused IP has a new host key
  for ip in "${NODES[@]}"; do ssh-keygen -R "$ip" >/dev/null 2>&1 || true; done
  MISSING=()
  for ip in "${NODES[@]}"; do ssh_to "$ip" true 2>/dev/null || MISSING+=("$ip"); done
  if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "ERROR: passwordless SSH as ${RUN_USER} failed to: ${MISSING[*]}" >&2
    echo "       Re-run with --bootstrap [password], or ssh-copy-id ${RUN_USER}@<ip> for each." >&2
    exit 1
  fi
  echo "    OK"
fi
if [[ -z "$DATA_DIR" ]]; then
  if [[ "$RUN_USER" == root ]]; then DATA_DIR=/root/portable-nccl
  elif [[ $DRY_RUN -eq 1 ]]; then DATA_DIR="/home/$RUN_USER/portable-nccl"
  else DATA_DIR="$(ssh_to "$NODE00" 'echo $HOME')/portable-nccl"; fi
fi
PACK_DIR="$DATA_DIR"                     # bin/ lib/ share/ land directly here, like GB300
PACK_CACHE="$DATA_DIR/$PACK_BASENAME"
RUN_DIR="$DATA_DIR/run"
echo "    data-dir on every tray: $DATA_DIR"

# --- [1/4] pack -> every target tray, in parallel ------------------------------
echo "--- [1/4] Pushing + extracting the pack on ${#TARGET_NODES[@]} tray(s) (parallel) ---"
deploy_pack_node() {
  local ip="$1" have
  have=$(ssh_to "$ip" "sha256sum '$PACK_CACHE' 2>/dev/null | cut -d' ' -f1" || true)
  if [[ "$have" == "$PACK_SHA" ]]; then
    echo "pack already cached (same sha256) -- skipping copy"
  else
    echo "copying pack ($(du -h "$PACK" | cut -f1))"
    ssh_to "$ip" "mkdir -p '$DATA_DIR' && cat > '$PACK_CACHE'.part.\$\$ && mv '$PACK_CACHE'.part.\$\$ '$PACK_CACHE'" < "$PACK"
  fi
  # Old bin/ lib/ share/ are replaced, so libraries of an older pack cannot linger
  ssh_to "$ip" "cd '$PACK_DIR' && rm -rf bin lib share PACK_INFO PACK_LIBS && tar xzf '$PACK_CACHE' && test -x bin/mpirun -a -x bin/orted -a -x bin/all_reduce_perf"
  echo "pack ready at $PACK_DIR"
}
if [[ $DRY_RUN -eq 1 ]]; then
  for ip in "${TARGET_NODES[@]}"; do
    echo "[DRY-RUN] $ip: copy $PACK_BASENAME to $PACK_CACHE unless its sha256 matches, extract into $PACK_DIR"
  done
else
  TMPD=$(mktemp -d); trap 'rm -rf "$TMPD"' EXIT
  declare -A PIDS
  for ip in "${TARGET_NODES[@]}"; do
    ( deploy_pack_node "$ip" > "$TMPD/$ip.log" 2>&1 ) & PIDS[$ip]=$!
  done
  FAILED=()
  for ip in "${TARGET_NODES[@]}"; do
    wait "${PIDS[$ip]}" || FAILED+=("$ip")
    sed "s/^/  [$ip] /" "$TMPD/$ip.log"
  done
  [[ ${#FAILED[@]} -eq 0 ]] || { echo "ERROR: pack deployment failed on: ${FAILED[*]}" >&2; exit 1; }
fi

# --- [2/4] tray 1 -> every tray (mpirun + the runner's pre-flight use this) --
echo "--- [2/4] SSH mesh: node-00 (${NODE00}) -> every tray, itself included ---"
if [[ $DRY_RUN -eq 1 ]]; then
  echo "[DRY-RUN] would create ~/.ssh/id_ed25519 on node-00 if missing, add its public key on every tray, verify from node-00"
else
  T1KEY=$(ssh_to "$NODE00" "test -f ~/.ssh/id_ed25519 || ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519 -q; cat ~/.ssh/id_ed25519.pub")
  for ip in "${NODES[@]}"; do
    ssh_to "$ip" "umask 077; mkdir -p ~/.ssh && touch ~/.ssh/authorized_keys && { grep -qxF '${T1KEY}' ~/.ssh/authorized_keys || echo '${T1KEY}' >> ~/.ssh/authorized_keys; }"
  done
  # Verify with node-00's own client: a missing link would hang mpirun silently.
  # Old host keys of these IPs on node-00 are cleared too (tray swap).
  if ! MESH=$(ssh_to "$NODE00" "bash -s -- $(qargs "${NODES[@]}")" <<'EOF'
bad=""
for ip in "$@"; do
  ssh-keygen -R "$ip" >/dev/null 2>&1 || true
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8 "$ip" true 2>/dev/null || bad="$bad $ip"
done
[[ -z "$bad" ]] || { echo "$bad"; exit 1; }
EOF
  ); then
    echo "ERROR: node-00 cannot SSH (as ${RUN_USER}) to:${MESH}" >&2
    echo "       Check sshd on those trays and /etc/ssh/ssh_config on node-00." >&2
    exit 1
  fi
  echo "    node-00 -> all ${#NODES[@]} tray(s): OK"
fi

# --- [3/4] stage the runner + its config on tray 1 ---------------------------
echo "--- [3/4] Staging run_vr_nccl_test.sh (rack settings injected) in ${RUN_DIR} on node-00 ---"
CONF="# Injected by deploy_vr_nccl_test.sh ${SCRIPT_VERSION}, $(date -u '+%Y-%m-%d %H:%M UTC')
# source: ${RACK_FILE:-"--hosts"}, pack ${PACK_BASENAME} sha256 ${PACK_SHA:0:16}
# Environment variables win over these values.
HOSTS=\"\${HOSTS:-$HOSTS_CSV}\"
PACK_DIR=\"\${PACK_DIR:-$PACK_DIR}\"
PACK_CACHE=\"\${PACK_CACHE:-$PACK_CACHE}\"
RESULTS_DIR=\"\${RESULTS_DIR:-$RUN_DIR/results}\"
"
[[ -n "$MNNVL_UUID" ]] && CONF+="MNNVL_UUID=\"\${MNNVL_UUID:-$MNNVL_UUID}\""$'\n'
[[ $VERSIONS_WARN -eq 1 ]] && CONF+="VERSIONS_WARN=\"\${VERSIONS_WARN:-1}\""$'\n'
for v in EXPECT_DRIVER EXPECT_CUDA EXPECT_NCCL EXPECT_NCCL_CUDA EXPECT_IMEX EXPECT_NVSHMEM EXPECT_DOCA; do
  # Only versions set on the jumper are pinned; otherwise the runner's defaults apply
  [[ -n "${!v+x}" ]] && CONF+="$v=\"\${$v-${!v}}\""$'\n'
done
# The runner that node-00 gets, embedded here like GB300's build_run_script:
# the jumper needs only this script and the pack tarball. The deploy puts the
# rack settings ($CONF) in place of its "# @@DEPLOY_SETTINGS@@" line.
read -r -d '' RUNNER_SH <<'VR_RUNNER_EOF' || true
#!/bin/bash
# run_vr_nccl_test.sh -- multi-tray NCCL test for VR200 compute trays, launched from node-00
#
# Staged on node-00 (tray 1) by ../deploy_vr_nccl_test.sh, which injects the
# rack settings (host list, pack location, UUID) into the staged copy. The VR200
# counterpart of the run_nccl_test.sh that deploy_rack_nccl_test.sh generates
# for GB300, and the same model:
#   - every tray runs the test PACK (nccl-test-pack-vr-arm64.tar.gz, built by
#     pack/build_vr_pack.sh) extracted under PACK_DIR: its own *_perf
#     binaries, its own Open MPI 4.1.6 (--without-slurm; mpirun/orted, with
#     OPAL_PREFIX=PACK_DIR) and its own stock NCCL 2.32.6 + CUDA 13.5 runtime.
#     Only the GPU driver and IMEX come from the tray.
#   - root (like GB300) or a normal user with passwordless SSH to every tray,
#     itself included; repairs then go through "sudo -n"
#   - NCCL settings as GB300 (NVLS on, IB off), except MNNVL is REQUIRED on
#     multi-tray runs: the test is meant to prove NVLink between trays, so no
#     silent fallback to sockets is allowed
#
# Run it ON NODE-00 (the first host in the list):
#   bash run_vr_nccl_test.sh                 # all trays
#   bash run_vr_nccl_test.sh 1               # node-00 only (4 GPUs, smoke test)
#   bash run_vr_nccl_test.sh --info          # NCCL INFO -> /tmp/nccl_info/ on each tray
#   bash run_vr_nccl_test.sh --check         # pre-flight only (no repairs), print mpirun
#   bash run_vr_nccl_test.sh --max-size 8G   # shorter sweep (default 32G)
#   bash run_vr_nccl_test.sh --hosts 10.0.0.1,10.0.0.2
#   bash run_vr_nccl_test.sh --versions-warn # version mismatches are warnings, not failures
#   (default on 2+ trays: NCCL_MNNVL_ENABLE=1, so NCCL init fails if NVLink between
#    trays is unavailable; 1 tray: 2 = auto. Multi-tray runs also fail if peak busbw
#    at >= 64 MB < MIN_MNNVL_BUSBW, default 100 GB/s, and with --info if any NET/
#    transport is seen)
#   bash run_vr_nccl_test.sh --mnnvl-auto    # NCCL_MNNVL_ENABLE=2 (the GB300 default): may fall
#                                            # back to sockets; the busbw floor still applies
#   bash run_vr_nccl_test.sh --allow-ib      # let NCCL use IB/RoCE between trays
#   bash run_vr_nccl_test.sh --uuid 0x01     # set NCCL_MNNVL_UUID
#   bash run_vr_nccl_test.sh --no-nvls       # NCCL_NVLS_ENABLE=0: skip NVLink-switch multicast
#                                            # (isolates multicast problems from plain MNNVL)
#   bash run_vr_nccl_test.sh --no-mnnvl      # NCCL_MNNVL_ENABLE=0 (+ no NVLS): control run that
#                                            # avoids NVLink between trays (TCP, slow) to prove
#                                            # MPI/NCCL/tests are fine apart from the fabric
#                                            # (default max size then 64M)
#
# Settings, lowest to highest priority: built-in defaults < vr_nccl.conf < injected rack settings <
# environment < command line. Environment / conf variables:
#   HOSTS        comma-separated tray IPs, node-00 first (or use --hosts)
#   PACK_DIR     extracted pack (default /root/portable-nccl, or ~/portable-nccl)
#   PACK_CACHE   the pack's tar.gz, re-extracted from if PACK_DIR is damaged
#   RESULTS_DIR  (default PACK_DIR/run/results)
#   Expected versions (substring match; set one to "" to skip that check):
#   EXPECT_DRIVER 620.70.08 (tray)   EXPECT_IMEX 2.0.0 (tray)      -> [FAIL]
#   EXPECT_NCCL 2.32.6 + EXPECT_NCCL_CUDA cuda13.5, EXPECT_CUDA 13.5.0
#                (the pack's PACK_INFO)                             -> [FAIL]
#   EXPECT_NVSHMEM 3.8.0, EXPECT_DOCA 3.3.3 (tray, informational)  -> [WARN]

set -euo pipefail

SCRIPT_VERSION="0.10-vr"
GPUS_PER_NODE=4
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# --- Rack settings: deploy_vr_nccl_test.sh injects them on the next line when
# it stages this script on node-00 (like GB300's generated run_nccl_test.sh).
# @@DEPLOY_SETTINGS@@
# --- end of rack settings ---
# Optional extra settings file (standalone use): vr_nccl.conf next to this script
CONF="${VR_NCCL_CONF:-$SCRIPT_DIR/vr_nccl.conf}"
# The conf only fills variables the environment has not set ("${X:-...}")
# shellcheck source=/dev/null
[[ -f "$CONF" ]] && source "$CONF"
PACK_DIR="${PACK_DIR:-$([[ $EUID -eq 0 ]] && echo /root/portable-nccl || echo "$HOME/portable-nccl")}"
PACK_CACHE="${PACK_CACHE:-$PACK_DIR/nccl-test-pack-vr-arm64.tar.gz}"
RESULTS_DIR="${RESULTS_DIR:-$PACK_DIR/run/results}"
# Target: VR NVL72 1.0.0 RC compute tray stack. ${X-default} (no colon) so
# that EXPECT_X="" really means "skip this check".
EXPECT_DRIVER="${EXPECT_DRIVER-620.70.08}"
EXPECT_CUDA="${EXPECT_CUDA-13.5.0}"
EXPECT_NCCL="${EXPECT_NCCL-2.32.6}"
EXPECT_NCCL_CUDA="${EXPECT_NCCL_CUDA-cuda13.5}"
EXPECT_IMEX="${EXPECT_IMEX-2.0.0}"
EXPECT_NVSHMEM="${EXPECT_NVSHMEM-3.8.0}"
EXPECT_DOCA="${EXPECT_DOCA-3.3.3}"
IMEX_CFG="/etc/nvidia-imex/nodes_config.cfg"
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8)

HOSTS_RAW="${HOSTS:-}"
VERSIONS_WARN="${VERSIONS_WARN:-0}"
INFO=0
DRY_RUN=0
MAX_SIZE=""
MNNVL_ENABLE=""     # 1 = required (default on 2+ trays), 2 = auto (default on 1 tray), 0 = off
# Multi-tray runs with MNNVL on must reach this peak busbw (GB/s) at >= 64 MB,
# else NVLink between trays was not (fully) used (see the summary check)
MIN_MNNVL_BUSBW="${MIN_MNNVL_BUSBW:-100}"
IB_DISABLE=1
NVLS_ENABLE=1
MNNVL_UUID="${MNNVL_UUID:-}"
NODES_REQUESTED=""

usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hosts)         HOSTS_RAW="$2"; shift 2 ;;
    --info)          INFO=1; shift ;;
    --dry-run|--check) DRY_RUN=1; shift ;;
    --versions-warn) VERSIONS_WARN=1; shift ;;
    --max-size)      MAX_SIZE="$2"; shift 2 ;;
    --require-mnnvl) MNNVL_ENABLE=1; shift ;;
    --mnnvl-auto)    MNNVL_ENABLE=2; shift ;;
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
MAX_SIZE="${MAX_SIZE:-$([[ $MNNVL_ENABLE == 0 ]] && echo 64M || echo 32G)}"

if [[ -z "$HOSTS_RAW" ]]; then
  echo "ERROR: no hosts: pass --hosts ip1,ip2,... (tray 1 first) or stage with deploy_vr_nccl_test.sh" >&2
  exit 1
fi
IFS=',' read -ra NODES <<< "$HOSTS_RAW"
NODES_REQUESTED="${NODES_REQUESTED:-${#NODES[@]}}"
if ! [[ "$NODES_REQUESTED" =~ ^[0-9]+$ ]] || (( NODES_REQUESTED < 1 || NODES_REQUESTED > ${#NODES[@]} )); then
  echo "ERROR: node count must be 1..${#NODES[@]} (got: $NODES_REQUESTED)" >&2
  exit 1
fi
RUN_NODES=("${NODES[@]:0:$NODES_REQUESTED}")
# NVLink robustness is the point of a multi-tray run, so MNNVL is required there
[[ -z "$MNNVL_ENABLE" ]] && MNNVL_ENABLE=$(( NODES_REQUESTED > 1 ? 1 : 2 ))
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
  # also honoured by the remote orted daemons / ranks
  export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
fi
echo "pack: ${PACK_DIR} | results: ${RESULTS_DIR}"

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
#   PREFLIGHT|<rc>|<driver>|<pack nccl>|<fabric>|<cluster>|<if>|<cidr>|<pack sha>|<pack cuda>|<imex>|<nvlinks/GPU>
read -r -d '' REMOTE_PREFLIGHT <<'REMOTE' || true
set -uo pipefail
EXPECT_DRIVER="$1"; EXPECT_NCCL="$2"; PACK_DIR="$3"; PACK_CACHE="$4"
IMEX_CFG="$5"; HOSTLIST="$6"; DRY_RUN="$7"; INFO="$8"; MULTI="$9"; MYIP="${10}"
EXPECT_CUDA="${11}"; EXPECT_NCCL_CUDA="${12}"; EXPECT_IMEX="${13}"
EXPECT_NVSHMEM="${14}"; EXPECT_DOCA="${15}"; VWARN="${16}"
# Label nNN/IP (position in the host list, n00 = node-00), like GB300 v0.24:
# trays may share one hostname, which is fine with the pack's Open MPI 4
TAG="[$(echo "$HOSTLIST" | tr ',' '\n' | awk -v me="$MYIP" '$1==me {printf "n%02d/%s", NR-1, me; f=1} END{if(!f) printf "%s", me}')]"
RC=0
ok()   { echo "  $TAG [OK]   $*"; }
warn() { echo "  $TAG [WARN] $*"; }
bad()  { echo "  $TAG [FAIL] $*"; RC=1; }
vbad() { if [[ "$VWARN" -eq 1 ]]; then warn "$*"; else bad "$*"; fi; }   # version mismatch
priv() { if [[ $EUID -eq 0 ]]; then "$@"; else sudo -n "$@"; fi; }
# installed dpkg packages matching an ERE on the package name: "<pkg> <version>" lines
pkgs() { dpkg-query -W -f='${db:Status-Abbrev}|${Package} ${Version}\n' 2>/dev/null \
           | awk -F'|' -v re="$1" '$1 ~ /^ii/ { split($2, a, " "); if (a[1] ~ re) print $2 }'; }

# Driver: user space + kernel module must both match (the 615/620 mix-up)
DRV="?"; KMD="?"
if OUT=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1); then
  DRV=$(echo "$OUT" | head -1 | tr -d ' ')
  NGPU=$(echo "$OUT" | wc -l)
  KMD=$(grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' /proc/driver/nvidia/version 2>/dev/null | head -1)
  if [[ "$DRV" != "$KMD" ]]; then
    bad "driver user-space=$DRV but kernel module=$KMD (two drivers installed? see README finding 1)"
  elif [[ -n "$EXPECT_DRIVER" && "$DRV" != "$EXPECT_DRIVER" ]]; then
    vbad "driver $DRV, expected $EXPECT_DRIVER"
  else ok "driver $DRV (user space = kernel), $NGPU GPUs"; fi
  [[ "$NGPU" -eq 4 ]] || bad "$NGPU GPUs visible, expected 4"
else
  bad "nvidia-smi failed: $(echo "$OUT" | head -1)  (check: journalctl -k -b | grep NVRM)"
fi

# The pack: present (re-extracted from the local cache if not, like GB300),
# and its PACK_INFO says what NCCL / CUDA runtime / Open MPI it carries
if [[ ! -x "$PACK_DIR/bin/mpirun" || ! -x "$PACK_DIR/bin/orted" || ! -x "$PACK_DIR/bin/all_reduce_perf" \
      || ! -x "$PACK_DIR/bin/alltoall_perf" || ! -f "$PACK_DIR/PACK_INFO" ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then warn "pack missing/incomplete in $PACK_DIR (would re-extract $PACK_CACHE)"
  elif [[ -f "$PACK_CACHE" ]] && (cd "$PACK_DIR" && rm -rf bin lib share PACK_INFO PACK_LIBS && tar xzf "$PACK_CACHE"); then
    ok "pack was incomplete -- re-extracted from $PACK_CACHE"
  else bad "pack missing in $PACK_DIR and no usable cache at $PACK_CACHE -- re-run deploy_vr_nccl_test.sh"; fi
fi
pinfo() { awk -F= -v k="$1" '$1==k {sub(/^[^=]*=/, ""); print; exit}' "$PACK_DIR/PACK_INFO" 2>/dev/null; }
NCCL=$(pinfo nccl); NCCL="${NCCL:-none}"
CUDA=$(pinfo cuda); CUDA="${CUDA:-none}"
PSHA=$(sha256sum "$PACK_CACHE" 2>/dev/null | cut -c1-12); PSHA="${PSHA:-none}"
if [[ ( -z "$EXPECT_NCCL" || "$NCCL" == *"$EXPECT_NCCL"* ) && ( -z "$EXPECT_NCCL_CUDA" || "$NCCL" == *"$EXPECT_NCCL_CUDA"* ) ]]; then
  ok "pack NCCL $NCCL"
else vbad "pack NCCL is '$NCCL', expected $EXPECT_NCCL (+$EXPECT_NCCL_CUDA)"; fi
if [[ -z "$EXPECT_CUDA" || "$CUDA" == "$EXPECT_CUDA"* ]]; then ok "pack CUDA runtime $CUDA, Open MPI $(pinfo openmpi), nccl-tests $(pinfo nccl_tests)"
else vbad "pack CUDA runtime is '$CUDA', expected $EXPECT_CUDA"; fi

# IMEX: needed for memory sharing between trays, so a mismatch fails multi-tray runs only
IMEX=$(pkgs '^nvidia-imex' | head -1); IMEX="${IMEX:-none}"
IMEX_V=$(nvidia-imex --version 2>/dev/null | head -3 | xargs)
if [[ -z "$EXPECT_IMEX" || "$IMEX $IMEX_V" == *"$EXPECT_IMEX"* ]]; then ok "IMEX $IMEX${IMEX_V:+ ($IMEX_V)}"
elif [[ "$MULTI" -eq 1 ]]; then vbad "IMEX is '$IMEX${IMEX_V:+ / $IMEX_V}', expected $EXPECT_IMEX"
else warn "IMEX is '$IMEX${IMEX_V:+ / $IMEX_V}', expected $EXPECT_IMEX (not used by a 1-tray run)"; fi

# Informational: not used by nccl-tests, reported so the stack is on record
NVSHMEM=$(pkgs '^(lib)?nvshmem' | head -1); NVSHMEM="${NVSHMEM:-none}"
if [[ -z "$EXPECT_NVSHMEM" || "$NVSHMEM" == *"$EXPECT_NVSHMEM"* ]]; then ok "NVSHMEM $NVSHMEM"
else warn "NVSHMEM is '$NVSHMEM', expected $EXPECT_NVSHMEM"; fi
DOCA=$(pkgs '^doca-host$' | head -1); [[ -n "$DOCA" ]] || DOCA=$(pkgs '^doca-' | head -1); DOCA="${DOCA:-none}"
if [[ -z "$EXPECT_DOCA" || "$DOCA" == *"$EXPECT_DOCA"* ]]; then ok "DOCA $DOCA"
else warn "DOCA is '$DOCA', expected DOCA_Host $EXPECT_DOCA"; fi

# Pack binaries: every library resolves (pack lib/ + the tray's base OS and
# driver), and MPI / NCCL come from the pack, not from the tray
RC_SAVE=$RC; RC=0
for b in all_reduce_perf alltoall_perf mpirun orted; do
  BIN="$PACK_DIR/bin/$b"
  [[ -x "$BIN" ]] || { bad "$BIN missing"; continue; }
  LDD=$(LD_LIBRARY_PATH="$PACK_DIR/lib" ldd "$BIN" 2>&1)
  echo "$LDD" | grep -q "not found" && bad "$b: missing libs on this tray: $(echo "$LDD" | awk '/not found/{print $1}' | xargs)"
  for l in libnccl.so libmpi.so libopen-pal.so; do
    LIB=$(echo "$LDD" | awk -v l="$l" 'index($1, l)==1 {print $3; exit}')
    [[ -z "$LIB" || "$LIB" == "$PACK_DIR/lib/"* ]] || bad "$b loads $l from '$LIB', not the pack"
  done
done
[[ $RC -eq 0 ]] && ok "pack binaries OK ($PACK_DIR/bin)"
RC=$(( RC | RC_SAVE ))

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

# Per-link NVLink state: no link down/inactive, and every GPU with the same
# number of active links (one dead link shows up here before it costs busbw)
NVL=$(nvidia-smi nvlink -s 2>/dev/null)
LINKS="?"
if [[ -n "$NVL" ]]; then
  read -r NG LMIN LMAX DOWN <<< "$(echo "$NVL" | awk '
    /^GPU [0-9]+/ {g++; a[g]=0; next}
    /Link [0-9]+:/ { if (tolower($0) ~ /inactive|down|n\/a/) d++; else a[g]++ }
    END { min=-1; max=0; for (i=1; i<=g; i++) { if (min<0 || a[i]<min) min=a[i]; if (a[i]>max) max=a[i] }
          print g+0, (min<0 ? 0 : min), max, d+0 }')"
  LINKS="$LMAX"
  if [[ "$DOWN" -eq 0 && "$LMIN" -eq "$LMAX" && "$LMAX" -gt 0 ]]; then ok "NVLink: $NG GPUs x $LMAX active links each"
  else bad "NVLink: $DOWN link(s) inactive/down, active links per GPU $LMIN..$LMAX (nvidia-smi nvlink -s)"; fi
else
  warn "nvidia-smi nvlink -s gave no output (NVLink state not checked)"
fi

# Finding 4: GPUs registered with different NVLink bandwidth modes refuse
# cross-tray memory import. The kernel log says so; reboot both trays once the
# partition is healthy. dmesg may need root: skipped if unreadable.
if DM=$(dmesg 2>/dev/null || priv dmesg 2>/dev/null); then
  BW=$(echo "$DM" | grep -c "bwMode does not match" || true)
  [[ "$BW" -eq 0 ]] && ok "no NVLink bwMode mismatch in dmesg" \
    || bad "dmesg: $BW 'bwMode does not match' line(s) since boot -- cross-tray NVLink will fail; reboot the trays (README finding 4)"
fi


# Stale test processes from an aborted run ("CUDA-capable device(s) is/are busy")
PIDS=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
  | grep -E 'all_reduce_perf|alltoall_perf|orted|mpirun' | awk -F, '{print $1}' | xargs)
if [[ -n "$PIDS" ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then warn "stale test processes on GPU: $PIDS (would kill)"
  else
    kill -9 $PIDS 2>/dev/null || priv kill -9 $PIDS 2>/dev/null || true
    sleep 2
    if nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null \
        | grep -qE 'all_reduce_perf|alltoall_perf|orted|mpirun'; then
      bad "stale test processes survived kill -9 -- check dmesg for Xid"
    else ok "cleared stale processes: $PIDS"; fi
  fi
fi

# PMIx "unable to find a usable compression library" startup noise (cosmetic)
mkdir -p "$HOME/.pmix"
grep -qxF 'pcompress_base_silence_warning = 1' "$HOME/.pmix/mca-params.conf" 2>/dev/null \
  || echo 'pcompress_base_silence_warning = 1' >> "$HOME/.pmix/mca-params.conf"

if [[ "$INFO" -eq 1 && "$DRY_RUN" -eq 0 ]]; then rm -rf /tmp/nccl_info && mkdir -p /tmp/nccl_info; fi

echo "PREFLIGHT|$RC|$DRV|$NCCL|$FABRIC|$CLUSTER|$IFNAME|$IFCIDR|$PSHA|$CUDA|$IMEX|$LINKS"
REMOTE

MULTI=$(( NODES_REQUESTED > 1 ? 1 : 0 ))
echo "=== Pre-flight on ${#RUN_NODES[@]} node(s)$([[ $DRY_RUN -eq 1 ]] && echo ' (dry run: check only)') ==="
FAILED=()
declare -A P_DRV P_NCCL P_CLUSTER P_IF P_CIDR P_SHA P_CUDA P_IMEX P_NVLINKS
for ip in "${RUN_NODES[@]}"; do
  OUT=$(ssh "${SSH_OPTS[@]}" "$ip" bash -s -- \
          "$EXPECT_DRIVER" "$EXPECT_NCCL" "$PACK_DIR" "$PACK_CACHE" \
          "$IMEX_CFG" "$HOSTS_RAW" "$DRY_RUN" "$INFO" "$MULTI" "$ip" \
          "$EXPECT_CUDA" "$EXPECT_NCCL_CUDA" "$EXPECT_IMEX" "$EXPECT_NVSHMEM" "$EXPECT_DOCA" \
          "$VERSIONS_WARN" <<< "$REMOTE_PREFLIGHT" 2>&1) || true
  echo "$OUT" | grep -v '^PREFLIGHT|'
  LINE=$(echo "$OUT" | grep '^PREFLIGHT|' | tail -1)
  if [[ -z "$LINE" ]]; then FAILED+=("$ip"); echo "  [$ip] [FAIL] pre-flight did not complete"; continue; fi
  IFS='|' read -r _ rc drv nccl _fabric cluster ifname ifcidr psha cuda imex links <<< "$LINE"
  [[ "$rc" == "0" ]] || FAILED+=("$ip")
  P_DRV[$ip]="$drv"; P_NCCL[$ip]="$nccl"; P_CLUSTER[$ip]="$cluster"
  P_IF[$ip]="$ifname"; P_CIDR[$ip]="$ifcidr"; P_SHA[$ip]="$psha"; P_NVLINKS[$ip]="$links"
  P_CUDA[$ip]="$cuda"; P_IMEX[$ip]="$imex"
done

# All nodes must run the identical stack
if [[ ${#RUN_NODES[@]} -gt 1 ]]; then
  for map in P_DRV P_NCCL P_CUDA P_IMEX P_SHA P_NVLINKS; do
    declare -n M="$map"
    if [[ $(printf '%s\n' "${M[@]}" | sort -u | wc -l) -gt 1 ]]; then
      echo "  [FAIL] ${map#P_} differs between nodes:"; for ip in "${RUN_NODES[@]}"; do echo "         $ip ${M[$ip]}"; done
      FAILED+=("mismatch")
    fi
    unset -n M
  done
  # No hostname check: the pack's Open MPI 4 runs fine when trays share one
  # hostname (as GB300 racks do); only the system Open MPI 5 hung on that.
  if [[ $(printf '%s\n' "${P_CLUSTER[@]}" | sort -u | wc -l) -gt 1 ]]; then
    echo "  [WARN] ClusterUUID differs between nodes -> they are NOT in the same NVLink domain;"
    echo "         NCCL cannot use MNNVL between them (needs --allow-ib and working IB/RoCE)."
  fi
fi

PREFLIGHT_RC=0
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "=== Pre-flight FAILED: ${FAILED[*]} -- not running the test ===" >&2
  [[ $DRY_RUN -eq 1 ]] || exit 1
  PREFLIGHT_RC=1   # --check still prints the mpirun lines, then exits 1
else
  echo "=== Pre-flight complete ==="
fi

# --- 3. Launch -------------------------------------------------------------
mkdir -p "$RESULTS_DIR"
HOSTFILE="${RESULTS_DIR}/hostfile_${NODES_REQUESTED}node.cfg"
for ip in "${RUN_NODES[@]}"; do echo "$ip slots=${GPUS_PER_NODE}"; done > "$HOSTFILE"

# Bootstrap network (MPI launch + NCCL's socket bootstrap; data goes over
# NVLink). Interface NAMES differ per node, so:
#   - Open MPI (BTL + OOB) gets the SUBNET(s) in CIDR form -> valid on every node
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

# The pack's own Open MPI 4.1.6 (built --without-slurm), never the tray's.
# OPAL_PREFIX makes this relocatable build find its own orted, plugins and
# help files; exported here so the rsh launcher starts the pack's orted on
# every tray, and forwarded with -x into every rank (same as GB300).
MPIRUN="$PACK_DIR/bin/mpirun"
export OPAL_PREFIX="$PACK_DIR"
export PATH="$PACK_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$PACK_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

run_one() {
  local label="$1" bin="$2"; shift 2
  # --mca schizo ompi: the standard Open MPI CLI personality (the GB300 fix
  # for "unknown option -np"). OOB + BTL both use the bootstrap subnet(s).
  local cmd=("$MPIRUN" --mca schizo ompi "${ROOT_FLAG[@]}"
    --hostfile "$HOSTFILE" -np "$RUN_GPUS" -N "$GPUS_PER_NODE" --bind-to none --oversubscribe
    --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include "$MPI_NET"
    --mca oob_tcp_if_include "$MPI_NET"
    --mca coll_hcoll_enable 0
    --mca mca_base_component_show_load_errors 0
    -x OPAL_PREFIX -x PATH -x LD_LIBRARY_PATH "${ENV_ARGS[@]}"
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
  run_one "All-Reduce (${RUN_GPUS} GPUs)" "$PACK_DIR/bin/all_reduce_perf" "${AR_ARGS[@]}"
  run_one "All-to-All (${RUN_GPUS} GPUs)" "$PACK_DIR/bin/alltoall_perf" "${A2A_ARGS[@]}"
  echo "=== Dry run: nothing executed$([[ $PREFLIGHT_RC -ne 0 ]] && echo '; pre-flight FAILED (see above)') ==="
  exit $PREFLIGHT_RC
fi

{
  echo "$(date) : ${NODES_REQUESTED} node(s) | ${RUN_GPUS} GPUs | ${RUN_NODES[*]} | driver ${P_DRV[$NODE00]:-?} | pack nccl ${P_NCCL[$NODE00]:-?} cuda ${P_CUDA[$NODE00]:-?} sha ${P_SHA[$NODE00]:-?} | launch $(hostname)"
  run_one "All-Reduce (${RUN_GPUS} GPUs)" "$PACK_DIR/bin/all_reduce_perf" "${AR_ARGS[@]}" \
    || echo "!!! All-Reduce exited with error $?"
  run_one "All-to-All (${RUN_GPUS} GPUs)" "$PACK_DIR/bin/alltoall_perf" "${A2A_ARGS[@]}" \
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

# With NCCL_MNNVL_ENABLE=1 NCCL init already fails without NVLink between trays.
# The busbw floor also catches =2 (auto) quietly falling back to sockets over the
# bootstrap NIC (~0.1 GB/s vs ~1000+ GB/s over NVLink) and a badly degraded fabric.
if [[ $NODES_REQUESTED -gt 1 && $MNNVL_ENABLE -ne 0 ]]; then
  read -r PEAK BIGGEST <<< "$(awk '$1 ~ /^[0-9]+$/ && NF >= 9 && $8 ~ /^[0-9.]+$/ {
      if ($8 > p) p = $8; if ($1 > s) s = $1 } END { print p+0, s+0 }' "$LOG")"
  if (( BIGGEST >= 64 * 1024 * 1024 )); then
    if awk -v p="$PEAK" -v m="$MIN_MNNVL_BUSBW" 'BEGIN { exit !(p < m) }'; then
      echo "!!! peak busbw ${PEAK} GB/s < ${MIN_MNNVL_BUSBW} GB/s across ${NODES_REQUESTED} trays: NVLink between trays was NOT used"
      echo "    Check IMEX/fabric/NVLink health, or re-run with --info to see the transports."
      RC=1
    else
      echo "Peak busbw ${PEAK} GB/s across ${NODES_REQUESTED} trays (>= ${MIN_MNNVL_BUSBW}: NVLink between trays in use)"
    fi
  fi
fi

if [[ $INFO -eq 1 ]]; then
  # Which transport carried the traffic (from this node's ranks)
  echo "Transports seen in /tmp/nccl_info on $(hostname):"
  cat /tmp/nccl_info/*.log 2>/dev/null \
    | grep -oE "MNNVL[^ ]*|NVLS multicast support is (available|not available)|via (P2P/[A-Za-z]+|NET/[A-Za-z]+|SHM)" \
    | sort | uniq -c | sed 's/^/  /' || true
  echo "  (MNNVL / NVLS = NVLink between nodes; NET/Socket = TCP fallback; NET/IB = network)"
  if [[ $NODES_REQUESTED -gt 1 && $MNNVL_ENABLE -ne 0 ]] && cat /tmp/nccl_info/*.log 2>/dev/null | grep -aq "via NET/"; then
    echo "!!! some NCCL traffic between trays went over the network (via NET/), not NVLink"
    RC=1
  fi
fi
echo "Log: $LOG"
if [[ $RC -eq 0 ]]; then echo "RESULT: PASS"; else echo "RESULT: FAIL"; fi
exit $RC
VR_RUNNER_EOF
build_runner() {
  printf '%s\n' "$RUNNER_SH" | CONF="$CONF" awk '$0 == "# @@DEPLOY_SETTINGS@@" { printf "%s", ENVIRON["CONF"]; n++; next } { print } END { exit n == 1 ? 0 : 1 }' \
    || { echo "ERROR: embedded runner has no @@DEPLOY_SETTINGS@@ marker" >&2; return 1; }
}
if [[ $DRY_RUN -eq 1 ]]; then
  echo "[DRY-RUN] would stage ${RUN_DIR}/run_vr_nccl_test.sh ($(build_runner | wc -l) lines) on node-00 with these settings injected:"
  echo "$CONF" | sed 's/^/    /'
else
  # GB300 style: one ready-to-run script on node-00, rack settings baked in.
  # A vr_nccl.conf from an older deploy is removed so it cannot override them.
  STAGED=$(build_runner)   # built first: a failure must not leave an empty script on node-00
  ssh_to "$NODE00" "mkdir -p '${RUN_DIR}/results' && rm -f '${RUN_DIR}/vr_nccl.conf' && cat > '${RUN_DIR}/run_vr_nccl_test.sh' && chmod +x '${RUN_DIR}/run_vr_nccl_test.sh'" <<< "$STAGED"
  echo "    OK"
fi

# --- [4/4] pre-flight on tray 1 ------------------------------------------------
RC=0
if [[ $CHECK -eq 1 && $AUTO -eq 0 ]]; then
  echo "--- [4/4] Pre-flight: run_vr_nccl_test.sh --check on node-00 ---"
  if [[ $DRY_RUN -eq 1 ]]; then echo "[DRY-RUN] ssh ${RUN_USER}@${NODE00} bash ${RUN_DIR}/run_vr_nccl_test.sh --check"
  else ssh_to "$NODE00" bash "${RUN_DIR}/run_vr_nccl_test.sh" --check || RC=$?; fi
fi
if [[ $RC -ne 0 ]]; then
  echo "=== Deployed, but the pre-flight on node-00 FAILED (see [FAIL] lines above). Fix those,"
  echo "    then re-check: ssh ${RUN_USER}@${NODE00} bash ${RUN_DIR}/run_vr_nccl_test.sh --check ==="
  exit $RC
fi

echo "=== Deploy complete: ${#NODES[@]} tray(s), node-00=${NODE00}, ${DATA_DIR} ==="
echo "    run on node-00: bash ${RUN_DIR}/run_vr_nccl_test.sh [N] [--info] [--no-nvls] ..."
echo "    or from here:   ssh ${RUN_USER}@${NODE00} bash ${RUN_DIR}/run_vr_nccl_test.sh"

if [[ $AUTO -eq 1 ]]; then
  echo "--- --auto: running the test on node-00 ---"
  if [[ $DRY_RUN -eq 1 ]]; then echo "[DRY-RUN] ssh ${RUN_USER}@${NODE00} bash ${RUN_DIR}/run_vr_nccl_test.sh ${RUN_ARGS[*]}"
  else
    # -t: Ctrl-C on the jumper reaches mpirun on node-00
    ssh -t "${SSH_OPTS[@]}" "${RUN_USER}@${NODE00}" "bash ${RUN_DIR}/run_vr_nccl_test.sh $(qargs "${RUN_ARGS[@]}")" || RC=$?
  fi
fi
exit $RC
