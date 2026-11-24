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
# EXPECT_* variables set here (see run_vr_nccl_test.sh) are injected into
# the staged script and so pinned for later runs.
#
# After a reboot nothing needs redeploying: run_vr_nccl_test.sh repairs
# IMEX / channel0, clears stale processes and, if needed, re-extracts the
# pack from its local cache on every run. Re-deploy after a tray swap, a
# reimage or a new pack.

set -euo pipefail

SCRIPT_VERSION="0.2-vr"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_vr_nccl_test.sh"
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

[[ -f "$RUNNER" ]] || { echo "ERROR: $RUNNER not found (keep this script next to it)" >&2; exit 1; }

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
build_runner() {  # the runner with $CONF in place of its @@DEPLOY_SETTINGS@@ line
  local n; n=$(grep -c '^# @@DEPLOY_SETTINGS@@$' "$RUNNER" || true)
  [[ "$n" -eq 1 ]] || { echo "ERROR: $RUNNER has no @@DEPLOY_SETTINGS@@ marker" >&2; return 1; }
  CONF="$CONF" awk '$0 == "# @@DEPLOY_SETTINGS@@" { printf "%s", ENVIRON["CONF"]; next } { print }' "$RUNNER"
}
if [[ $DRY_RUN -eq 1 ]]; then
  echo "[DRY-RUN] would stage ${RUN_DIR}/run_vr_nccl_test.sh on node-00 with these settings injected:"
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
