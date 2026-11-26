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
#   [4/4] only with --auto (like GB300): runs the test on node-00, including its
#         pre-flight. Without --auto nothing is run on the trays after staging;
#         --check runs just the pre-flight (run_vr_nccl_test.sh --check) instead.
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
#   --versions-check     pin version checking ON in the staged runner: a tray or pack
#                        version other than the target fails the pre-flight
#   --versions-warn      pin version checking to WARN (default: off, like GB300)
#   --check              run only the pre-flight on node-00 after staging
#   --any-gpu            skip the check that every tray's GPU arch matches the pack
#
# EXPECT_* variables set here (see the embedded run_vr_nccl_test.sh) are injected into
# the staged script and so pinned for later runs.
#
# After a reboot nothing needs redeploying: run_vr_nccl_test.sh repairs
# IMEX / channel0, clears stale processes and, if needed, re-extracts the
# pack from its local cache on every run. Re-deploy after a tray swap, a
# reimage or a new pack.

set -euo pipefail

SCRIPT_VERSION="0.9-vr"
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
CHECK=0
ANY_GPU=0
GB300_PACK=nccl-test-pack-arm64.tar.gz   # what ../deploy_rack_nccl_test.sh leaves in the data dir
PACK_MARKER=.pack_extracted               # in the data dir: "pack <sha256>", then sha256sum of the key files
PACK_KEY_FILES="bin/mpirun bin/orted bin/all_reduce_perf bin/alltoall_perf lib/libnccl.so.2 PACK_INFO"
VERSIONS_CHECK=""
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
    --check)          CHECK=1; shift ;;
    --any-gpu)        ANY_GPU=1; shift ;;
    --no-check)       CHECK=0; shift ;;   # the default now; kept for old command lines
    --versions-warn)  VERSIONS_CHECK=warn; shift ;;
    --versions-check) VERSIONS_CHECK=fail; shift ;;
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
# Reading the pack takes a few seconds for 200+ MB, so say so first. PACK_INFO:
# try the name build_vr_pack.sh uses first (one gunzip pass, not two);
# --occurrence=1 stops reading at it, which is at once for packs from
# build_vr_pack.sh 1.1-vr+ (PACK_INFO first in the tarball).
echo "Reading $(basename "$PACK") ($(du -h "$PACK" | cut -f1)): PACK_INFO + sha256 ..."
PACK_INFO=$(tar -xzOf "$PACK" --occurrence=1 PACK_INFO 2>/dev/null \
            || tar -xzOf "$PACK" --occurrence=1 ./PACK_INFO 2>/dev/null || true)
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
echo "$PACK_INFO" | grep -E '^(nccl|cuda|openmpi|nccl_tests|gpu_rank|built|builder)=' | sed 's/^/           /'
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
  # Wrong rack type (e.g. GB300) stops here, before anything is changed: the
  # pack's binaries are built for one GPU arch only
  PACK_SM=$(echo "$PACK_INFO" | awk -F= '$1=="gencode"' | grep -oE 'sm_[0-9]+' | head -1 | tr -d 'sm_')
  if [[ -n "$PACK_SM" && $ANY_GPU -eq 0 ]]; then
    WRONG=()
    for ip in "${NODES[@]}"; do
      cc=$(ssh_to "$ip" "nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u | xargs" || true)
      [[ "${cc//./}" == "$PACK_SM" ]] || WRONG+=("$ip(${cc:-no GPU info})")
    done
    if [[ ${#WRONG[@]} -gt 0 ]]; then
      echo "ERROR: the pack is built for sm_${PACK_SM} (VR200), but these trays report compute capability: ${WRONG[*]}" >&2
      echo "       Not a VR200 rack? GB300 (10.3) uses ../deploy_rack_nccl_test.sh. Nothing was changed." >&2
      echo "       (--any-gpu skips this check)" >&2
      exit 1
    fi
    echo "    GPU arch sm_${PACK_SM} on every tray, matches the pack"
  fi
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
  # Old bin/ lib/ share/ are replaced, so libraries of an older pack cannot linger.
  # A GB300 deploy's runner and pack tarball are removed too, so nobody runs
  # the GB300 test (other settings, none of the VR checks) on this rack.
  local old
  old=$(ssh_to "$ip" "cd '$PACK_DIR' && for f in run/run_nccl_test.sh $GB300_PACK; do [ \"\$f\" != '$PACK_BASENAME' ] && [ -e \"\$f\" ] && rm -f \"\$f\" && echo \"\$f\"; done; true" || true)
  [[ -n "$old" ]] && echo "removed GB300 leftovers: $(echo "$old" | xargs)"
  # The marker records which pack was extracted and the sha256 of its key files.
  # Same pack and the files unchanged (no interrupted extract, no other deploy
  # over it): nothing to do. Otherwise delete and extract again.
  if ssh_to "$ip" "cd '$PACK_DIR' && [ \"\$(head -1 $PACK_MARKER 2>/dev/null)\" = 'pack $PACK_SHA' ] && tail -n +2 $PACK_MARKER | sha256sum -c --quiet >/dev/null 2>&1"; then
    echo "pack already extracted (same sha256, files intact) -- skipping extract"
  else
    echo "extracting the pack (old bin/ lib/ share/ deleted first)"
    ssh_to "$ip" "cd '$PACK_DIR' && rm -rf bin lib share PACK_INFO PACK_LIBS $PACK_MARKER && tar xzf '$PACK_CACHE' && test -x bin/mpirun -a -x bin/orted -a -x bin/all_reduce_perf \
      && { { echo 'pack $PACK_SHA'; sha256sum $PACK_KEY_FILES; } > $PACK_MARKER.part 2>/dev/null && mv $PACK_MARKER.part $PACK_MARKER || rm -f $PACK_MARKER.part; }"
  fi
  echo "pack ready at $PACK_DIR"
}
if [[ $DRY_RUN -eq 1 ]]; then
  for ip in "${TARGET_NODES[@]}"; do
    echo "[DRY-RUN] $ip: copy $PACK_BASENAME to $PACK_CACHE unless its sha256 matches, extract into $PACK_DIR unless already extracted and intact"
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
[[ -n "$VERSIONS_CHECK" ]] && CONF+="VERSIONS_CHECK=\"\${VERSIONS_CHECK:-$VERSIONS_CHECK}\""$'\n'
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
#   bash run_vr_nccl_test.sh                 # full rack (default: every tray, e.g. 18 trays / 72 GPUs)
#   bash run_vr_nccl_test.sh [n]             # the first n trays of the list, node-00 first
#                                            # (n trays / n*4 GPUs), e.g. 9 = half rack (36 GPUs)
#   bash run_vr_nccl_test.sh 1               # node-00 only (4 GPUs, smoke test)
#   bash run_vr_nccl_test.sh --info          # NCCL INFO -> /tmp/nccl_info/ on each tray
#   bash run_vr_nccl_test.sh --check         # pre-flight only (no repairs), print mpirun
#   bash run_vr_nccl_test.sh --max-size 8G   # shorter sweep (default 32G)
#   bash run_vr_nccl_test.sh --hosts 10.0.0.1,10.0.0.2
#   bash run_vr_nccl_test.sh --versions-check # versions other than the targets FAIL the pre-flight
#   bash run_vr_nccl_test.sh --versions-warn  # ... are WARN lines (default: off, like GB300:
#                                             # versions are printed, not judged)
#   (default on 2+ trays: NCCL_MNNVL_ENABLE=1, so NCCL init fails if NVLink between
#    trays is unavailable; 1 tray: 2 = auto. Multi-tray runs also fail if peak busbw
#    at >= 64 MB < MIN_MNNVL_BUSBW, default 100 GB/s, and with --info if any NET/
#    transport is seen)
#   bash run_vr_nccl_test.sh --mnnvl-auto    # NCCL_MNNVL_ENABLE=2 (the GB300 default): may fall
#                                            # back to sockets; the busbw floor still applies
#   bash run_vr_nccl_test.sh --gpu-rank      # the test, then rank every GPU by NCCL point-to-point
#                                            # bandwidth (nccl_pairwise_rank from the pack,
#                                            # ../gpu_rank), slowest first. Same pre-flight,
#                                            # network and NCCL settings; one log, one RESULT.
#   bash run_vr_nccl_test.sh --gpu-rank -- -b 2G -n 20   # options for nccl_pairwise_rank
#   bash run_vr_nccl_test.sh --verbose       # pre-flight: every check of every tray, not grouped
#   bash run_vr_nccl_test.sh --all-warnings  # also show the 2 known-harmless NCCL init WARNs per
#                                            # rank (ibv_alloc_pd wrapper, RMA socket needs GDRCopy)
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
#   VERSIONS_CHECK off (default, like GB300) | warn | fail: off only reports the
#   versions found, as [INFO]; warn / fail compare them with the targets below
#   and give [WARN] / [FAIL] for a mismatch. The driver user space / kernel
#   check and the cross-tray "same stack" check apply in every mode.
#   Target versions (substring match; set one to "" to skip it):
#   EXPECT_DRIVER 620.70.08 (tray)   EXPECT_IMEX 2.0.0 (tray)
#   EXPECT_NCCL 2.32.6 + EXPECT_NCCL_CUDA cuda13.5, EXPECT_CUDA 13.5.0 (the pack's PACK_INFO)
#   EXPECT_NVSHMEM 3.8.0, EXPECT_DOCA 3.3.3 (tray; at most [WARN], even with fail)

set -euo pipefail

SCRIPT_VERSION="0.18-vr"
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
VERSIONS_CHECK="${VERSIONS_CHECK:-$([[ "${VERSIONS_WARN:-0}" == 1 ]] && echo warn || echo off)}"
INFO=0
DRY_RUN=0
MAX_SIZE=""
MNNVL_ENABLE=""     # 1 = required (default on 2+ trays), 2 = auto (default on 1 tray), 0 = off
# Multi-tray runs with MNNVL on must reach this peak busbw (GB/s) at >= 64 MB,
# else NVLink between trays was not (fully) used (see the summary check)
MIN_MNNVL_BUSBW="${MIN_MNNVL_BUSBW:-100}"
# Kernel log lines that mean the NVLink control path or a GPU had an error
NVLINK_ERR_RE='Failed to send inband data|NVRM: Xid'  # same pattern in the remote pre-flight
IB_DISABLE=1
NVLS_ENABLE=1
MNNVL_UUID="${MNNVL_UUID:-}"
NODES_REQUESTED=""
ALL_WARNINGS=0
VERBOSE=0
GPU_RANK=0
RANK_ARGS=()

usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hosts)         HOSTS_RAW="$2"; shift 2 ;;
    --info)          INFO=1; shift ;;
    --dry-run|--check) DRY_RUN=1; shift ;;
    --versions-warn) VERSIONS_CHECK=warn; shift ;;
    --versions-check) VERSIONS_CHECK=fail; shift ;;
    --max-size)      MAX_SIZE="$2"; shift 2 ;;
    --require-mnnvl) MNNVL_ENABLE=1; shift ;;
    --mnnvl-auto)    MNNVL_ENABLE=2; shift ;;
    --all-warnings)  ALL_WARNINGS=1; shift ;;
    --verbose)       VERBOSE=1; shift ;;
    --gpu-rank)      GPU_RANK=1; shift ;;
    --)              shift; RANK_ARGS=("$@"); break ;;
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
info() { echo "  $TAG [INFO] $*"; }
# A version other than its target: [INFO] (check off), [WARN] or [FAIL]
vbad()  { case "$VWARN" in fail) bad "$*" ;; warn) warn "$*" ;; *) info "$* (version check off)" ;; esac; }
vwarn() { case "$VWARN" in fail|warn) warn "$*" ;; *) info "$* (version check off)" ;; esac; }
# Check off (the default, like GB300): versions are only reported, as [INFO],
# with nothing to compare them to
if [[ "$VWARN" != warn && "$VWARN" != fail ]]; then
  EXPECT_DRIVER=""; EXPECT_NCCL=""; EXPECT_NCCL_CUDA=""; EXPECT_CUDA=""
  EXPECT_IMEX=""; EXPECT_NVSHMEM=""; EXPECT_DOCA=""
  vok() { info "$@"; }
else
  vok() { ok "$@"; }
fi
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
  elif [[ -f "$PACK_CACHE" ]] && (cd "$PACK_DIR" && rm -rf bin lib share PACK_INFO PACK_LIBS .pack_extracted && tar xzf "$PACK_CACHE"); then
    ok "pack was incomplete -- re-extracted from $PACK_CACHE"
  else bad "pack missing in $PACK_DIR and no usable cache at $PACK_CACHE -- re-run deploy_vr_nccl_test.sh"; fi
fi
pinfo() { awk -F= -v k="$1" '$1==k {sub(/^[^=]*=/, ""); print; exit}' "$PACK_DIR/PACK_INFO" 2>/dev/null; }
NCCL=$(pinfo nccl); NCCL="${NCCL:-none}"
CUDA=$(pinfo cuda); CUDA="${CUDA:-none}"
PSHA=$(sha256sum "$PACK_CACHE" 2>/dev/null | cut -c1-12); PSHA="${PSHA:-none}"
if [[ ( -z "$EXPECT_NCCL" || "$NCCL" == *"$EXPECT_NCCL"* ) && ( -z "$EXPECT_NCCL_CUDA" || "$NCCL" == *"$EXPECT_NCCL_CUDA"* ) ]]; then
  vok "pack NCCL $NCCL"
else vbad "pack NCCL is '$NCCL', expected $EXPECT_NCCL (+$EXPECT_NCCL_CUDA)"; fi
if [[ -z "$EXPECT_CUDA" || "$CUDA" == "$EXPECT_CUDA"* ]]; then vok "pack CUDA runtime $CUDA, Open MPI $(pinfo openmpi), nccl-tests $(pinfo nccl_tests)"
else vbad "pack CUDA runtime is '$CUDA', expected $EXPECT_CUDA"; fi

# IMEX: needed for memory sharing between trays, so a mismatch fails multi-tray runs only
IMEX=$(pkgs '^nvidia-imex' | head -1); IMEX="${IMEX:-none}"
IMEX_V=$(nvidia-imex --version 2>/dev/null | head -3 | xargs)
if [[ -z "$EXPECT_IMEX" || "$IMEX $IMEX_V" == *"$EXPECT_IMEX"* ]]; then vok "IMEX $IMEX${IMEX_V:+ ($IMEX_V)}"
elif [[ "$MULTI" -eq 1 ]]; then vbad "IMEX is '$IMEX${IMEX_V:+ / $IMEX_V}', expected $EXPECT_IMEX"
else vwarn "IMEX is '$IMEX${IMEX_V:+ / $IMEX_V}', expected $EXPECT_IMEX (not used by a 1-tray run)"; fi

# Informational: not used by nccl-tests, reported so the stack is on record
NVSHMEM=$(pkgs '^(lib)?nvshmem' | head -1); NVSHMEM="${NVSHMEM:-none}"
if [[ -z "$EXPECT_NVSHMEM" || "$NVSHMEM" == *"$EXPECT_NVSHMEM"* ]]; then vok "NVSHMEM $NVSHMEM"
else vwarn "NVSHMEM is '$NVSHMEM', expected $EXPECT_NVSHMEM"; fi
DOCA=$(pkgs '^doca-host$' | head -1); [[ -n "$DOCA" ]] || DOCA=$(pkgs '^doca-' | head -1); DOCA="${DOCA:-none}"
if [[ -z "$EXPECT_DOCA" || "$DOCA" == *"$EXPECT_DOCA"* ]]; then vok "DOCA $DOCA"
else vwarn "DOCA is '$DOCA', expected DOCA_Host $EXPECT_DOCA"; fi

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
  # NVLink control-path errors (GB300: dropped GPU->GFM inband messages broke
  # NVLS multicast setup and cost busbw). Counted again after the run.
  NVERR=$(echo "$DM" | grep -cE "Failed to send inband data|NVRM: Xid" || true)
  [[ "$NVERR" -eq 0 ]] && ok "no NVLink inband/Xid errors in dmesg" \
    || warn "dmesg: $NVERR NVLink inband/Xid error line(s) since boot (any new ones during the run fail it)"
else
  NVERR="?"
fi


# Stale test processes from an aborted run ("CUDA-capable device(s) is/are busy")
PIDS=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
  | grep -E 'all_reduce_perf|alltoall_perf|nccl_pairwise_rank|orted|mpirun' | awk -F, '{print $1}' | xargs)
if [[ -n "$PIDS" ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then warn "stale test processes on GPU: $PIDS (would kill)"
  else
    kill -9 $PIDS 2>/dev/null || priv kill -9 $PIDS 2>/dev/null || true
    sleep 2
    if nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null \
        | grep -qE 'all_reduce_perf|alltoall_perf|nccl_pairwise_rank|orted|mpirun'; then
      bad "stale test processes survived kill -9 -- check dmesg for Xid"
    else ok "cleared stale processes: $PIDS"; fi
  fi
fi

# PMIx "unable to find a usable compression library" startup noise (cosmetic)
mkdir -p "$HOME/.pmix"
grep -qxF 'pcompress_base_silence_warning = 1' "$HOME/.pmix/mca-params.conf" 2>/dev/null \
  || echo 'pcompress_base_silence_warning = 1' >> "$HOME/.pmix/mca-params.conf"

if [[ "$INFO" -eq 1 && "$DRY_RUN" -eq 0 ]]; then rm -rf /tmp/nccl_info && mkdir -p /tmp/nccl_info; fi

echo "PREFLIGHT|$RC|$DRV|$NCCL|$FABRIC|$CLUSTER|$IFNAME|$IFCIDR|$PSHA|$CUDA|$IMEX|$LINKS|${NVERR:-?}"
REMOTE

MULTI=$(( NODES_REQUESTED > 1 ? 1 : 0 ))
echo "=== Pre-flight on ${#RUN_NODES[@]} node(s)$([[ $DRY_RUN -eq 1 ]] && echo ' (dry run: check only)') ==="
FAILED=()
# Every tray's full pre-flight goes to PF_LOG; the console groups identical lines
# (one line per check for the whole rack, naming only the trays that differ)
mkdir -p "$RESULTS_DIR"
PF_LOG="${RESULTS_DIR}/$(date +%Y%m%d_%H%M%S)_preflight.log"
: > "$PF_LOG"
declare -A P_DRV P_NCCL P_CLUSTER P_IF P_CIDR P_SHA P_CUDA P_IMEX P_NVLINKS P_NVERR
for ip in "${RUN_NODES[@]}"; do
  OUT=$(ssh "${SSH_OPTS[@]}" "$ip" bash -s -- \
          "$EXPECT_DRIVER" "$EXPECT_NCCL" "$PACK_DIR" "$PACK_CACHE" \
          "$IMEX_CFG" "$HOSTS_RAW" "$DRY_RUN" "$INFO" "$MULTI" "$ip" \
          "$EXPECT_CUDA" "$EXPECT_NCCL_CUDA" "$EXPECT_IMEX" "$EXPECT_NVSHMEM" "$EXPECT_DOCA" \
          "$VERSIONS_CHECK" <<< "$REMOTE_PREFLIGHT" 2>&1) || true
  echo "$OUT" | grep -v '^PREFLIGHT|' >> "$PF_LOG"
  [[ $VERBOSE -eq 1 ]] && echo "$OUT" | grep -v '^PREFLIGHT|'
  LINE=$(echo "$OUT" | grep '^PREFLIGHT|' | tail -1)
  if [[ -z "$LINE" ]]; then FAILED+=("$ip"); echo "  [$ip] [FAIL] pre-flight did not complete" | tee -a "$PF_LOG"; continue; fi
  IFS='|' read -r _ rc drv nccl _fabric cluster ifname ifcidr psha cuda imex links nverr <<< "$LINE"
  P_NVERR[$ip]="$nverr"
  [[ "$rc" == "0" ]] || FAILED+=("$ip")
  P_DRV[$ip]="$drv"; P_NCCL[$ip]="$nccl"; P_CLUSTER[$ip]="$cluster"
  P_IF[$ip]="$ifname"; P_CIDR[$ip]="$ifcidr"; P_SHA[$ip]="$psha"; P_NVLINKS[$ip]="$links"
  P_CUDA[$ip]="$cuda"; P_IMEX[$ip]="$imex"
done

# One line per distinct check result: FAIL and WARN first, then the rest. A line
# seen on every tray says "all N trays"; otherwise it names the trays (n00-n03 n07).
[[ $VERBOSE -eq 1 ]] || awk -v N="${#RUN_NODES[@]}" '
  function rng(list,   a, n, i, out, s, p, v) {
    n = split(list, a, " "); out = ""; s = p = -1
    for (i = 1; i <= n + 1; i++) {
      v = (i <= n) ? a[i] + 0 : -2
      if (s >= 0 && v == p + 1) { p = v; continue }
      if (s >= 0) out = out (out ? " " : "") (s == p ? sprintf("n%02d", s) : sprintf("n%02d-n%02d", s, p))
      s = p = v
    }
    return out
  }
  match($0, /^  \[n[0-9]+\/[^]]*\] \[(OK|INFO|WARN|FAIL)\] */) {
    tag = substr($0, 5, index($0, "/") - 5)          # "00" from "  [n00/ip]"
    ip[tag + 0] = substr($0, index($0, "/") + 1, index($0, "]") - index($0, "/") - 1)
    rest = substr($0, RLENGTH + 1)
    lvl = substr($0, index($0, "] [") + 3); lvl = substr(lvl, 1, index(lvl, "]") - 1)
    if (rest ~ /^bootstrap interface /) rest = "bootstrap interface found (names and IPs in the log)"
    k = lvl SUBSEP rest
    if (!(k in cnt)) { order[++nk] = k }
    cnt[k]++; who[k] = who[k] " " tag
    next
  }
  { other[++no] = $0 }
  function ips(list,   a, n, i, out) {                # problems: name each tray with its IP
    n = split(list, a, " "); out = ""
    for (i = 1; i <= n; i++) out = out (i > 1 ? " " : "") sprintf("n%02d/%s", a[i], ip[a[i] + 0])
    return out
  }
  END {
    for (pass = 1; pass <= 3; pass++)                   # FAIL, then WARN, then the rest
      for (i = 1; i <= nk; i++) {
        split(order[i], kv, SUBSEP)
        if (pass != (kv[1] == "FAIL" ? 1 : kv[1] == "WARN" ? 2 : 3)) continue
        l = substr(who[order[i]], 2)
        where = (cnt[order[i]] == N) ? (N > 2 ? "all " N " trays" : N == 2 ? "both trays" : "") \
              : (pass < 3 && cnt[order[i]] <= 6) ? ips(l) : rng(l)
        printf "  [%s]%s %s%s\n", kv[1], substr("   ", 1, 4 - length(kv[1])), kv[2], (where ? "  (" where ")" : "")
      }
    for (i = 1; i <= no; i++) print other[i]
  }' "$PF_LOG"
echo "  (every tray's full pre-flight: $PF_LOG; --verbose prints it here)"

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
for ip in "${RUN_NODES[@]}"; do echo "Bootstrap: $ip -> ${P_IF[$ip]:-?} (${P_CIDR[$ip]:-?})"; done >> "$PF_LOG"
if [[ ${#RUN_NODES[@]} -le 4 || $VERBOSE -eq 1 ]]; then grep '^Bootstrap: ' "$PF_LOG"
else echo "Bootstrap: ${#RUN_NODES[@]} trays, one interface each (names and IPs in $PF_LOG)"; fi
echo "MPI network: $MPI_NET | NCCL_SOCKET_IFNAME: $NCCL_IFS"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_KIND=""; [[ $GPU_RANK -eq 1 ]] && LOG_KIND=_gpu_rank
LOG="${RESULTS_DIR}/${STAMP}_${NODES_REQUESTED}node${LOG_KIND}.log"
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
if [[ $GPU_RANK -eq 1 ]]; then
  # nccl_pairwise_rank does not link MPI: NCCL bootstraps from rank 0's host
  # (the first hostfile entry), and the node list labels the ranking table.
  # Only the ranking run gets these: all-reduce/all-to-all stay exactly as
  # without --gpu-rank.
  RANK_ENV_ARGS=(-x "NCCL_COMM_ID=${RUN_NODES[0]}:29517" -x "GPU_RANK_NODES=$(IFS=,; echo "${RUN_NODES[*]}")")
  RANK_BIN="$PACK_DIR/bin/nccl_pairwise_rank"
  RANK_CSV="${RESULTS_DIR}/${STAMP}_${NODES_REQUESTED}node_gpu_rank_matrix.csv"
  if [[ ! -x "$RANK_BIN" ]]; then
    echo "ERROR: $RANK_BIN not in this pack -- rebuild it with pack/build_vr_pack.sh (it adds" >&2
    echo "       nccl_pairwise_rank) and redeploy" >&2
    exit 1
  fi
fi
ENV_ARGS=(); for kv in "${NCCL_ENV[@]}"; do ENV_ARGS+=(-x "$kv"); done
EXTRA_ENV_ARGS=()   # per-run extras (set only around the GPU rank run)

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
    -x OPAL_PREFIX -x PATH -x LD_LIBRARY_PATH "${ENV_ARGS[@]}" ${EXTRA_ENV_ARGS[@]+"${EXTRA_ENV_ARGS[@]}"}
    "$bin" "$@")
  echo "--- ${label} ---"
  echo "+ ${cmd[*]}"
  [[ $DRY_RUN -eq 1 ]] && return 0
  "${cmd[@]}"
}

AR_ARGS=(-b 8 -e "$MAX_SIZE" -f 2 -g 1)
A2A_ARGS=(-d uint8 -b 8 -e "$MAX_SIZE" -f 2 -g 1)

if [[ $DRY_RUN -eq 1 ]]; then
  echo "Hostfile ${HOSTFILE}:"; sed 's/^/  /' "$HOSTFILE"
  run_one "All-Reduce (${RUN_GPUS} GPUs)" "$PACK_DIR/bin/all_reduce_perf" "${AR_ARGS[@]}"
  run_one "All-to-All (${RUN_GPUS} GPUs)" "$PACK_DIR/bin/alltoall_perf" "${A2A_ARGS[@]}"
  if [[ $GPU_RANK -eq 1 ]]; then
    EXTRA_ENV_ARGS=("${RANK_ENV_ARGS[@]}")
    run_one "GPU rank (${RUN_GPUS} GPUs)" "$RANK_BIN" -o "$RANK_CSV" ${RANK_ARGS[@]+"${RANK_ARGS[@]}"}
    EXTRA_ENV_ARGS=()
  fi
  echo "=== Dry run: nothing executed$([[ $PREFLIGHT_RC -ne 0 ]] && echo '; pre-flight FAILED (see above)') ==="
  exit $PREFLIGHT_RC
fi

# NCCL 2.32 probes its internal RMA/GIN plugins at every comm init, even with
# NCCL_IB_DISABLE=1 and no GDRCopy, and each rank prints two WARN lines that do
# not affect the run (no NCCL setting skips that probe). They are hidden on the
# console only (with the empty line NCCL prints before each), counted at the end,
# and kept in the log; --all-warnings shows them.
KNOWN_WARN_RE='ibvwrap[.]cc:[0-9]+ .*NCCL WARN lib wrapper not initialized|rma_socket[.]cc:[0-9]+ .*NCCL WARN RMA/Socket : rma_socket requires GDRCopy'
hide_known_warns() {
  if [[ $ALL_WARNINGS -eq 1 ]]; then cat; return; fi
  # NCCL prints an empty line before every WARN: a hidden WARN takes one pending
  # empty line with it; other empty lines are printed as they were.
  awk -v re="$KNOWN_WARN_RE" '
    /^[[:space:]]*$/ { blank++; next }
    $0 ~ re { n++; if (blank) blank--; next }
    { while (blank) { print ""; blank-- } print; fflush() }
    END { while (blank) { print ""; blank-- }
          if (n) printf "(%d known-harmless NCCL init WARN line(s) hidden: ibv_alloc_pd wrapper / RMA socket GDRCopy; see the log or --all-warnings)\n", n }'
}

{
  echo "$(date) : ${NODES_REQUESTED} node(s) | ${RUN_GPUS} GPUs | ${RUN_NODES[*]} | driver ${P_DRV[$NODE00]:-?} | pack nccl ${P_NCCL[$NODE00]:-?} cuda ${P_CUDA[$NODE00]:-?} sha ${P_SHA[$NODE00]:-?} | launch $(hostname)"
  run_one "All-Reduce (${RUN_GPUS} GPUs)" "$PACK_DIR/bin/all_reduce_perf" "${AR_ARGS[@]}" \
    || echo "!!! All-Reduce exited with error $?"
  run_one "All-to-All (${RUN_GPUS} GPUs)" "$PACK_DIR/bin/alltoall_perf" "${A2A_ARGS[@]}" \
    || echo "!!! All-to-All exited with error $?"
  [[ $GPU_RANK -eq 1 ]] || echo "$(date) : Done."
} 2>&1 | tee "$LOG" | hide_known_warns

# --- 4. Summary ------------------------------------------------------------
# NCCL test summary first. With --gpu-rank it comes right after all-to-all,
# before the ranking runs, with its own verdict; the GPU rank summary and the
# overall RESULT (which also covers the dmesg/--info checks) come at the end.
echo ""
if [[ $GPU_RANK -eq 1 ]]; then echo "=== Summary: NCCL test (all-reduce + all-to-all) ==="
else echo "=== Summary ==="; fi
# per-test lines always; repeated CUDA failure lines (one per rank) only once
grep -aE "^--- |Out of bounds|Avg bus bandwidth|!!! |Cuda failure" "$LOG" \
  | awk '/Cuda failure/ { k=$0; sub(/^.*\(/, "", k); if (seen[k]++) next } { print }' | head -20 || true
NCCL_RC=0
grep -aq "!!! " "$LOG" && NCCL_RC=1
grep -ac "Avg bus bandwidth" "$LOG" | grep -qx 2 || NCCL_RC=1
grep -a "Out of bounds values" "$LOG" | grep -avq " 0 OK" && NCCL_RC=1

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
      NCCL_RC=1
    else
      echo "Peak busbw ${PEAK} GB/s across ${NODES_REQUESTED} trays (>= ${MIN_MNNVL_BUSBW}: NVLink between trays in use)"
    fi
  fi
fi
RC=$NCCL_RC

if [[ $GPU_RANK -eq 1 ]]; then
  echo "NCCL test: $([[ $NCCL_RC -eq 0 ]] && echo PASS || echo FAIL)"
  echo ""
  # the ranking appends to the same log
  {
    EXTRA_ENV_ARGS=("${RANK_ENV_ARGS[@]}")
    run_one "GPU rank (${RUN_GPUS} GPUs)" "$RANK_BIN" -o "$RANK_CSV" ${RANK_ARGS[@]+"${RANK_ARGS[@]}"} \
      || echo "!!! GPU rank exited with error $?"
    EXTRA_ENV_ARGS=()
    echo "$(date) : Done."
  } 2>&1 | tee -a "$LOG" | hide_known_warns

  echo ""
  echo "=== Summary: GPU rank (nccl_pairwise_rank, ${RUN_GPUS} GPUs) ==="
  # the full ranking table is printed above by rank 0; here only its key lines
  RANK_RC=0
  grep -a "^!!! GPU rank" "$LOG" && RANK_RC=1
  grep -a "^rack median score" "$LOG" | tail -1 || true
  awk '/^=== GPU ranking/ { f = 1 } f && $1 == "1" && $2 ~ /^[0-9]+$/ && NF >= 7 {
         printf "slowest GPU: rank %s %s %s, score %s GB/s (%s of median)\n", $2, $3, $4, $5, $6
         exit }' "$LOG"
  if [[ -s "$RANK_CSV" ]]; then echo "pairwise matrix (GB/s): $RANK_CSV"
  else echo "!!! GPU rank wrote no matrix ($RANK_CSV)"; RANK_RC=1; fi
  echo "GPU rank: $([[ $RANK_RC -eq 0 ]] && echo PASS || echo FAIL)"
  [[ $RANK_RC -eq 0 ]] || RC=1
  echo ""
fi

# New NVLink inband/Xid kernel errors on any tray during the run fail it, even
# when the numbers look fine: the test is about NVLink robustness.
for ip in "${RUN_NODES[@]}"; do
  [[ "${P_NVERR[$ip]:-?}" =~ ^[0-9]+$ ]] || continue
  NOW=$(ssh "${SSH_OPTS[@]}" "$ip" "{ dmesg 2>/dev/null || sudo -n dmesg 2>/dev/null; } | grep -cE '$NVLINK_ERR_RE'" 2>/dev/null) || true
  [[ "$NOW" =~ ^[0-9]+$ ]] || continue
  if (( NOW > P_NVERR[$ip] )); then
    echo "!!! $ip: $(( NOW - P_NVERR[$ip] )) new NVLink inband/Xid error line(s) in dmesg during the run"
    RC=1
  fi
done

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

# --- [4/4] only on request: --auto runs the test, --check only its pre-flight --
RC=0
if [[ $CHECK -eq 1 && $AUTO -eq 0 ]]; then
  echo "--- [4/4] --check: pre-flight only (run_vr_nccl_test.sh --check) on node-00 ---"
  if [[ $DRY_RUN -eq 1 ]]; then echo "[DRY-RUN] ssh ${RUN_USER}@${NODE00} bash ${RUN_DIR}/run_vr_nccl_test.sh --check"
  else ssh_to "$NODE00" bash "${RUN_DIR}/run_vr_nccl_test.sh" --check || RC=$?; fi
fi
if [[ $RC -ne 0 ]]; then
  echo "=== Deployed, but the pre-flight on node-00 FAILED (see [FAIL] lines above). Fix those,"
  echo "    then re-check: ssh ${RUN_USER}@${NODE00} bash ${RUN_DIR}/run_vr_nccl_test.sh --check ==="
  exit $RC
fi

echo "=== Deploy complete: ${#NODES[@]} tray(s), node-00=${NODE00}, ${DATA_DIR} ==="
if [[ $AUTO -eq 0 ]]; then
  HALF_LINE=""; HALF=$(( ${#NODES[@]} / 2 ))
  (( HALF >= 2 )) && printf -v HALF_LINE '        bash %s/run_vr_nccl_test.sh %-3s        # half rack (first %d trays / %d GPUs)\n' \
    "$RUN_DIR" "$HALF" "$HALF" $(( HALF * 4 ))
  cat <<MSG

Next: ssh ${RUN_USER}@${NODE00} then 'bash ${RUN_DIR}/run_vr_nccl_test.sh'
      (it re-checks/repairs IMEX + channel0 on every tray first, then runs)
      (or re-run this script with --auto to have it run for you)

        bash ${RUN_DIR}/run_vr_nccl_test.sh            # full rack (default: all ${#NODES[@]} trays / $(( ${#NODES[@]} * 4 )) GPUs)
${HALF_LINE}        bash ${RUN_DIR}/run_vr_nccl_test.sh [n]        # the first n trays, node-00 first (n trays / n*4 GPUs)
        bash ${RUN_DIR}/run_vr_nccl_test.sh 1          # node-00 only (4 GPUs, smoke test)
        bash ${RUN_DIR}/run_vr_nccl_test.sh --check    # pre-flight only, nothing changed
        bash ${RUN_DIR}/run_vr_nccl_test.sh --info     # NCCL INFO -> /tmp/nccl_info/
        bash ${RUN_DIR}/run_vr_nccl_test.sh --gpu-rank # the test, then every GPU ranked slowest first

MSG
fi

if [[ $AUTO -eq 1 ]]; then
  echo "--- [4/4] --auto: running run_vr_nccl_test.sh on node-00 (includes pre-flight) ---"
  if [[ $DRY_RUN -eq 1 ]]; then echo "[DRY-RUN] ssh ${RUN_USER}@${NODE00} bash ${RUN_DIR}/run_vr_nccl_test.sh ${RUN_ARGS[*]}"
  else
    # -t: Ctrl-C on the jumper reaches mpirun on node-00
    ssh -t "${SSH_OPTS[@]}" "${RUN_USER}@${NODE00}" "bash ${RUN_DIR}/run_vr_nccl_test.sh $(qargs "${RUN_ARGS[@]}")" || RC=$?
  fi
fi
exit $RC
