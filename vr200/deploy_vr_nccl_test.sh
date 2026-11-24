#!/bin/bash
# deploy_vr_nccl_test.sh -- set up VR200 compute trays for NCCL tests, from the jumper
#
# The VR200 counterpart of ../deploy_rack_nccl_test.sh. The GB300 flow pushes
# a pre-built test pack; VR200 trays instead run their own stock release stack
# (system driver, CUDA, libnccl2, Open MPI 5), so this script:
#
#   [0/5] checks passwordless SSH jumper -> every tray as --user
#         (--bootstrap [password] seeds the key first, like the GB300 script)
#   [1/5] meshes SSH tray 1 -> every tray, itself included (mpirun uses it)
#         and verifies it from tray 1
#   [2/5] checks hostnames are unique (identical hostnames hang Open MPI 5);
#         --set-hostnames <prefix> renames trays to <prefix>-nNN and writes
#         an /etc/hosts block for all trays
#   [3/5] builds stock nccl-tests (MPI=1, system NCCL) ONCE on tray 1, then
#         copies build/ to the same path on every tray. Skipped when the
#         build stamp (NCCL / CUDA / MPI / source checksum) is unchanged
#   [4/5] stages run_vr_nccl_test.sh + vr_nccl.conf on tray 1
#   [5/5] runs its pre-flight (run_vr_nccl_test.sh --check) on tray 1:
#         stack versions, IMEX, fabric health, P2P, binaries, hostnames
#   --auto then runs the test itself.
#
# Nothing on the trays is patched: no pack, no LD_LIBRARY_PATH overrides,
# no rebuilt NCCL. Everything lands under --data-dir on each tray:
#   <data-dir>/nccl-tests/          source + build/ (same path on every tray)
#   <data-dir>/run/run_vr_nccl_test.sh, vr_nccl.conf, results/   (tray 1 only)
#
# Usage (node order = tray 1 first; tray 1 is the launch node):
#   ./deploy_vr_nccl_test.sh rackNN.sh                  # CM_IPS from a GB300-style
#                                                       # rack file (listed high-IP
#                                                       # first, reversed like GB300)
#   ./deploy_vr_nccl_test.sh --hosts ip1,ip2            # or an explicit list, tray 1 first
#   ./deploy_vr_nccl_test.sh rackNN.sh --user nvidia    # SSH/run as a normal user
#                                                       # (repairs then use sudo -n)
#   ./deploy_vr_nccl_test.sh rackNN.sh --nccl-tests-src ~/nccl-tests.tar.gz
#                                                       # trays without internet: push
#                                                       # a source dir or tarball
#   ./deploy_vr_nccl_test.sh rackNN.sh --rebuild        # force a clean nccl-tests build
#   ./deploy_vr_nccl_test.sh rackNN.sh --set-hostnames vr200
#   ./deploy_vr_nccl_test.sh rackNN.sh --bootstrap [password]
#   ./deploy_vr_nccl_test.sh rackNN.sh --auto [-- <run_vr_nccl_test.sh args>]
#   ./deploy_vr_nccl_test.sh rackNN.sh --dry-run        # print what would be done
#   ./deploy_vr_nccl_test.sh --version
#
# Other options:
#   --data-dir <path>    default <user's home>/vr-nccl
#   --uuid 0xNN          NCCL_MNNVL_UUID written to vr_nccl.conf (default: unset,
#                        NCCL picks it; GB300 derived it from the rack file name)
#   --versions-warn      version mismatches in the pre-flight are warnings
#   --no-check           skip step [5/5]
#
# Environment: NCCL_TESTS_REPO (default https://github.com/NVIDIA/nccl-tests.git),
# NCCL_TESTS_REF (git ref to check out, default: the repo's default branch),
# MPI_HOME (default /usr/lib/aarch64-linux-gnu/openmpi), CUDA_HOME
# (default /usr/local/cuda), NVCC_GENCODE (passed to make when set), and the
# EXPECT_* version variables of run_vr_nccl_test.sh (written to vr_nccl.conf
# when set here).
#
# After a reboot nothing needs redeploying: run_vr_nccl_test.sh repairs
# IMEX / channel0 and clears stale processes on every run. Re-run this script
# after a tray swap, a reimage, or a stack upgrade (the build stamp then
# changes and nccl-tests is rebuilt).

set -euo pipefail

SCRIPT_VERSION="0.1-vr"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_vr_nccl_test.sh"
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8)
NCCL_TESTS_REPO="${NCCL_TESTS_REPO:-https://github.com/NVIDIA/nccl-tests.git}"
NCCL_TESTS_REF="${NCCL_TESTS_REF:-}"
MPI_HOME="${MPI_HOME:-/usr/lib/aarch64-linux-gnu/openmpi}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC_GENCODE="${NVCC_GENCODE:-}"

usage() { sed -n '2,62p' "$0" | sed 's/^# \{0,1\}//'; }

RACK_FILE=""
HOSTS_RAW=""
RUN_USER="root"
DATA_DIR=""
SRC=""
REBUILD=0
HOST_PREFIX=""
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
    --user)           RUN_USER="$2"; shift 2 ;;
    --data-dir)       DATA_DIR="${2%/}"; shift 2 ;;
    --nccl-tests-src) SRC="$2"; shift 2 ;;
    --rebuild)        REBUILD=1; shift ;;
    --set-hostnames)  HOST_PREFIX="$2"; shift 2 ;;
    --bootstrap)
      BOOTSTRAP=1; shift
      if [[ $# -gt 0 && "$1" != -* && "$1" != *.sh ]]; then BOOTSTRAP_PASS="$1"; shift; fi ;;
    --auto)           AUTO=1; shift ;;
    --dry-run)        DRY_RUN=1; shift ;;
    --no-check)       CHECK=0; shift ;;
    --versions-warn)  VERSIONS_WARN=1; shift ;;
    --uuid)           MNNVL_UUID="$2"; shift 2 ;;
    --version)        echo "deploy_vr_nccl_test.sh version ${SCRIPT_VERSION}"; exit 0 ;;
    -h|--help)        usage; exit 0 ;;
    --)               shift; RUN_ARGS=("$@"); break ;;
    -*)               echo "ERROR: unknown option: $1 (see --help)" >&2; exit 1 ;;
    *)                RACK_FILE="$1"; shift ;;
  esac
done

[[ -f "$RUNNER" ]] || { echo "ERROR: $RUNNER not found (keep this script next to it)" >&2; exit 1; }
if [[ -n "$HOST_PREFIX" && ! "$HOST_PREFIX" =~ ^[a-zA-Z][a-zA-Z0-9-]*$ ]]; then
  echo "ERROR: --set-hostnames prefix must be letters, digits and '-', starting with a letter" >&2; exit 1
fi

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
else
  echo "Usage: $0 <rack_file.sh> | --hosts ip1,ip2,...  [options]  (see --help)" >&2
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

ssh_to() { local h="$1"; shift; ssh "${SSH_OPTS[@]}" "${RUN_USER}@${h}" "$@"; }
# ssh joins its arguments into one remote command line: quote each one
# (printf '%q ' with no arguments would print '' , an empty argument)
qargs() { [[ $# -eq 0 ]] || printf '%q ' "$@"; }
# Run a local script ($1 = host, rest = its args) on a tray; dry run prints it
ssh_script() {
  local host="$1"; shift
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] ssh ${RUN_USER}@${host} bash -s -- $*  <<script (${#SCRIPT} bytes)>>"
    return 0
  fi
  ssh_to "$host" "bash -s -- $(qargs "$@")" <<< "$SCRIPT"
}

echo "=== deploy_vr_nccl_test.sh ${SCRIPT_VERSION} | ${#NODES[@]} tray(s) | tray 1=${NODE00} | user=${RUN_USER} ==="
echo "    trays: ${NODES[*]}"
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
           "umask 077; mkdir -p ~/.ssh && touch ~/.ssh/authorized_keys && grep -qxF '${PUBKEY}' ~/.ssh/authorized_keys || echo '${PUBKEY}' >> ~/.ssh/authorized_keys" 2>/dev/null; then
        echo "  [$ip] key seeded"
      else echo "  [$ip] FAILED (wrong password, unreachable, or password login disabled?)" >&2; BOOT_FAIL+=("$ip"); fi
    done
    [[ ${#BOOT_FAIL[@]} -eq 0 ]] || { echo "ERROR: bootstrap failed for: ${BOOT_FAIL[*]}" >&2; exit 1; }
  fi
fi

# --- [0/5] jumper -> trays ---------------------------------------------------
echo "--- [0/5] Passwordless SSH jumper -> ${#NODES[@]} tray(s) as ${RUN_USER} ---"
if [[ $DRY_RUN -eq 1 ]]; then
  echo "[DRY-RUN] would clear stale known_hosts entries and run: ssh ${RUN_USER}@<ip> true"
  REMOTE_HOME="$([[ $RUN_USER == root ]] && echo /root || echo "/home/$RUN_USER")"
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
  REMOTE_HOME="$(ssh_to "$NODE00" 'echo $HOME')"
  echo "    OK"
fi
DATA_DIR="${DATA_DIR:-$REMOTE_HOME/vr-nccl}"
TESTS_DIR="$DATA_DIR/nccl-tests"
RUN_DIR="$DATA_DIR/run"
echo "    data-dir on every tray: $DATA_DIR"

# --- [1/5] tray 1 -> every tray (mpirun + the runner's pre-flight use this) --
echo "--- [1/5] SSH mesh: tray 1 (${NODE00}) -> every tray, itself included ---"
if [[ $DRY_RUN -eq 1 ]]; then
  echo "[DRY-RUN] would create ~/.ssh/id_ed25519 on tray 1 if missing, add its public key on every tray, verify from tray 1"
else
  T1KEY=$(ssh_to "$NODE00" "test -f ~/.ssh/id_ed25519 || ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519 -q; cat ~/.ssh/id_ed25519.pub")
  for ip in "${NODES[@]}"; do
    ssh_to "$ip" "umask 077; mkdir -p ~/.ssh && touch ~/.ssh/authorized_keys && { grep -qxF '${T1KEY}' ~/.ssh/authorized_keys || echo '${T1KEY}' >> ~/.ssh/authorized_keys; }"
  done
  # Verify with tray 1's own client: a missing link would hang mpirun silently.
  # Old host keys of these IPs on tray 1 are cleared too (tray swap).
  if ! MESH=$(ssh_to "$NODE00" bash -s -- "${NODES[@]}" <<'EOF'
bad=""
for ip in "$@"; do
  ssh-keygen -R "$ip" >/dev/null 2>&1 || true
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8 "$ip" true 2>/dev/null || bad="$bad $ip"
done
[[ -z "$bad" ]] || { echo "$bad"; exit 1; }
EOF
  ); then
    echo "ERROR: tray 1 cannot SSH (as ${RUN_USER}) to:${MESH}" >&2
    echo "       Check sshd on those trays and /etc/ssh/ssh_config on tray 1." >&2
    exit 1
  fi
  echo "    tray 1 -> all ${#NODES[@]} tray(s): OK"
fi

# --- [2/5] hostnames -----------------------------------------------------------
echo "--- [2/5] Hostnames (Open MPI 5 hangs when trays share one) ---"
if [[ $DRY_RUN -eq 1 ]]; then
  if [[ -n "$HOST_PREFIX" ]]; then
    for i in "${!NODES[@]}"; do printf '[DRY-RUN] %s -> %s-n%02d\n' "${NODES[$i]}" "$HOST_PREFIX" "$i"; done
    echo "[DRY-RUN] would write a '# vr-nccl' block with all trays into /etc/hosts on every tray"
  else
    echo "[DRY-RUN] would check that hostnames are unique"
  fi
else
  if [[ -n "$HOST_PREFIX" ]]; then
    HOSTS_BLOCK=""
    for i in "${!NODES[@]}"; do HOSTS_BLOCK+="$(printf '%s %s-n%02d,' "${NODES[$i]}" "$HOST_PREFIX" "$i")"; done
    SCRIPT=$(cat <<'EOF'
set -euo pipefail
NEW="$1"; BLOCK="$2"   # "ip name,ip name,..."
priv() { if [[ $EUID -eq 0 ]]; then "$@"; else sudo -n "$@"; fi; }
[[ "$(hostname)" == "$NEW" ]] || priv hostnamectl set-hostname "$NEW"
# Replace our marked block in /etc/hosts (other lines are left alone)
TMP=$(mktemp)
awk '/^# vr-nccl begin/{skip=1} !skip{print} /^# vr-nccl end/{skip=0}' /etc/hosts > "$TMP"
{ echo "# vr-nccl begin (deploy_vr_nccl_test.sh)"; echo "$BLOCK" | tr ',' '\n' | sed '/^$/d'; echo "# vr-nccl end"; } >> "$TMP"
priv cp "$TMP" /etc/hosts; rm -f "$TMP"
echo "$(hostname)"
EOF
)
    for i in "${!NODES[@]}"; do
      new=$(printf '%s-n%02d' "$HOST_PREFIX" "$i")
      got=$(ssh_script "${NODES[$i]}" "$new" "$HOSTS_BLOCK") \
        || { echo "ERROR: could not set the hostname on ${NODES[$i]} (needs root or sudo -n)" >&2; exit 1; }
      echo "    ${NODES[$i]} -> $got"
    done
  fi
  declare -A HN
  for ip in "${NODES[@]}"; do HN[$ip]=$(ssh_to "$ip" hostname); done
  if [[ $(printf '%s\n' "${HN[@]}" | sort -u | wc -l) -lt ${#NODES[@]} ]]; then
    echo "ERROR: trays share a hostname: $(for ip in "${NODES[@]}"; do printf '%s=%s ' "$ip" "${HN[$ip]}"; done)" >&2
    echo "       Re-run with --set-hostnames <prefix> (e.g. vr200 -> vr200-n00, vr200-n01, ...)." >&2
    exit 1
  fi
  for ip in "${NODES[@]}"; do echo "    $ip ${HN[$ip]}"; done
fi

# --- [3/5] nccl-tests: build once on tray 1, copy to every tray -------------
echo "--- [3/5] nccl-tests: stock source, MPI=1, system NCCL -> ${TESTS_DIR} ---"
# Optional source from the jumper (trays without internet). A tarball whose
# entries share one top folder (GitHub archives) is unpacked one level down.
if [[ -n "$SRC" ]]; then
  [[ -e "$SRC" ]] || { echo "ERROR: --nccl-tests-src not found: $SRC" >&2; exit 1; }
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] would replace ${TESTS_DIR} on tray 1 with $SRC"
  else
    echo "    pushing $SRC to tray 1"
    PREP="rm -rf '${TESTS_DIR}' && mkdir -p '${TESTS_DIR}'"
    if [[ -d "$SRC" ]]; then
      tar -C "$SRC" --exclude=./build --exclude=./.git -cf - . | ssh_to "$NODE00" "$PREP && tar -C '${TESTS_DIR}' -xf -"
    else
      STRIP=0
      [[ $(tar -tzf "$SRC" | awk -F/ 'NF{print $1}' | sort -u | wc -l) -eq 1 ]] && STRIP=1
      ssh_to "$NODE00" "$PREP && tar -C '${TESTS_DIR}' -xzf - --strip-components=$STRIP" < "$SRC"
    fi
  fi
fi

SCRIPT=$(cat <<'EOF'
set -euo pipefail
DIR="$1"; REPO="$2"; REF="$3"; CUDA_HOME="$4"; MPI_HOME="$5"; REBUILD="$6"; GENCODE="$7"
if [[ ! -f "$DIR/Makefile" ]]; then
  echo "    no source at $DIR: cloning $REPO"
  rm -rf "$DIR"; mkdir -p "$(dirname "$DIR")"
  git clone -q "$REPO" "$DIR" || { echo "ERROR: git clone failed (no internet? use --nccl-tests-src)" >&2; exit 1; }
fi
if [[ -n "$REF" && -d "$DIR/.git" ]]; then git -C "$DIR" fetch -q origin "$REF" 2>/dev/null || true; git -C "$DIR" checkout -q "$REF"; fi
[[ -x "$CUDA_HOME/bin/nvcc" ]] || { echo "ERROR: no nvcc at $CUDA_HOME/bin (set CUDA_HOME)" >&2; exit 1; }
[[ -f "$MPI_HOME/include/mpi.h" ]] || { echo "ERROR: no $MPI_HOME/include/mpi.h (apt install libopenmpi-dev, or set MPI_HOME)" >&2; exit 1; }
[[ -f /usr/include/nccl.h ]] || { echo "ERROR: no /usr/include/nccl.h (apt install libnccl-dev, same version as libnccl2)" >&2; exit 1; }
NCCL_DEV=$(dpkg-query -W -f='${Version}' libnccl-dev 2>/dev/null || echo none)
NCCL_LIB=$(dpkg-query -W -f='${Version}' libnccl2 2>/dev/null || echo none)
[[ "$NCCL_DEV" == "$NCCL_LIB" ]] || { echo "ERROR: libnccl-dev $NCCL_DEV != libnccl2 $NCCL_LIB" >&2; exit 1; }
SUM=$(cd "$DIR" && find Makefile src verifiable -type f 2>/dev/null | LC_ALL=C sort | xargs md5sum | md5sum | cut -c1-12)
STAMP="nccl=$NCCL_LIB cuda=$("$CUDA_HOME/bin/nvcc" --version | grep -oE 'V[0-9.]+' | head -1) mpi=$(dpkg-query -W -f='${Version}' libopenmpi-dev 2>/dev/null || echo ?) src=$SUM gencode=${GENCODE:-default}"
if [[ "$REBUILD" -eq 0 && -x "$DIR/build/all_reduce_perf" && -x "$DIR/build/alltoall_perf" \
      && "$(cat "$DIR/build/.vr_build_stamp" 2>/dev/null)" == "$STAMP" ]]; then
  echo "    build is current ($STAMP) -- skipping"
else
  echo "    building ($STAMP)"
  LOG="$DIR/build.log"
  MK=(MPI=1 "MPI_HOME=$MPI_HOME" "CUDA_HOME=$CUDA_HOME")
  [[ -n "$GENCODE" ]] && MK+=("NVCC_GENCODE=$GENCODE")
  if ! { make -C "$DIR" clean >/dev/null 2>&1; make -C "$DIR" -j"$(nproc)" "${MK[@]}"; } > "$LOG" 2>&1; then
    tail -20 "$LOG" >&2; echo "ERROR: nccl-tests build failed (full log: $LOG)" >&2; exit 1
  fi
  echo "$STAMP" > "$DIR/build/.vr_build_stamp"
  echo "    built: $(ls "$DIR/build" | grep -c '_perf$') *_perf binaries"
fi
cat "$DIR/build/.vr_build_stamp"
EOF
)
if [[ $DRY_RUN -eq 1 ]]; then
  ssh_script "$NODE00" "$TESTS_DIR" "$NCCL_TESTS_REPO" "$NCCL_TESTS_REF" "$CUDA_HOME" "$MPI_HOME" "$REBUILD" "$NVCC_GENCODE"
  echo "[DRY-RUN] would copy ${TESTS_DIR}/build from tray 1 to: ${NODES[*]:1}"
else
  OUT=$(ssh_script "$NODE00" "$TESTS_DIR" "$NCCL_TESTS_REPO" "$NCCL_TESTS_REF" "$CUDA_HOME" "$MPI_HOME" "$REBUILD" "$NVCC_GENCODE") \
    || { echo "$OUT"; echo "ERROR: nccl-tests build on tray 1 failed" >&2; exit 1; }
  echo "$OUT" | sed '$d'
  STAMP=$(echo "$OUT" | tail -1)
  if [[ ${#NODES[@]} -gt 1 ]]; then
    # Copied from tray 1 over the mesh (in parallel); skipped where the stamp matches
    SCRIPT=$(cat <<'EOF'
set -uo pipefail
DIR="$1"; STAMP="$2"; shift 2
O=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8)
declare -A PID
for ip in "$@"; do
  (
    if [[ "$(ssh "${O[@]}" "$ip" cat "$DIR/build/.vr_build_stamp" 2>/dev/null)" == "$STAMP" ]]; then
      echo "    $ip: build already current"
    else
      tar -C "$DIR" -cf - build | ssh "${O[@]}" "$ip" "rm -rf '$DIR/build' && mkdir -p '$DIR' && tar -C '$DIR' -xf -" \
        && echo "    $ip: build copied"
    fi
  ) & PID[$ip]=$!
done
rc=0
for ip in "$@"; do wait "${PID[$ip]}" || { echo "    $ip: COPY FAILED"; rc=1; }; done
exit $rc
EOF
)
    ssh_script "$NODE00" "$TESTS_DIR" "$STAMP" "${NODES[@]:1}" \
      || { echo "ERROR: copying nccl-tests to some trays failed" >&2; exit 1; }
  fi
fi

# --- [4/5] stage the runner + its config on tray 1 ---------------------------
echo "--- [4/5] Staging run_vr_nccl_test.sh + vr_nccl.conf in ${RUN_DIR} on tray 1 ---"
CONF="# vr_nccl.conf -- written by deploy_vr_nccl_test.sh ${SCRIPT_VERSION}, $(date -u '+%Y-%m-%d %H:%M UTC')
# source: ${RACK_FILE:-"--hosts"}. Environment variables win over these values.
HOSTS=\"\${HOSTS:-$HOSTS_CSV}\"
NCCL_TESTS_DIR=\"\${NCCL_TESTS_DIR:-$TESTS_DIR}\"
RESULTS_DIR=\"\${RESULTS_DIR:-$RUN_DIR/results}\"
CUDA_HOME=\"\${CUDA_HOME:-$CUDA_HOME}\"
"
[[ -n "$MNNVL_UUID" ]] && CONF+="MNNVL_UUID=\"\${MNNVL_UUID:-$MNNVL_UUID}\""$'\n'
[[ $VERSIONS_WARN -eq 1 ]] && CONF+="VERSIONS_WARN=\"\${VERSIONS_WARN:-1}\""$'\n'
for v in EXPECT_DRIVER EXPECT_CUDA EXPECT_NCCL EXPECT_NCCL_CUDA EXPECT_IMEX EXPECT_NVSHMEM EXPECT_DOCA; do
  # Only versions set on the jumper are pinned; otherwise the runner's defaults apply
  [[ -n "${!v+x}" ]] && CONF+="$v=\"\${$v-${!v}}\""$'\n'
done
if [[ $DRY_RUN -eq 1 ]]; then
  echo "[DRY-RUN] would copy $RUNNER and write ${RUN_DIR}/vr_nccl.conf:"
  echo "$CONF" | sed 's/^/    /'
else
  ssh_to "$NODE00" "mkdir -p '${RUN_DIR}/results' && cat > '${RUN_DIR}/run_vr_nccl_test.sh' && chmod +x '${RUN_DIR}/run_vr_nccl_test.sh'" < "$RUNNER"
  ssh_to "$NODE00" "cat > '${RUN_DIR}/vr_nccl.conf'" <<< "$CONF"
  echo "    OK"
fi

# --- [5/5] pre-flight on tray 1 ------------------------------------------------
RC=0
if [[ $CHECK -eq 1 && $AUTO -eq 0 ]]; then
  echo "--- [5/5] Pre-flight: run_vr_nccl_test.sh --check on tray 1 ---"
  if [[ $DRY_RUN -eq 1 ]]; then echo "[DRY-RUN] ssh ${RUN_USER}@${NODE00} bash ${RUN_DIR}/run_vr_nccl_test.sh --check"
  else ssh_to "$NODE00" bash "${RUN_DIR}/run_vr_nccl_test.sh" --check || RC=$?; fi
fi
if [[ $RC -ne 0 ]]; then
  echo "=== Deployed, but the pre-flight on tray 1 FAILED (see [FAIL] lines above). Fix those,"
  echo "    then re-check: ssh ${RUN_USER}@${NODE00} bash ${RUN_DIR}/run_vr_nccl_test.sh --check ==="
  exit $RC
fi

echo "=== Deploy complete: ${#NODES[@]} tray(s), tray 1=${NODE00}, ${RUN_DIR} ==="
echo "    run on tray 1:  bash ${RUN_DIR}/run_vr_nccl_test.sh [N] [--info] [--no-nvls] ..."
echo "    or from here:   ssh ${RUN_USER}@${NODE00} bash ${RUN_DIR}/run_vr_nccl_test.sh"

if [[ $AUTO -eq 1 ]]; then
  echo "--- --auto: running the test on tray 1 ---"
  if [[ $DRY_RUN -eq 1 ]]; then echo "[DRY-RUN] ssh ${RUN_USER}@${NODE00} bash ${RUN_DIR}/run_vr_nccl_test.sh ${RUN_ARGS[*]}"
  else
    # -t: Ctrl-C on the jumper reaches mpirun on tray 1
    ssh -t "${SSH_OPTS[@]}" "${RUN_USER}@${NODE00}" "bash ${RUN_DIR}/run_vr_nccl_test.sh $(qargs "${RUN_ARGS[@]}")" || RC=$?
  fi
fi
exit $RC
