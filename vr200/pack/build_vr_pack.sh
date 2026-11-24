#!/bin/bash
# build_vr_pack.sh -- build nccl-test-pack-vr-arm64.tar.gz natively ON a VR200 tray
#
# The VR200 counterpart of the GB300 pack (nccl-test-pack-arm64.tar.gz): a
# small tar.gz with the nccl-tests *_perf binaries, a private Open MPI 4.1.6
# built --without-slurm (mpirun/orterun/orted/ompi_info + MCA plugins), the
# tray's stock NCCL library and the CUDA runtime. deploy_vr_nccl_test.sh
# pushes it to every tray.
#
# Run it on one VR200 tray that has the target stack installed (CUDA 13.5
# toolkit, libnccl2 + libnccl-dev 2.32.6-1+cuda13.5). It needs no root and
# changes nothing on the system:
#   - Open MPI 4.1.6 is built into its own prefix (<work>/pack), with its own
#     bundled PMIx / libevent / hwloc, so it cannot clash with the system's
#     Open MPI 5 and needs no -dev packages
#   - nccl-tests is copied out of --nccl-tests-src and built in <work>, so
#     an existing checkout/build there is not touched
#   - the GPU architecture for nccl-tests comes from this tray's GPUs
#     (nvidia-smi compute_cap), e.g. Rubin, unless NVCC_GENCODE is given
#   - a 4-GPU smoke run of the finished pack (its own mpirun) checks it
#
#   bash build_vr_pack.sh                         # work dir ~/vr-pack-build
#   bash build_vr_pack.sh --nccl-tests-src ~/nccl-tests --nccl-tests-ref v2.17.8
#   bash build_vr_pack.sh --ompi-tarball ~/openmpi-4.1.6.tar.gz   # tray without internet
#   bash build_vr_pack.sh --no-smoke              # skip the 4-GPU test run
#   bash build_vr_pack.sh --clean                 # rebuild Open MPI too
#
# Options: --work <dir> (default ~/vr-pack-build), -o <out.tar.gz> (default
# <work>/nccl-test-pack-vr-arm64.tar.gz), --nccl-tests-repo <url> (used when
# --nccl-tests-src is not a checkout; default github NVIDIA/nccl-tests).
# Environment: CUDA_HOME (default /usr/local/cuda), NVCC_GENCODE,
# EXPECT_NCCL (default 2.32.6-1+cuda13.5; "" accepts any), OMPI_VERSION (4.1.6).
#
# The pack's PACK_INFO lists what went in; run_vr_nccl_test.sh checks it.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$HOME/vr-pack-build"
OUT=""
TESTS_SRC="$HOME/nccl-tests"
TESTS_REF=""
TESTS_REPO="https://github.com/NVIDIA/nccl-tests.git"
OMPI_TARBALL=""
SMOKE=1
CLEAN=0
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC_GENCODE="${NVCC_GENCODE:-}"
EXPECT_NCCL="${EXPECT_NCCL-2.32.6-1+cuda13.5}"
OMPI_VERSION="${OMPI_VERSION:-4.1.6}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --work)            WORK="${2%/}"; shift 2 ;;
    -o)                OUT="$2"; shift 2 ;;
    --nccl-tests-src)  TESTS_SRC="${2%/}"; shift 2 ;;
    --nccl-tests-ref)  TESTS_REF="$2"; shift 2 ;;
    --nccl-tests-repo) TESTS_REPO="$2"; shift 2 ;;
    --ompi-tarball)    OMPI_TARBALL="$2"; shift 2 ;;
    --no-smoke)        SMOKE=0; shift ;;
    --clean)           CLEAN=1; shift ;;
    -h|--help)         awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
    *)                 echo "ERROR: unknown argument: $1 (see --help)" >&2; exit 1 ;;
  esac
done
OUT="${OUT:-$WORK/nccl-test-pack-vr-arm64.tar.gz}"
PACK="$WORK/pack"
ARCH_LIB="/usr/lib/$(uname -m)-linux-gnu"
step() { echo; echo "=== $* ==="; }
die()  { echo "ERROR: $*" >&2; exit 1; }

# --- 0. what this tray has -----------------------------------------------------
step "Checking the build tray"
[[ "$(uname -m)" == "aarch64" ]] || die "this is $(uname -m); build on a VR200 tray (aarch64)"
for t in gcc g++ make tar sha256sum objdump ldd; do command -v "$t" >/dev/null || die "'$t' not found (build-essential / binutils)"; done
[[ -x "$CUDA_HOME/bin/nvcc" ]] || die "no nvcc at $CUDA_HOME/bin (set CUDA_HOME)"
NVCC_V=$("$CUDA_HOME/bin/nvcc" --version | grep -oE 'V[0-9.]+' | head -1)
NCCL_V=$(dpkg-query -W -f='${Version}' libnccl2 2>/dev/null || echo none)
NCCL_DEV_V=$(dpkg-query -W -f='${Version}' libnccl-dev 2>/dev/null || echo none)
[[ -f /usr/include/nccl.h ]] || die "no /usr/include/nccl.h: install libnccl-dev $NCCL_V (same version as libnccl2)"
[[ "$NCCL_DEV_V" == "$NCCL_V" ]] || die "libnccl-dev $NCCL_DEV_V != libnccl2 $NCCL_V"
[[ -z "$EXPECT_NCCL" || "$NCCL_V" == "$EXPECT_NCCL" ]] || die "libnccl2 is $NCCL_V, expected $EXPECT_NCCL (EXPECT_NCCL=\"\" to accept it)"
NCCL_SO=$(readlink -f "$ARCH_LIB/libnccl.so.2" 2>/dev/null || true)
[[ -f "$NCCL_SO" ]] || die "no $ARCH_LIB/libnccl.so.2"
echo "  CUDA $NVCC_V ($CUDA_HOME) | libnccl2 $NCCL_V ($NCCL_SO) | $(gcc --version | head -1)"

# GPU architecture: this tray's GPUs (e.g. compute_cap 10.3 -> sm_103)
if [[ -z "$NVCC_GENCODE" ]]; then
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .' || true)
  if [[ "$CC" =~ ^[0-9]+$ ]]; then
    if "$CUDA_HOME/bin/nvcc" --list-gpu-arch 2>/dev/null | grep -qx "compute_$CC"; then
      # SASS for this GPU + PTX so a newer GPU can still JIT it
      NVCC_GENCODE="-gencode=arch=compute_$CC,code=sm_$CC -gencode=arch=compute_$CC,code=compute_$CC"
    else
      echo "  WARNING: nvcc does not list compute_$CC; using nccl-tests' default gencode" >&2
    fi
  else
    echo "  WARNING: could not read the GPUs' compute capability; using nccl-tests' default gencode" >&2
  fi
fi
echo "  nccl-tests gencode: ${NVCC_GENCODE:-nccl-tests default}"

mkdir -p "$WORK"
cd "$WORK"
JOBS=$(nproc)
# GCC 15 (Ubuntu 26.04) defaults to C23, which older C code such as Open MPI
# 4.1's bundled libraries may not compile as; pin the dialect Open MPI 4.1 was written for
OMPI_CFLAGS="-O2 -std=gnu17"

# --- 1. Open MPI 4.1.6 --without-slurm, into the pack prefix ----------------------
step "Open MPI $OMPI_VERSION -> $PACK (own prefix, bundled PMIx/libevent/hwloc)"
[[ $CLEAN -eq 1 ]] && rm -rf "$PACK" "openmpi-$OMPI_VERSION"
if [[ -x "$PACK/bin/orted" && "$("$PACK/bin/ompi_info" --version 2>/dev/null | head -1)" == *"v$OMPI_VERSION"* ]]; then
  echo "  already built ($("$PACK/bin/ompi_info" --version | head -1)) -- skipping (--clean rebuilds)"
else
  rm -rf "$PACK"
  TB="${OMPI_TARBALL:-$WORK/openmpi-$OMPI_VERSION.tar.gz}"
  if [[ ! -f "$TB" ]]; then
    echo "  downloading openmpi-$OMPI_VERSION.tar.gz"
    URL="https://download.open-mpi.org/release/open-mpi/v${OMPI_VERSION%.*}/openmpi-$OMPI_VERSION.tar.gz"
    { command -v wget >/dev/null && wget -q -O "$TB.part" "$URL"; } || curl -fsSL -o "$TB.part" "$URL" \
      || die "download failed (no internet?): copy openmpi-$OMPI_VERSION.tar.gz here and use --ompi-tarball"
    mv "$TB.part" "$TB"
  fi
  rm -rf "openmpi-$OMPI_VERSION" && tar xzf "$TB"
  UCX=--without-ucx
  [[ -f /usr/include/ucp/api/ucp.h ]] && UCX=--with-ucx   # as GB300 when libucx-dev is there
  (
    cd "openmpi-$OMPI_VERSION"
    ./configure --prefix="$PACK" --without-slurm "$UCX" --disable-mpi-fortran \
      --with-pmix=internal --with-libevent=internal --with-hwloc=internal \
      CFLAGS="$OMPI_CFLAGS" > "$WORK/ompi_configure.log" 2>&1 \
      || { tail -30 "$WORK/ompi_configure.log"; die "Open MPI configure failed (log: $WORK/ompi_configure.log)"; }
    echo "  configured ($UCX); building with $JOBS jobs (a few minutes)..."
    { make -j"$JOBS" && make install; } > "$WORK/ompi_build.log" 2>&1 \
      || { tail -30 "$WORK/ompi_build.log"; die "Open MPI build failed (log: $WORK/ompi_build.log)"; }
  )
  echo "  $("$PACK/bin/ompi_info" --version | head -1)"
fi
"$PACK/bin/ompi_info" | grep -qE "MCA [a-z]+: slurm" && die "this Open MPI has Slurm support compiled in (expected --without-slurm)"

# --- 2. nccl-tests against the pack's Open MPI + the system NCCL ---------------
step "nccl-tests (MPI=1) from $TESTS_SRC"
rm -rf "$WORK/nccl-tests"
if [[ -f "$TESTS_SRC/Makefile" ]]; then
  # copy, not build in place: the user's checkout and its build/ stay as they are
  tar -C "$TESTS_SRC" --exclude=./build -cf - . | (mkdir -p "$WORK/nccl-tests" && tar -C "$WORK/nccl-tests" -xf -)
else
  echo "  $TESTS_SRC has no nccl-tests: cloning $TESTS_REPO"
  git clone -q "$TESTS_REPO" "$WORK/nccl-tests" || die "clone failed: pass --nccl-tests-src <checkout>"
fi
if [[ -n "$TESTS_REF" ]]; then
  git -C "$WORK/nccl-tests" checkout -q "$TESTS_REF" || die "nccl-tests has no ref '$TESTS_REF'"
fi
TESTS_DESC=$(git -C "$WORK/nccl-tests" describe --tags --always --dirty 2>/dev/null || echo "unknown (not a git checkout)")
echo "  nccl-tests $TESTS_DESC"
MK=(MPI=1 "MPI_HOME=$PACK" "CUDA_HOME=$CUDA_HOME" NCCL_HOME=/usr)
[[ -n "$NVCC_GENCODE" ]] && MK+=("NVCC_GENCODE=$NVCC_GENCODE")
make -C "$WORK/nccl-tests" -j"$JOBS" "${MK[@]}" > "$WORK/nccl_tests_build.log" 2>&1 \
  || { tail -30 "$WORK/nccl_tests_build.log"; die "nccl-tests build failed (log: $WORK/nccl_tests_build.log)"; }
cp "$WORK"/nccl-tests/build/*_perf "$PACK/bin/"
echo "  $(ls "$WORK"/nccl-tests/build/*_perf | wc -l) *_perf binaries -> $PACK/bin"

# --- 3. libraries + PACK_INFO ------------------------------------------------------
step "Assembling lib/ and PACK_INFO"
rm -f "$PACK"/lib/libnccl.so* "$PACK"/lib/libcudart.so*
CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="$NVCC_GENCODE" \
  bash "$HERE/assemble_pack.sh" "$PACK" "$TESTS_DESC" "$OMPI_VERSION"

# --- 4. smoke run: the pack's own mpirun, 4 GPUs on this tray ---------------------
if [[ $SMOKE -eq 1 ]]; then
  step "Smoke run: pack mpirun, all_reduce_perf, 4 GPUs on $(hostname)"
  ROOT=(); [[ $EUID -eq 0 ]] && ROOT=(--allow-run-as-root)
  # Only the pack's libraries + the tray's base OS / driver: nothing from CUDA_HOME
  if env -u LD_LIBRARY_PATH OPAL_PREFIX="$PACK" PATH="$PACK/bin:/usr/bin:/bin" LD_LIBRARY_PATH="$PACK/lib" \
       "$PACK/bin/mpirun" --mca schizo ompi "${ROOT[@]}" -np 4 --host localhost:4 --bind-to none \
       --mca pml ob1 --mca btl self,tcp --mca mca_base_component_show_load_errors 0 \
       -x OPAL_PREFIX -x PATH -x LD_LIBRARY_PATH -x NCCL_DEBUG=WARN \
       "$PACK/bin/all_reduce_perf" -b 8 -e 256M -f 4 -g 1 2>&1 | tee "$WORK/smoke.log"; then :; fi
  grep -q "Avg bus bandwidth" "$WORK/smoke.log" && ! grep -q "Out of bounds values : [1-9]" "$WORK/smoke.log" \
    || die "smoke run failed (log: $WORK/smoke.log); the pack was NOT written"
  echo "  smoke run OK"
fi

# --- 5. tar it up: same top-level layout as the GB300 pack -------------------------
step "Writing $OUT"
tar -C "$PACK" -czf "$OUT.part" bin lib share PACK_INFO PACK_LIBS && mv "$OUT.part" "$OUT"
cat "$PACK/PACK_INFO"
echo
echo "Pack ready: $OUT ($(du -h "$OUT" | cut -f1), sha256 $(sha256sum "$OUT" | cut -c1-16)...)"
echo "Copy it next to deploy_vr_nccl_test.sh on the jumper, e.g.:"
echo "  scp $(whoami)@$(hostname -I 2>/dev/null | awk '{print $1}'):$OUT <repo>/vr200/"
