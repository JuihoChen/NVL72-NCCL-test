#!/bin/bash
# build_vr_pack.sh -- build nccl-test-pack-vr-arm64.tar.gz natively ON a VR200 tray
#
# The VR200 counterpart of the GB300 pack (nccl-test-pack-arm64.tar.gz): a
# small tar.gz with the nccl-tests *_perf binaries, a private Open MPI 4.1.6
# built --without-slurm (mpirun/orterun/orted/ompi_info + MCA plugins), the
# stock NCCL 2.32.6 library and the CUDA runtime. deploy_vr_nccl_test.sh
# pushes it to every tray.
#
# Run it on one VR200 tray with the CUDA 13.5 toolkit. It needs no root and
# changes nothing on the system:
#   - NCCL 2.32.6-1+cuda13.5 is fetched into <work>/nccl, whatever NCCL the
#     tray itself has (the system libnccl2 is not used or touched). First
#     match wins:
#       1. --nccl-debs <dir> with libnccl2_* + libnccl-dev_*.deb (release drop)
#       2. "apt-get download" (no root needed) if the tray's apt lists have it
#       3. the .debs from NVIDIA's CUDA repo (NCCL_REPO_URLS)
#       4. a build of the NVIDIA/nccl tag v2.32.6-1 (slow, ~20-40 min;
#          --nccl-source forces this). Stock sources, no patches.
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
#   bash build_vr_pack.sh --nccl-debs ~/drop/     # NCCL .debs you already have
#   bash build_vr_pack.sh --no-smoke              # skip the 4-GPU test run
#   bash build_vr_pack.sh --clean                 # rebuild Open MPI too
#
# Options: --work <dir> (default ~/vr-pack-build), -o <out.tar.gz> (default
# <work>/nccl-test-pack-vr-arm64.tar.gz), --nccl-tests-repo <url> (used when
# --nccl-tests-src is not a checkout; default github NVIDIA/nccl-tests).
# Environment: CUDA_HOME (default /usr/local/cuda), NVCC_GENCODE,
# NCCL_VERSION (default 2.32.6-1+cuda13.5), OMPI_VERSION (4.1.6),
# NCCL_REPO_URLS (CUDA repo folders to try for the .debs).
#
# The pack's PACK_INFO lists what went in; run_vr_nccl_test.sh checks it.

set -euo pipefail

WORK="${HOME%/}/vr-pack-build"
OUT=""
TESTS_SRC="${HOME%/}/nccl-tests"
TESTS_REF=""
TESTS_REPO="https://github.com/NVIDIA/nccl-tests.git"
OMPI_TARBALL=""
SMOKE=1
CLEAN=0
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC_GENCODE="${NVCC_GENCODE:-}"
NCCL_VERSION="${NCCL_VERSION:-2.32.6-1+cuda13.5}"
NCCL_REPO_URLS="${NCCL_REPO_URLS:-https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/sbsa https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa}"
NCCL_DEBS=""
NCCL_SOURCE=0
OMPI_VERSION="${OMPI_VERSION:-4.1.6}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --work)            WORK="${2%/}"; shift 2 ;;
    -o)                OUT="$2"; shift 2 ;;
    --nccl-tests-src)  TESTS_SRC="${2%/}"; shift 2 ;;
    --nccl-tests-ref)  TESTS_REF="$2"; shift 2 ;;
    --nccl-tests-repo) TESTS_REPO="$2"; shift 2 ;;
    --ompi-tarball)    OMPI_TARBALL="$2"; shift 2 ;;
    --nccl-debs)       NCCL_DEBS="${2%/}"; shift 2 ;;
    --nccl-source)     NCCL_SOURCE=1; shift ;;
    --no-smoke)        SMOKE=0; shift ;;
    --clean)           CLEAN=1; shift ;;
    -h|--help)         awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
    *)                 echo "ERROR: unknown argument: $1 (see --help)" >&2; exit 1 ;;
  esac
done
OUT="${OUT:-$WORK/nccl-test-pack-vr-arm64.tar.gz}"
PACK="$WORK/pack"
step() { echo; echo "=== $* ==="; }
die()  { echo "ERROR: $*" >&2; exit 1; }

# --- 0. what this tray has -----------------------------------------------------
step "Checking the build tray"
[[ "$(uname -m)" == "aarch64" ]] || die "this is $(uname -m); build on a VR200 tray (aarch64)"
for t in gcc g++ make tar sha256sum objdump ldd; do command -v "$t" >/dev/null || die "'$t' not found (build-essential / binutils)"; done
[[ -x "$CUDA_HOME/bin/nvcc" ]] || die "no nvcc at $CUDA_HOME/bin (set CUDA_HOME)"
NVCC_V=$("$CUDA_HOME/bin/nvcc" --version | grep -oE 'V[0-9.]+' | head -1)
SYS_NCCL=$(dpkg-query -W -f='${Version}' libnccl2 2>/dev/null || echo none)
echo "  CUDA $NVCC_V ($CUDA_HOME) | $(gcc --version | head -1)"
echo "  system libnccl2: $SYS_NCCL (not used: the pack gets NCCL $NCCL_VERSION)"

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

# --- 1. NCCL 2.32.6, into <work>/nccl (include/ + lib/), never the system --------
NCCL_DIR="$WORK/nccl"
step "NCCL $NCCL_VERSION -> $NCCL_DIR"
nccl_have() { [[ -f "$NCCL_DIR/include/nccl.h" && -e "$NCCL_DIR/lib/libnccl.so.2" && "$(cat "$NCCL_DIR/VERSION" 2>/dev/null)" == "$NCCL_VERSION"* ]]; }
# unpack libnccl2 + libnccl-dev .debs (no root: dpkg-deb -x) into include/ + lib/
nccl_from_debs() {  # <dir with libnccl2_*.deb and libnccl-dev_*.deb>
  local d="$1" lib dev v
  lib=$(ls "$d"/libnccl2_"${NCCL_VERSION}"_*.deb 2>/dev/null | head -1)
  dev=$(ls "$d"/libnccl-dev_"${NCCL_VERSION}"_*.deb 2>/dev/null | head -1)
  [[ -n "$lib" && -n "$dev" ]] || return 1
  v=$(dpkg-deb -f "$lib" Version)
  [[ "$v" == "$NCCL_VERSION" && "$(dpkg-deb -f "$dev" Version)" == "$NCCL_VERSION" ]] || return 1
  rm -rf "$NCCL_DIR" "$WORK/nccl-root" && mkdir -p "$NCCL_DIR" "$WORK/nccl-root"
  dpkg-deb -x "$lib" "$WORK/nccl-root" && dpkg-deb -x "$dev" "$WORK/nccl-root" || return 1
  mv "$WORK/nccl-root/usr/include" "$NCCL_DIR/include"
  mv "$WORK/nccl-root/usr/lib/$(uname -m)-linux-gnu" "$NCCL_DIR/lib"
  rm -rf "$WORK/nccl-root"
  echo "$v (release .debs: $(basename "$lib"), $(basename "$dev"))" > "$NCCL_DIR/VERSION"
}
if nccl_have && [[ -z "$NCCL_DEBS" && $NCCL_SOURCE -eq 0 ]]; then
  echo "  already there: $(cat "$NCCL_DIR/VERSION")"
else
  GOT=0
  DL="$WORK/nccl-debs"; mkdir -p "$DL"
  if [[ $NCCL_SOURCE -eq 0 && -n "$NCCL_DEBS" ]]; then
    nccl_from_debs "$NCCL_DEBS" && GOT=1 || die "no libnccl2_/libnccl-dev_${NCCL_VERSION}_*.deb in $NCCL_DEBS"
  fi
  if [[ $GOT -eq 0 && $NCCL_SOURCE -eq 0 ]]; then
    echo "  trying: apt-get download (the tray's apt lists)"
    if (cd "$DL" && apt-get download -q "libnccl2=$NCCL_VERSION" "libnccl-dev=$NCCL_VERSION" >/dev/null 2>&1); then
      nccl_from_debs "$DL" && GOT=1
    fi
  fi
  if [[ $GOT -eq 0 && $NCCL_SOURCE -eq 0 ]]; then
    for u in $NCCL_REPO_URLS; do
      echo "  trying: $u"
      ok=1
      for pkg in libnccl2 libnccl-dev; do
        f="${pkg}_${NCCL_VERSION}_arm64.deb"
        [[ -f "$DL/$f" ]] && continue
        { command -v wget >/dev/null && wget -q -O "$DL/$f.part" "$u/$f"; } || curl -fsSL -o "$DL/$f.part" "$u/$f" \
          || { rm -f "$DL/$f.part"; ok=0; break; }
        mv "$DL/$f.part" "$DL/$f"
      done
      [[ $ok -eq 1 ]] && nccl_from_debs "$DL" && { GOT=1; break; }
    done
  fi
  if [[ $GOT -eq 0 ]]; then
    # Stock source of the same release, built for this tray's GPU arch
    TAG="v${NCCL_VERSION%%+*}"            # 2.32.6-1+cuda13.5 -> v2.32.6-1
    echo "  no .debs found: building NCCL from the NVIDIA/nccl tag $TAG (slow)"
    rm -rf "$WORK/nccl-src"
    git clone -q --depth 1 -b "$TAG" https://github.com/NVIDIA/nccl.git "$WORK/nccl-src" \
      || die "could not get NCCL $NCCL_VERSION: no .debs and no tag $TAG. Copy libnccl2_/libnccl-dev_${NCCL_VERSION}_arm64.deb here and use --nccl-debs <dir>"
    NMK=(src.build "CUDA_HOME=$CUDA_HOME" "BUILDDIR=$WORK/nccl-src/build")
    [[ -n "$NVCC_GENCODE" ]] && NMK+=("NVCC_GENCODE=$NVCC_GENCODE")
    make -C "$WORK/nccl-src" -j"$JOBS" "${NMK[@]}" > "$WORK/nccl_build.log" 2>&1 \
      || { tail -30 "$WORK/nccl_build.log"; die "NCCL build failed (log: $WORK/nccl_build.log)"; }
    rm -rf "$NCCL_DIR" && mkdir -p "$NCCL_DIR"
    cp -a "$WORK/nccl-src/build/include" "$WORK/nccl-src/build/lib" "$NCCL_DIR/"
    echo "${NCCL_VERSION} (built from NVIDIA/nccl $TAG, gencode ${NVCC_GENCODE:-default})" > "$NCCL_DIR/VERSION"
  fi
fi
nccl_have || die "NCCL in $NCCL_DIR is incomplete"
# The header and the library must both be the wanted release
HV=$(awk '/define NCCL_MAJOR/{a=$3} /define NCCL_MINOR/{b=$3} /define NCCL_PATCH/{c=$3} END{print a"."b"."c}' "$NCCL_DIR/include/nccl.h")
[[ "$NCCL_VERSION" == "$HV"-* ]] || die "$NCCL_DIR/include/nccl.h is $HV, expected $NCCL_VERSION"
echo "  NCCL $(cat "$NCCL_DIR/VERSION") | nccl.h $HV | $(basename "$(readlink -f "$NCCL_DIR/lib/libnccl.so.2")")"
# GCC 15 (Ubuntu 26.04) defaults to C23, which older C code such as Open MPI
# 4.1's bundled libraries may not compile as; pin the dialect Open MPI 4.1 was written for
OMPI_CFLAGS="-O2 -std=gnu17"

# --- 2. Open MPI 4.1.6 --without-slurm, into the pack prefix ----------------------
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

# --- 3. nccl-tests against the pack's Open MPI + NCCL 2.32.6 ------------------
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
MK=(MPI=1 "MPI_HOME=$PACK" "CUDA_HOME=$CUDA_HOME" "NCCL_HOME=$NCCL_DIR")
[[ -n "$NVCC_GENCODE" ]] && MK+=("NVCC_GENCODE=$NVCC_GENCODE")
make -C "$WORK/nccl-tests" -j"$JOBS" "${MK[@]}" > "$WORK/nccl_tests_build.log" 2>&1 \
  || { tail -30 "$WORK/nccl_tests_build.log"; die "nccl-tests build failed (log: $WORK/nccl_tests_build.log)"; }
cp "$WORK"/nccl-tests/build/*_perf "$PACK/bin/"
echo "  $(ls "$WORK"/nccl-tests/build/*_perf | wc -l) *_perf binaries -> $PACK/bin"

# --- 4. libraries + PACK_INFO ------------------------------------------------------
step "Assembling lib/ and PACK_INFO"
rm -f "$PACK"/lib/libnccl.so* "$PACK"/lib/libcudart.so*
# (kept inline so this one file is all a build tray needs)
ASSEMBLE_SH=$(cat <<'ASSEMBLE'
# Args: <pack dir> <nccl-tests ref> <ompi version>. Runs after Open MPI and
# nccl-tests are in <pack dir>. Completes the pack:
#   - copies libnccl.so.2 (from $NCCL_DIR/lib, the fetched 2.32.6) and the
#     CUDA runtime into lib/
#   - copies every other shared library the pack's binaries and Open MPI
#     plugins need, except the base-OS floor (glibc, libstdc++, libgcc_s,
#     libz, libudev -- same floor as the GB300 pack) and the GPU driver
#     (libcuda, libnvidia-*), which must come from the tray itself
#   - writes PACK_INFO (versions; run_vr_nccl_test.sh checks it) and
#     PACK_LIBS (the ldd audit)
set -euo pipefail
PACK="$1"; TESTS_REF="$2"; OMPI_VERSION="$3"
cd "$PACK"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# NCCL + CUDA runtime: copy the real file, recreate the soname link
copy_lib() {  # <path of any name of the lib>
  local real so
  real=$(readlink -f "$1")
  [[ -e "lib/$(basename "$real")" ]] || cp "$real" lib/
  so=$(objdump -p "$real" 2>/dev/null | awk '/SONAME/{print $2}' || true)
  so="${so:-$(basename "$1")}"   # no SONAME: keep the name it was found by
  [[ "$so" != "$(basename "$real")" ]] && ln -sf "$(basename "$real")" "lib/$so"
  return 0
}
[[ -n "${NCCL_DIR:-}" && -e "$NCCL_DIR/lib/libnccl.so.2" ]] || { echo "ERROR: NCCL_DIR/lib/libnccl.so.2 not set or missing" >&2; exit 1; }
copy_lib "$NCCL_DIR/lib/libnccl.so.2"
copy_lib "$(ls "$CUDA_HOME"/lib64/libcudart.so.[0-9]* | sort -V | head -1)"   # libcudart.so.13

# Everything else the binaries / plugins resolve outside the pack
FLOOR='^(libc|libm|libdl|libpthread|librt|libresolv|libutil|ld-linux.*|libstdc\+\+|libgcc_s|libz|libudev|libcuda|libnvidia-.*)\.so'
is_elf() { [[ "$(head -c4 "$1" 2>/dev/null | od -An -c | tr -d ' ')" == '177ELF' ]]; }
ELFS=()
while IFS= read -r f; do is_elf "$f" && ELFS+=("$f"); done < <(find bin lib -type f \( -perm -u+x -o -name '*.so*' \))
for _ in 1 2; do   # second pass picks up deps of libs copied in the first
  for f in "${ELFS[@]}" lib/*.so*; do
    LD_LIBRARY_PATH="$PACK/lib" ldd "$f" 2>/dev/null | awk '$2=="=>" && $3 ~ /^\// {print $1, $3}'
  done | sort -u | while read -r name path; do
    [[ "$name" =~ $FLOOR ]] && continue
    [[ "$path" == "$PACK"/* ]] && continue
    copy_lib "$path"
  done
done

# Audit: nothing may be "not found" with only the pack + the floor present
MISSING=$(for f in "${ELFS[@]}"; do
  LD_LIBRARY_PATH="$PACK/lib" ldd "$f" 2>/dev/null | awk '/not found/{print $1}' \
    | while read -r name; do [[ "$name" =~ $FLOOR ]] || echo "$f: $name"; done
done)
[[ -z "$MISSING" ]] || { echo "ERROR: unresolved libraries:"; echo "$MISSING"; exit 1; }
for b in bin/all_reduce_perf bin/alltoall_perf; do
  LIB=$(LD_LIBRARY_PATH="$PACK/lib" ldd "$b" | awk '/libnccl\.so/{print $3}')
  [[ "$LIB" == "$PACK/lib/libnccl.so.2" ]] || { echo "ERROR: $b resolves NCCL to '$LIB', not the pack"; exit 1; }
done
LD_LIBRARY_PATH="$PACK/lib" ldd bin/all_reduce_perf > PACK_LIBS

NCCL_V="${NCCL_INFO:-$(dpkg-query -W -f='${Version}' libnccl2)}"
CUDA_V=$(sed -n '/"cuda"/,/}/{s/.*"version" *: *"\([^"]*\)".*/\1/p}' "$CUDA_HOME/version.json" 2>/dev/null | head -1)
[[ -n "$CUDA_V" ]] || CUDA_V=$("$CUDA_HOME/bin/nvcc" --version | grep -oE 'V[0-9.]+' | tr -d V | head -1)
cat > PACK_INFO <<INFO
pack=nccl-test-pack-vr
built=$(date -u +%Y-%m-%dT%H:%M:%SZ)
arch=$(uname -m)
nccl=${NCCL_V}
cuda=${CUDA_V}
cudart=$(basename "$(readlink -f lib/libcudart.so.[0-9]*)")
openmpi=${OMPI_VERSION} (--without-slurm)
nccl_tests=${TESTS_REF}
gencode=${NVCC_GENCODE:-default}
built_on=$(hostname) $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME") $(uname -r)
glibc=$(ldd --version | head -1 | awk '{print $NF}')
INFO
cat PACK_INFO
echo "lib/: $(ls lib | wc -l) entries, bin/: $(ls bin | wc -l) entries, $(du -sh . | cut -f1)"
ASSEMBLE
)
CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="$NVCC_GENCODE" NCCL_DIR="$NCCL_DIR" \
  NCCL_INFO="$(cat "$NCCL_DIR/VERSION")" NCCL_VERSION="$NCCL_VERSION" \
  bash -c "$ASSEMBLE_SH" assemble "$PACK" "$TESTS_DESC" "$OMPI_VERSION"

# --- 5. smoke run: the pack's own mpirun, 4 GPUs on this tray ---------------------
if [[ $SMOKE -eq 1 ]]; then
  step "Smoke run: pack mpirun, all_reduce_perf, 4 GPUs on $(hostname)"
  ROOT=(); [[ $EUID -eq 0 ]] && ROOT=(--allow-run-as-root)
  # Only the pack's libraries + the tray's base OS / driver: nothing from CUDA_HOME
  if env -u LD_LIBRARY_PATH OPAL_PREFIX="$PACK" PATH="$PACK/bin:/usr/bin:/bin" LD_LIBRARY_PATH="$PACK/lib" \
       "$PACK/bin/mpirun" --mca schizo ompi "${ROOT[@]}" -np 4 --host localhost:4 --bind-to none \
       --mca pml ob1 --mca btl self,tcp --mca mca_base_component_show_load_errors 0 \
       -x OPAL_PREFIX -x PATH -x LD_LIBRARY_PATH -x NCCL_DEBUG=VERSION -x NCCL_IB_DISABLE=1 \
       "$PACK/bin/all_reduce_perf" -b 8 -e 256M -f 4 -g 1 2>&1 | tee "$WORK/smoke.log"; then :; fi
  grep -q "Avg bus bandwidth" "$WORK/smoke.log" && ! grep -q "Out of bounds values : [1-9]" "$WORK/smoke.log" \
    || die "smoke run failed (log: $WORK/smoke.log); the pack was NOT written"
  # NCCL_DEBUG=VERSION: NCCL prints the version it actually loaded
  grep -q "NCCL version ${NCCL_VERSION%%-*}" "$WORK/smoke.log" \
    || die "smoke run did not load NCCL ${NCCL_VERSION%%-*} ($(grep -o 'NCCL version [^ ]*' "$WORK/smoke.log" | head -1)); the pack was NOT written"
  echo "  smoke run OK"
fi

# --- 6. tar it up: same top-level layout as the GB300 pack -------------------------
step "Writing $OUT"
tar -C "$PACK" -czf "$OUT.part" bin lib share PACK_INFO PACK_LIBS && mv "$OUT.part" "$OUT"
cat "$PACK/PACK_INFO"
echo
echo "Pack ready: $OUT ($(du -h "$OUT" | cut -f1), sha256 $(sha256sum "$OUT" | cut -c1-16)...)"
echo "Copy it next to deploy_vr_nccl_test.sh on the jumper, e.g.:"
echo "  scp $(whoami)@$(hostname -I 2>/dev/null | awk '{print $1}'):$OUT <repo>/vr200/"
