#!/bin/bash
# build_vr_pack.sh -- build nccl-test-pack-vr-arm64.tar.gz, the VR200 test pack
#
# The VR200 counterpart of the GB300 pack (nccl-test-pack-arm64.tar.gz): a
# small tar.gz with the *_perf binaries, a private Open MPI 4.1.6 built
# --without-slurm (mpirun/orterun/orted/ompi_info + MCA plugins), the stock
# NCCL release library and the CUDA runtime. deploy_vr_nccl_test.sh pushes
# it to every tray; nothing else is installed on the trays.
#
# Build once, on any machine with Docker. On x86 it needs arm64 emulation
# (docker buildx + qemu/binfmt, e.g. "docker run --privileged --rm
# tonistiigi/binfmt --install arm64"); on an aarch64 host (e.g. a VR200 tray
# or a Grace box) it builds natively and much faster.
#
#   ./build_vr_pack.sh                          # -> ../nccl-test-pack-vr-arm64.tar.gz
#   ./build_vr_pack.sh -o /tmp/pack.tar.gz
#   ./build_vr_pack.sh --nccl-debs ~/drop/      # use libnccl2/libnccl-dev .debs from the
#                                               # release drop instead of the apt repo
#
# Build arguments (environment), defaults = the VR NVL72 1.0.0 RC stack:
#   CUDA_IMAGE        nvcr.io/nvidia/cuda:13.5.0-devel-ubuntu24.04
#   NCCL_DEB_VERSION  2.32.6-1+cuda13.5   (stock libnccl2 / libnccl-dev, no patches)
#   OMPI_VERSION      4.1.6               (same Open MPI as the GB300 pack)
#   NCCL_TESTS_REF    v2.17.8             (same nccl-tests tag as the GB300 pack)
#   NCCL_TESTS_REPO   https://github.com/NVIDIA/nccl-tests.git
#   NVCC_GENCODE      (empty: nccl-tests' own default for this CUDA)
#
# The pack's PACK_INFO lists what went in; run_vr_nccl_test.sh checks it.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$(cd "$HERE/.." && pwd)/nccl-test-pack-vr-arm64.tar.gz"
DEBS=""
CUDA_IMAGE="${CUDA_IMAGE:-nvcr.io/nvidia/cuda:13.5.0-devel-ubuntu24.04}"
NCCL_DEB_VERSION="${NCCL_DEB_VERSION:-2.32.6-1+cuda13.5}"
OMPI_VERSION="${OMPI_VERSION:-4.1.6}"
NCCL_TESTS_REF="${NCCL_TESTS_REF:-v2.17.8}"
NCCL_TESTS_REPO="${NCCL_TESTS_REPO:-https://github.com/NVIDIA/nccl-tests.git}"
NVCC_GENCODE="${NVCC_GENCODE:-}"
TAG="nccl-test-pack-vr:build"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o)          OUT="$2"; shift 2 ;;
    --nccl-debs) DEBS="$2"; shift 2 ;;
    -h|--help)   sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)           echo "ERROR: unknown argument: $1 (see --help)" >&2; exit 1 ;;
  esac
done
command -v docker >/dev/null || { echo "ERROR: docker not found" >&2; exit 1; }

# Build context: the Dockerfile, assemble_pack.sh and debs/ (empty = apt repo)
CTX=$(mktemp -d)
trap 'rm -rf "$CTX"; docker rm -f vr-pack-extract >/dev/null 2>&1 || true' EXIT
cp "$HERE/nccl_vr_arm64.Dockerfile" "$HERE/assemble_pack.sh" "$CTX/"
mkdir -p "$CTX/debs"
if [[ -n "$DEBS" ]]; then
  shopt -s nullglob
  DEB_FILES=("$DEBS"/libnccl2_*.deb "$DEBS"/libnccl-dev_*.deb)
  [[ ${#DEB_FILES[@]} -eq 2 ]] || { echo "ERROR: expected one libnccl2_*.deb and one libnccl-dev_*.deb in $DEBS" >&2; exit 1; }
  cp "${DEB_FILES[@]}" "$CTX/debs/"
  NCCL_DEB_VERSION=$(dpkg-deb -f "${DEB_FILES[0]}" Version 2>/dev/null || echo "$NCCL_DEB_VERSION")
  echo "Using local NCCL debs: ${DEB_FILES[*]##*/} ($NCCL_DEB_VERSION)"
fi

BUILD=(docker build)
[[ "$(uname -m)" == "aarch64" ]] || BUILD=(docker buildx build --load)
echo "=== Building pack image ($CUDA_IMAGE, NCCL $NCCL_DEB_VERSION, Open MPI $OMPI_VERSION, nccl-tests $NCCL_TESTS_REF) ==="
"${BUILD[@]}" --platform linux/arm64 -t "$TAG" -f "$CTX/nccl_vr_arm64.Dockerfile" \
  --build-arg CUDA_IMAGE="$CUDA_IMAGE" \
  --build-arg NCCL_DEB_VERSION="$NCCL_DEB_VERSION" \
  --build-arg OMPI_VERSION="$OMPI_VERSION" \
  --build-arg NCCL_TESTS_REF="$NCCL_TESTS_REF" \
  --build-arg NCCL_TESTS_REPO="$NCCL_TESTS_REPO" \
  --build-arg NVCC_GENCODE="$NVCC_GENCODE" \
  "$CTX"

echo "=== Extracting /opt/pack -> $OUT ==="
docker rm -f vr-pack-extract >/dev/null 2>&1 || true
docker create --name vr-pack-extract "$TAG" >/dev/null
mkdir -p "$CTX/out"
docker cp vr-pack-extract:/opt/pack/. "$CTX/out/"
# Same top-level layout as the GB300 pack: bin/ lib/ share/ (+ PACK_INFO, PACK_LIBS)
tar -C "$CTX/out" -czf "$OUT" .
echo "--- PACK_INFO ---"; cat "$CTX/out/PACK_INFO"
echo "=== Pack ready: $OUT ($(du -h "$OUT" | cut -f1)), sha256 $(sha256sum "$OUT" | cut -c1-16)... ==="
