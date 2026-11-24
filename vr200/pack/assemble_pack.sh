#!/bin/bash
# assemble_pack.sh <pack dir> <nccl-tests ref> <ompi version>
# Called by build_vr_pack.sh on the build tray, after Open MPI and nccl-tests
# are in <pack dir>. Completes the pack:
#   - copies the stock libnccl.so.2 and the CUDA runtime into lib/
#   - copies every other shared library the pack's binaries and Open MPI
#     plugins need, except the base-OS floor (glibc, libstdc++, libgcc_s,
#     libz, libudev -- same floor as the GB300 pack) and the GPU driver
#     (libcuda, libnvidia-*), which must come from the tray itself
#   - writes PACK_INFO (versions; run_vr_nccl_test.sh checks it) and
#     PACK_LIBS (the ldd audit)
set -euo pipefail
PACK="$1"; TESTS_REF="$2"; OMPI_VERSION="$3"
ARCH_LIB="/usr/lib/$(uname -m)-linux-gnu"
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
copy_lib "$ARCH_LIB/libnccl.so.2"
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

NCCL_V=$(dpkg-query -W -f='${Version}' libnccl2)
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
