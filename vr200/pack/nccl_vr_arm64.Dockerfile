# nccl_vr_arm64.Dockerfile -- builds the VR200 NCCL test pack (see build_vr_pack.sh)
#
# Same recipe as the GB300 pack (GB300_mpirun_NCCL_diag_deployment.md,
# "How the pack is built"), retargeted to the VR200 stack:
#   - Open MPI 4.1.6 built --without-slurm (same as GB300), distro PMIx
#   - NCCL: the stock NVIDIA release package, libnccl2 2.32.6-1+cuda13.5,
#     unpatched (from the CUDA apt repo, or .deb files placed in debs/)
#   - CUDA 13.5 runtime (libcudart) from the base image
#   - stock nccl-tests built with MPI=1 against the two above
# Everything lands in /opt/pack in the pack's final layout (bin/ lib/ share/).
# Only that folder is shipped; the image itself is never used at test time.

ARG CUDA_IMAGE=nvcr.io/nvidia/cuda:13.5.0-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ARG OMPI_VERSION=4.1.6
ARG NCCL_DEB_VERSION=2.32.6-1+cuda13.5
ARG NCCL_TESTS_REPO=https://github.com/NVIDIA/nccl-tests.git
ARG NCCL_TESTS_REF=v2.17.8
ARG NVCC_GENCODE=
ARG PACK=/opt/pack

RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      git wget build-essential autoconf automake libtool pkg-config python3 \
      libhwloc-dev libevent-dev libpmix-dev libucx-dev file \
    && rm -rf /var/lib/apt/lists/*

# Open MPI 4.1.6 WITHOUT Slurm integration, installed straight into the pack
# prefix. It is relocatable: run_vr_nccl_test.sh sets OPAL_PREFIX to wherever
# the pack is extracted on the trays.
RUN wget -q https://download.open-mpi.org/release/open-mpi/v${OMPI_VERSION%.*}/openmpi-${OMPI_VERSION}.tar.gz \
    && tar -xzf openmpi-${OMPI_VERSION}.tar.gz && cd openmpi-${OMPI_VERSION} \
    && ./configure --prefix=${PACK} \
         --with-pmix=/usr/lib/$(uname -m)-linux-gnu/pmix2 \
         --without-slurm --with-ucx --disable-mpi-fortran \
    && make -j"$(nproc)" && make install \
    && cd / && rm -rf openmpi-${OMPI_VERSION}* \
    && ${PACK}/bin/ompi_info | grep -iE "Open MPI:|pmix" | head -5

# NCCL: exactly the release package, nothing rebuilt or patched. Local .debs
# in debs/ (e.g. from the release drop) win over the apt repo. The base image
# may carry another libnccl: it is replaced, and the check below makes sure
# only the pinned version is left.
COPY debs/ /tmp/debs/
RUN if ls /tmp/debs/*.deb >/dev/null 2>&1; then \
      apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-downgrades /tmp/debs/*.deb; \
    else \
      apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-downgrades \
        --allow-change-held-packages libnccl2=${NCCL_DEB_VERSION} libnccl-dev=${NCCL_DEB_VERSION}; \
    fi \
    && rm -rf /var/lib/apt/lists/* /tmp/debs \
    && test "$(dpkg-query -W -f='${Version}' libnccl2)" = "${NCCL_DEB_VERSION}" \
    && test "$(dpkg-query -W -f='${Version}' libnccl-dev)" = "${NCCL_DEB_VERSION}" \
    && test "$(find / -xdev -name 'libnccl.so.2*' -not -path '/proc/*' | xargs -n1 readlink -f | sort -u | wc -l)" = 1

# Stock nccl-tests, linked against the pack's Open MPI and the system NCCL
RUN git clone -q ${NCCL_TESTS_REPO} /var/nccl-tests && cd /var/nccl-tests \
    && git checkout -q ${NCCL_TESTS_REF} \
    && set -- MPI=1 MPI_HOME=${PACK} CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr \
    && if [ -n "${NVCC_GENCODE}" ]; then set -- "$@" "NVCC_GENCODE=${NVCC_GENCODE}"; fi \
    && make -j"$(nproc)" "$@" \
    && cp build/*_perf ${PACK}/bin/

COPY assemble_pack.sh /tmp/assemble_pack.sh
RUN bash /tmp/assemble_pack.sh ${PACK} "${NCCL_TESTS_REF}" "${OMPI_VERSION}"
