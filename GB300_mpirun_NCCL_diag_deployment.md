# GB300 NVL72 — Slurm-free Multi-Node NCCL Diag Deployment

**Doc version: v4** (2026-07-15)
- v4: added deploy script v0.8 `[0/4]` preflight SSH key-check, v0.9
  `--bootstrap` first-time key-seeding flag, bootstrap lifecycle
  explanation, and `run_nccl_test.sh` password-prompt/hang diagnosis.
- v3: added "Operational findings" section covering: `NCCL_MNNVL_ENABLE`
  behaviour and busbw impact, NVLS/SHARP busbw-exceeds-ceiling explanation,
  `knvlinkSendInbandData_IMPL` dmesg diagnosis and GB300 System SW fix
  reference, nvbandwidth CE test interpretation, OOB TCP clarification,
  and deploy script v0.7 root-privilege guard.
- v2: added "How the pack is built" section (`nccl_arm64.Dockerfile`),
  corrected the pack-origin description (one-off Docker build, not a
  manual unpack from the production container), added the Dockerfile to
  the files-delivered table and a rebuild note under known assumptions.
- v1: initial bare-`mpirun`/hostfile deployment writeup.

## Purpose

The original SOP (`GB300_NVL72_NCCL_SOP_v6.md`) runs multi-node NCCL tests
through Slurm + Pyxis/Enroot (`sbatch` → `srun --mpi=pmix`). That's too much
operational overhead for the diag team to install and run on every rack
they touch, and isn't appropriate for a workflow where deployment happens
on the production line and testing happens later, at the customer's site.

This deployment replaces that path entirely with a **bare `mpirun` +
hostfile** workflow, driven from a jump host, rack-by-rack, with no Slurm,
no Ansible, and (after some iteration) no container runtime at test time
either.

## Topology

- **Jumper**: e.g. `nv_ae@bu18-nv-ae-01`, working directory `~/carlonext`.
  Holds the rack inventory files and the pre-built test pack. Orchestrates
  deployment over SSH — no GPU needed here.
- **Rack inventory files**: `rackXX.sh`, one per physical rack, defining
  bash arrays of IPs by role (`CM_IPS`, `CM_BMC_IPS`, `SW_BMC_IPS`, `IPS`,
  `PMC_BMC_IPS`). The NCCL test targets **`CM_IPS`** — the 18 compute-tray
  host-OS IPs (root login). `CM_IPS` is listed high-IP-first in the file;
  the script reverses it so index 0 ("IP0") is **node-00**, the lowest IP,
  used as the mpirun launch node.
- **node-00**: the launch/anchor node. After deploy, it holds
  `run_nccl_test.sh` and has passwordless root SSH to every sibling.

## Why not a container at test time

Earlier iterations of this deployment used `enroot`/the original compiled
container image directly. That was dropped for three concrete reasons that
came up working through this with the diag team:

1. **Unpacking a multi-GB `.sqsh` per node is slow**, and doing it
   sequentially across 18 nodes made it worse.
2. **Deploy and test happen at different times and places** — deployment
   runs once on the production line; the rack then ships, and the actual
   NCCL test runs later at the customer's site, after an unknown number of
   power cycles in transit. Anything ephemeral (like a container rootfs in
   tmpfs) doesn't survive that gap.
3. **The node's own filesystem/environment must not be touched** — no
   extracting container content onto paths that are part of the OS image.
4. **The pack must be self-contained** — deployment (production line) and
   testing (customer site) happen at different times and places, and the
   field site may not have apt/package access if anything needs fixing.
   This same principle is why the pack ships its own `mpirun`/`orted`
   later on (see below), not just the NCCL test binaries.

The resolution: build the test pack **once**, outside this deployment
pipeline entirely, in a one-off Docker image whose sole job is compiling a
Slurm-free Open MPI plus pinned NCCL/nccl-tests, then copy the relevant
`bin/`/`lib/`/`share/` content out of that image into a small `.tar.gz`.
The deployment script just distributes and persists that tarball — no
container runtime is involved at test time at all.

## How the pack is built (`nccl_arm64.Dockerfile`)

The pack is **not** unpacked from the production NCCL test container —
it's built from a small, separate Dockerfile whose only purpose is
producing a Slurm-free Open MPI alongside the same pinned NCCL/nccl-tests
versions used elsewhere in this workflow:

```dockerfile
FROM nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04

# Install build dependencies
RUN apt update -y && apt install -y \
    git \
    wget \
    build-essential \
    autoconf \
    automake \
    libtool \
    libhwloc-dev \
    libevent-dev \
    libpmix-dev \
    libucx-dev \
    pkg-config \
    gfortran \
    python3

# Verify PMIx version in container (should show MAJOR=5)
RUN if [ -f /usr/lib/aarch64-linux-gnu/pmix2/include/pmix_version.h ]; then \
        cat /usr/lib/aarch64-linux-gnu/pmix2/include/pmix_version.h | grep -E "MAJOR|MINOR|RELEASE"; \
    fi

# Build OpenMPI 4.1.6 WITHOUT Slurm integration
RUN wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz && \
    tar -xzf openmpi-4.1.6.tar.gz && \
    cd openmpi-4.1.6 && \
    ./configure \
        --prefix=/usr/local \
        --with-pmix=/usr/lib/aarch64-linux-gnu/pmix2 \
        --without-slurm \
        --with-ucx \
        --disable-mpi-fortran && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# Verify PMIx in new OpenMPI
RUN /usr/local/bin/ompi_info | grep -i pmix

# Pin to exact versions per NVIDIA GB300 NVL72 benchmark specification
RUN git clone -b v2.28.7-1 https://github.com/NVIDIA/nccl.git /var/nccl && \
    git clone -b v2.17.8 https://github.com/NVIDIA/nccl-tests.git /var/nccl-tests

# Build NCCL from source
WORKDIR /var/nccl
RUN make -j$(nproc) src.build && \
    make install && \
    ldconfig

# CRITICAL: Delete pre-bundled libnccl 2.25 from the CUDA base image
RUN rm -f /usr/lib/aarch64-linux-gnu/libnccl* \
    && rm -f /usr/lib/x86_64-linux-gnu/libnccl* \
    && ldconfig

# Build nccl-tests linking against the pinned NCCL build
WORKDIR /var/nccl-tests
RUN make MPI=1 \
         MPI_HOME=/usr/local \
         CUDA_HOME=/usr/local/cuda \
         NCCL_HOME=/var/nccl
```

Points worth calling out:

- **`--without-slurm` is set at configure time**, not bolted on after —
  this is the actual origin of the pack's private Open MPI, which is why
  `--mca schizo ompi` is still needed at runtime (see below): the build
  flag controls what the binary is *capable* of, the MCA flag controls
  which CLI personality it *uses* by default.
- **Base image is the CUDA devel image** (`cuda:13.0.1-devel-ubuntu24.04`,
  arm64/GB300), not the production NCCL test container — this Dockerfile
  has no dependency on that container at all.
- **PMIx comes from the distro package** (`libpmix-dev`, MCA `pmix2`,
  expected MAJOR=5) rather than being built from source; Open MPI is
  pointed at it via `--with-pmix=/usr/lib/aarch64-linux-gnu/pmix2`.
- **UCX is enabled at configure time** (`--with-ucx`) even though the
  deployed `run_nccl_test.sh` invocation forces `--mca btl tcp,self` —
  UCX support is built in but isn't the transport actually selected for
  this rack-internal test path.
- **NCCL and nccl-tests are pinned to exact tags** (`v2.28.7-1` and
  `v2.17.8`) per the NVIDIA GB300 NVL72 benchmark spec, built from source
  against this same Open MPI — not whatever NCCL ships in the base image.
- **The base image's bundled `libnccl` is deleted before the pinned build
  installs** — without this, the dynamic linker could resolve the CUDA
  image's stock `libnccl.so` (e.g. 2.25.x) ahead of or instead of the
  pinned 2.28.7-1 build, silently testing the wrong NCCL version.
- **`ldconfig` runs after both the Open MPI and NCCL installs** so the
  build environment's linker cache reflects the new libraries
  immediately, before nccl-tests links against them.

**From this image to the deployed tarball:** the pack is assembled by
copying `/usr/local/{bin,lib,share}` (the Slurm-free Open MPI install —
`mpirun`/`orterun`/`orted`/`ompi_info` plus its MCA plugins and libs) and
the compiled `*_perf` binaries from `/var/nccl-tests/build/` into a single
directory tree, then `tar czf`'d into `nccl-test-pack-arm64.tar.gz`. No
other container content is included — this is a deliberately minimal,
self-contained extraction, not a generic "copy the whole container" step.

## What's actually in the pack

`nccl-test-pack-arm64.tar.gz` mirrors a standard Open MPI install prefix —
`bin/`, `lib/`, `share/` — plus the nccl-tests binaries dropped into that
same `bin/`:

```
bin/      all_reduce_perf, alltoall_perf, all_gather_perf, broadcast_perf,
          gather_perf, hypercube_perf, reduce_perf, reduce_scatter_perf,
          scatter_perf, sendrecv_perf      <- from /var/nccl-tests/build
          mpirun, orterun, orted, ompi_info   <- custom Open MPI 4.1.6 build
                                                  (--without-slurm, from /usr/local)
lib/      libmpi.so.40, libopen-rte.so.40, libopen-pal.so.40, libpmix.so.2,
          libnccl.so.2 (pinned v2.28.7-1), libcudart.so.13, libhwloc.so.15,
          libevent_core...
          lib/openmpi/   <- MCA plugin .so's (btl, pml, coll, schizo, etc.)
share/    man pages, openmpi help-*.txt, wrapper-data.txt
```

**Why the pack ships its own `mpirun`/`orted`, not just the NCCL test
binaries:** the system/container's own Open MPI was built `--with-slurm`.
Under that build, plain CLI invocation rejected ordinary flags outright
(`mpirun: Error: unknown option "-np"`, then `"-n"` too — neither was a
typo issue). Two changes were made in response: a second Open MPI build
configured `--without-slurm`, and `--mca schizo ompi` forced explicitly in
the invocation to select the standard (non-Slurm) CLI personality.

**Confirmed: `--mca schizo ompi` is the actual fix for the CLI rejection**
— after switching to the `--without-slurm` build, `-n`/`-np` were *still*
rejected; the error only went away once `--mca schizo ompi` was added.
So technically, this flag alone might also have resolved it on the
original system build, with no rebuild needed.

The portable rebuild is kept anyway, deliberately, for a separate reason:
**deployment and testing happen at different times and places** (the
production line vs. the customer's site, per the constraint this whole
pipeline is built around), and the field site may not have apt/package
access if anything needs fixing or redeploying once the rack has shipped.
Shipping a self-contained `mpirun`/`orted` means the test never depends on
whatever Open MPI build happens to be installed on the customer's system
— it's the same dependency-minimization reasoning that already motivated
moving off NFS and `enroot` earlier in this pipeline's design. `OPAL_PREFIX`
is pointed at the pack directory so this private install's `orted` and
libs resolve correctly on every node, independent of the system's own.

## Deploy default location: `/root/portable-nccl`

The pack is extracted to a single local folder per node, default
`/root/portable-nccl`, override-able with `--data-dir <path>`. This
started out requiring a path proven (via `findmnt`) to be on a separate
persistent volume from the OS disk — that check was removed once it was
confirmed the pack doesn't modify any system libraries or otherwise touch
anything else on the OS image, so a plain folder under `/root` was judged
acceptable for this use case.

## End-to-end flow (`deploy_rack_nccl_test.sh`)

One script, one rack file as its argument:

```bash
./deploy_rack_nccl_test.sh rack17.sh
#   uses ./nccl-test-pack-arm64.tar.gz and /root/portable-nccl by default
```

1. **Pack distribution, in parallel** — the tarball is `scp`'d directly to
   all 18 nodes at once (background jobs, not sequential), then
   `tar xzf`'d locally on each. No NFS server, no shared mount. Skips the
   copy on any node where a same-size copy is already present (cheap
   re-runs). `--only <ip1,ip2,...>` limits this step to specific node(s)
   — e.g. after a hardware swap — without re-touching the rest of the
   rack; if node-00 itself is in that list, the script falls back to a
   full pass.
2. **SSH key meshing** — node-00 generates a keypair (if it doesn't have
   one) and gets it appended to every sibling's `authorized_keys`. mpirun
   launches node→node (not jumper→node), and this same mesh is what lets
   node-00 run the pre-flight (below) against every sibling at test time.
3. **Staging** — generates and pushes `run_nccl_test.sh` onto node-00 at
   `/root/portable-nccl/run/run_nccl_test.sh`. By the time deploy
   finishes, node-00 is fully ready — no second script needed.
4. **(optional `--auto`)** — SSHes into node-00 and runs it immediately.

`NCCL_MNNVL_UUID` is auto-derived from the rack filename if not given
(`rack17.sh` → `0x17`) rather than reusing an arbitrary fixed value — it's
just a human-traceable NVLink-domain tag, not a value with intrinsic
meaning; override with `--uuid 0xNNNN` anytime.

## What runs at test time (`run_nccl_test.sh`, on node-00)

```bash
ssh root@<node-00 ip>
bash /root/portable-nccl/run/run_nccl_test.sh
```

### Pre-flight (every single invocation, against every node)

Two things get checked — and repaired if needed — every time this runs,
not just once at deploy time:

1. **`nvidia-imex` / `channel0`** — `channel0` is a kernel-driver-backed
   `/dev` node created with a manual `mknod`, not a persistent udev rule.
   It is recreated fresh on every boot, on every system, unconditionally —
   there is no way to make this survive a reboot, so it must be re-checked
   at test time regardless of when/where deploy ran:
   ```bash
   if ! systemctl is-active --quiet nvidia-imex; then
     systemctl restart nvidia-imex   # only if not already active
   fi
   if [[ ! -e /dev/nvidia-caps-imex-channels/channel0 ]]; then
     MAJOR=$(cat /proc/devices | grep nvidia-caps-imex-channels | awk '{print $1}')
     mkdir -p /dev/nvidia-caps-imex-channels
     mknod /dev/nvidia-caps-imex-channels/channel0 c "$MAJOR" 0
     chmod 0666 /dev/nvidia-caps-imex-channels/channel0
   fi
   ```
2. **Pack presence (safety net only)** — the pack *does* persist across
   reboots (it's a plain file on local disk, not tmpfs), so this is
   normally a no-op. If somehow missing/incomplete, it's re-extracted from
   the **locally cached** `.tar.gz` — no network round-trip needed.

Only after every node passes does it proceed to the actual run.

### The mpirun invocation

```bash
export OPAL_PREFIX="/root/portable-nccl"
export PATH="/root/portable-nccl/bin:$PATH"
export LD_LIBRARY_PATH="/root/portable-nccl/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

/root/portable-nccl/bin/mpirun \
  --mca schizo ompi \
  --mca pml ob1 \
  --mca btl tcp,self \
  --mca btl_tcp_if_include <auto-detected iface> \
  --mca coll_hcoll_enable 0 \
  -np 72 -N 4 \
  --hostfile /etc/nvidia-imex/nodes_config.cfg \
  --bind-to none --oversubscribe --allow-run-as-root \
  -x OPAL_PREFIX="/root/portable-nccl" \
  -x PATH="/root/portable-nccl/bin:$PATH" \
  -x LD_LIBRARY_PATH="/root/portable-nccl/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH" \
  -x NCCL_DEBUG=WARN -x NCCL_MNNVL_ENABLE=2 -x NCCL_NVLS_ENABLE=1 \
  -x NCCL_P2P_DISABLE=0 -x NCCL_IB_DISABLE=1 -x CUDA_IPC_HANDLE_SHARING_SUPPORT=1 \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 -x NCCL_MNNVL_UUID=0x17 -x NCCL_MIN_CTAS=32 \
  /root/portable-nccl/bin/all_reduce_perf -b 8 -e 32G -f 2 -g 1
```

Notable points:
- **The pack's own `mpirun` is called by full path**, not whatever
  `mpirun` resolves to on `$PATH` — that's deliberately the
  system/container's Slurm-aware build, which this avoids entirely.
- **`--mca schizo ompi`** forces the standard Open MPI CLI personality —
  confirmed as the actual fix for the "unknown option" errors (the
  `--without-slurm` rebuild alone did not resolve them). The portable
  rebuild is kept anyway so the test has no dependency on the field site's
  own Open MPI install/package availability — see rationale above.
- **`-np`/`-N` (short forms) work once `--mca schizo ompi` is set** — the
  earlier theory that this Open MPI build required long-only options was
  wrong; the real cause was the CLI personality auto-detection.
- **`--hostfile` is `/etc/nvidia-imex/nodes_config.cfg` itself** — same
  file IMEX uses, same convention as the original reference diag log.
- **`OPAL_PREFIX`/`PATH`/`LD_LIBRARY_PATH` are both exported locally on
  node-00 *and* forwarded via `-x`** — local export lets Open MPI's `rsh`
  launcher construct the correct remote `orted` bootstrap command for
  every other node; `-x` forwards the same values into each rank's own
  environment.
- Both **All-Reduce** and **All-to-All** passes run; output is logged to
  `/root/portable-nccl/run/results/<timestamp>.log` on node-00.

## Verified working result

A full run across an 18-node / 72-GPU rack completed cleanly: correct
device mapping for all 72 ranks, **zero "out of bounds" / wrong values**
across every message size tested, peak All-Reduce bus bandwidth ~927 GB/s
at 32GB, peak All-to-All bus bandwidth ~636 GB/s at 34GB.

A quick note on reading those numbers: nccl-tests reports both `algbw`
(message size ÷ time) and `busbw` (`algbw × factor`, where the factor
accounts for how much data the collective's algorithm actually moves
across the wire relative to buffer size). For All-Reduce on N ranks the
factor is `2×(N-1)/N` (≈1.97 at N=72); for All-to-All it's `(N-1)/N`
(≈0.986 at N=72) — which is why All-Reduce's `busbw` is roughly double its
`algbw`, while All-to-All's two columns are nearly identical. **`busbw` is
the number to compare against expected fabric throughput, not `algbw`.**

## Re-run / node-swap safety

Built for the diag team's actual workflow: a node fails in the field,
gets swapped, and the rack needs to be retested without redoing
everything.

- **Idempotent by default** — pack distribution and SSH key meshing are
  check-before-act; re-running on an unchanged rack does almost nothing.
- **Stale SSH host keys are cleared automatically** before every run —
  swapped hardware on a reused IP gets a new host key, which `ssh`
  otherwise refuses to reconnect to.
- **`--only <ip1,ip2,...>`** limits the pack copy+extract to just the
  swapped node(s). If node-00 itself is swapped, the script automatically
  falls back to a full rack pass (it's the source of truth for the SSH
  mesh and staging).
- If a replacement node gets a **new IP**, update `CM_IPS` in the rack's
  `rackXX.sh` file first — everything else is driven off that array.
- **No redeploy needed after a reboot, ever** — `nvidia-imex`/`channel0`
  self-heal on every test invocation by design; the pack persists on local
  disk and isn't affected by reboots at all.

## Usage

```bash
# One rack, defaults (./nccl-test-pack-arm64.tar.gz, /root/portable-nccl)
./deploy_rack_nccl_test.sh rack17.sh

# Then, on the diag side:
ssh root@<node-00 ip>
bash /root/portable-nccl/run/run_nccl_test.sh

# Explicit pack path / data dir / UUID
./deploy_rack_nccl_test.sh rack17.sh my-pack.tar.gz --data-dir /opt/portable-nccl --uuid 0x1969

# Have the jumper run the test for you instead of doing it by hand
./deploy_rack_nccl_test.sh rack17.sh --auto

# Preview every action without touching any node
./deploy_rack_nccl_test.sh rack17.sh --dry-run

# Check script version
./deploy_rack_nccl_test.sh --version

# After swapping one node, redeploy just that node
./deploy_rack_nccl_test.sh rack17.sh --only 192.168.14.191

# Whole fleet, one rack at a time
./test_all_racks.sh                 # all rack*.sh next to the scripts
./test_all_racks.sh --auto          # unattended, runs mpirun on every rack too
```

## Files delivered

| File | Purpose |
|---|---|
| `deploy_rack_nccl_test.sh` | The whole pipeline for one rack: parallel pack distribution, SSH mesh, staging, optional auto-run. Takes `rackXX.sh` as its primary argument. Current version: **v0.9**. |
| `test_all_racks.sh` | Thin loop over every `rackXX.sh` found next to it, calling `deploy_rack_nccl_test.sh` per rack and writing a pass/fail summary under `results/`. |
| `nccl_arm64.Dockerfile` | One-off build recipe for the pack's contents (Slurm-free Open MPI 4.1.6 + pinned NCCL v2.28.7-1 + nccl-tests v2.17.8). Built once, not part of the per-rack deploy pipeline; its `/usr/local` + `/var/nccl-tests/build` output is copied out and `tar czf`'d into `nccl-test-pack-arm64.tar.gz`. |

Both expect to live in the same directory as the rack inventory files and
the pack `.tar.gz` (e.g. `~/carlonext` on the jumper).

## Known assumptions / caveats

- **Pack layout is assumed fixed**: `bin/` (containing both the `*_perf`
  binaries and the custom Open MPI build) and `lib/` at the pack's top
  level. If that layout ever changes, the static paths in
  `run_nccl_test.sh`'s generation logic need updating to match (this was
  originally `find`-based/dynamic, simplified to static paths once the
  layout was confirmed fixed).
- **`nodes_config.cfg` ownership**: the script overwrites
  `/etc/nvidia-imex/nodes_config.cfg` directly on every node, every test
  run. If anything else on these nodes manages that file independently,
  this will clobber it.
- **`nvidia-imex` restarts only when needed** — the pre-flight only
  restarts the service if found inactive; a healthy rack sees no fabric
  disruption when the test is re-run.
- **apt access**: if `nfs-kernel-server`/`nfs-common` ever reappear as a
  dependency in a future revision, note this deployment currently has
  *no* such dependency at all — pack distribution is pure `scp`+`tar`.
- **Rebuilding the pack**: `nccl_arm64.Dockerfile` requires network/apt
  access (git clones, `wget`, `apt install`) and is meant to be built
  once, wherever that's convenient (does not need to be the jumper) —
  not re-run per rack or per deploy. Bumping the pinned NCCL/nccl-tests
  tags or the Open MPI version means rebuilding this image and
  re-extracting a new `nccl-test-pack-arm64.tar.gz`.

## Operational findings

Findings and clarifications captured during live rack deployments and benchmark
analysis sessions. Ordered roughly by topic.

### Script privilege requirement (v0.7)

`deploy_rack_nccl_test.sh` must be invoked as root. Every SSH/SCP call inside
the script connects to remote nodes as `root@<ip>`, and the SSH trust chain
(node-00's key meshed to every sibling's `/root/.ssh/authorized_keys`) is only
set up for root on the controller. Running as a regular user causes cascading
`Permission denied` failures across all 18 nodes.

From v0.7 the script detects this early and exits with a clear prompt:

```
ERROR: this script must be run as root.
       Please run 'sudo -s' first, then re-run this script.
```

This check uses bash's `${EUID}` built-in and runs after `--version` (so
`--version` still works without root) but before any rack files or SSH
connections are touched.

### Open MPI's "OOB" channel is plain TCP, not the BMC

Open MPI uses the term *out-of-band (OOB) channel* for the TCP path that
`mpirun`/`orted` use for process launch, wireup, and abort signals — in
contrast to the high-speed fabric (NVLink/IB) that carries actual job data.
This is the same host management NIC and IP range already listed in each
`rackXX.sh` (e.g. `192.168.14.x`). It has no relation to the physical BMC
network (iDRAC/iLO/Redfish sideband).

An external controller (running `mpirun` from outside the rack) does not need
a separate BMC IP table — the existing node-IP list is sufficient. What it
does need is:
- Network reachability from the controller into the management subnet across
  all 18 node IPs (not just node-00).
- Open MPI pinned to the correct management interface on multi-NIC GB300 nodes:
  `--mca oob_tcp_if_include <iface/subnet>` to prevent Open MPI from picking
  the wrong NIC on nodes that also have IB/RoCE interfaces.

Note that bare `mpirun` (without Slurm) is a single point of failure: if the
controller's process or network path to the rack drops, the job dies. A
`tmux`/`screen` session protects against terminal disconnection but not a
real network partition.

### nvbandwidth CE test — interpreting the 4×4 matrix

`nvbandwidth`'s `device_to_device_bidirectional_memcpy_read_ce` test produces
a 4×4 per-GPU-pair bandwidth matrix for a single node (4 GPUs per GB300
compute tray). A healthy result looks like:

```
           0         1         2         3
 0       N/A   1523.78   1527.04   1526.90
 1   1524.52       N/A   1526.90   1527.51
 2   1527.24   1527.38       N/A   1530.64
 3   1526.57   1528.46   1530.91       N/A

SUM   18327.85
COEFFICIENT_OF_VARIATION   0.00
```

Key things to check:
- **`COEFFICIENT_OF_VARIATION = 0.00`** is the headline pass/fail signal —
  all links performing identically; a non-trivial CoV would flag a degraded
  or asymmetric NVLink pair, which would then appear as a low row/column.
- **Per-pair values ~1527 GB/s**: Blackwell Ultra NVLink 5 spec is 1800 GB/s
  total aggregate bidirectional (900 GB/s each direction). ~1527 GB/s measured
  in a raw CE memcpy is ~85% of spec — normal, healthy efficiency. Protocol
  and copy-engine overhead account for the gap.
- **`SUM`** is a convenient single-number regression indicator for
  comparing across racks or across reboots without eyeballing the whole matrix.

### `NCCL_MNNVL_ENABLE` — `=2` vs `=1` and busbw impact

`NCCL_MNNVL_ENABLE` controls whether NCCL is allowed to treat GPUs in
separate physical nodes as part of the same NVLink fabric (multi-node NVLink
domain). This is the knob behind `NCCL_MNNVL_UUID` auto-derived per rack by
the deploy script.

| Value | Behaviour |
|---|---|
| `2` (default) | Auto-detect; gracefully skips MNNVL if IMEX fabric isn't fully healthy, routes over whatever working NVLink path already exists. |
| `1` | Force-enable; NCCL init fails if MNNVL is unsupported, but may also partially commit to MNNVL paths that are degraded, causing silent fallback to a slower transport. |
| `0` | Disable entirely; useful to isolate IMEX issues. |

In practice on GB300 NVL72: busbw of ~920 GB/s observed with `=2` dropped to
~70x GB/s when forcing `=1`. This was traced to `NCCL_MNNVL_ENABLE=1`
exposing intermittent NVLS multicast setup failures (see next section) that
`=2`'s auto-detect quietly avoids by falling back to a healthy NVLink path.

Note: `NCCL_MNNVL_ENABLE` (fabric scope) and `NCCL_NVLS_ENABLE` (switch
in-flight reduction) are separate knobs. MNNVL being enabled is what allows
the NVLS/SHARP path to extend *across* node boundaries on a multi-node rack;
they are complementary, not the same setting.

### Why AllReduce busbw can exceed the per-GPU NVLink ceiling

`all_reduce_perf` reports two bandwidth numbers:
- **Algorithm BW** = S ÷ t (message size ÷ measured time) — raw throughput.
- **Bus BW (busbw)** = AlgBW × 2(N−1)/N — the number to compare against
  hardware link specs.

The `2(N−1)/N` multiplier was calibrated for **Ring/Tree** algorithms, which
model every byte crossing each NVLink twice (scatter out, gather back). On
NVSwitch hardware, NCCL frequently selects **NVLS (NVLink SHARP)** instead:
the switch itself sums data from multiple GPUs in-flight, so each link carries
its chunk only once. Since `busbw` is computed from time/size via the ring
multiplier (not measured off the wire), applying a 2× model to NVLS's lighter
real traffic mathematically produces a number that can exceed the link's
physical ceiling (~900 GB/s unidirectional on Blackwell Ultra).

**920 GB/s busbw is therefore not an error** — it indicates NVLS/SHARP is
active and efficient. Documented in NVIDIA/nccl-tests issues #153, #272, #312.

To confirm: run with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH,NVLS` and check
the chosen algorithm in the init log.

### `knvlinkSendInbandData_IMPL: Failed to send inband data` — known GB300 bug

`dmesg` showing this error across all 18 nodes after a test run:

```
NVRM: knvlinkSendInbandData_IMPL: Failed to send inband data: 0
```

is the GPU driver failing to deliver inband control messages from the GPU to
the GPU Fabric Manager (GFM) over the NVLink fabric — these messages are how
each GPU coordinates NVLS multicast group setup with Fabric Manager. Dropped
messages cause NVLS multicast setup to fail, forcing NCCL to fall back to
Ring/Tree for that collective, which is why busbw drops when this occurs.

**This is a documented GB300 bug fixed in System SW Release 1.0.1:**

> *"Resolves an intermittent issue where NCCL all-reduce throughput drops.
> This occurs only when NVLink SHARP is enabled and control messages from GPU
> to GFM were dropped (for example, buffer contention), resulting in multicast
> setup failure. This version adds fixes across the GPU driver, Fabric Manager,
> and GPU firmware to improve the reliability of NVLS initialization."*
> — NVIDIA DGX GB300 NVL72 Release Notes (RN-11874-001_1.0.6), §6.2 item #20

**How to determine if you're affected:**
Compare installed component versions against the 1.0.6 stack:

| Component | 1.0.6 version |
|---|---|
| GPU Driver | 580.126.20 |
| IMEX | 580.126.20 |
| GPU Fabric Manager | 580.105.18 |
| NVOS (NVSwitch OS) | 25.02.4347 |

**Workaround (pre-upgrade):** set `NCCL_NVLS_ENABLE=0` to bypass NVLS
multicast setup entirely. You lose the NVLS speed-up but eliminate this
failure mode — more reliable for benchmark runs while awaiting the upgrade.

**Upgrade note:** the release notes require upgrading *all* components
rack-wide (compute trays + switch trays together). A partial upgrade may
cause incompatibility during NVLink Recovery operations.

**Diagnosis steps:**
1. `journalctl -u nvidia-fabricmanager` around the dmesg timestamps — look
   for multicast/partition setup failures.
2. On the NVSwitch: `nv show sdn partition` — `Health` column should read
   `healthy`.
3. Rerun with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,NVLS,ENV` and
   diff `=1` vs `=2` log for IMEX/MNNVL fallback lines and chosen algorithm.

### SSH trust bootstrap — first-time rack deployment (`--bootstrap`)

On a freshly imaged rack, no node has node-00's SSH public key in its
`/root/.ssh/authorized_keys` yet — a chicken-and-egg problem: the deploy
script meshes keys node-00 → siblings, but can't reach siblings without a
key already in place.

**Symptom** in `run_nccl_test.sh` output:
```
root@192.168.14.154's password:
root@192.168.14.152's password:
...
--- All-Reduce (72 GPUs) ---
<hangs indefinitely>
```
The pre-flight can prompt interactively for missing-key nodes; `mpirun`
cannot (BatchMode) — it silently hangs waiting for the missing ranks to
join the collective barrier.

**Resolution: run `--bootstrap` once on first deployment:**
```bash
./deploy_rack_nccl_test.sh rack01.sh --bootstrap
```

Added in v0.9. Seeds node-00's public key to every node via password SSH
using `sshpass`, then immediately continues into the normal deploy.
Requires `sshpass` (`apt-get install -y sshpass`).

**What it does per node:**
- Creates `/root/.ssh/` with `700` permissions if missing
- Appends node-00's public key to `authorized_keys` — idempotent (no
  duplicates if already present)
- Sets `authorized_keys` to `600`
- Reports `OK` or `FAILED` per IP, exits on any failure before deploying

**The root password** is the Linux root account password set during OS
imaging — typically a site-wide default shared across all 18 nodes on a
fresh rack. Nodes that don't match are reported as `FAILED` for manual
follow-up via `ssh-copy-id root@<ip>`.

**Full first-time lifecycle:**
```
Fresh rack (no SSH trust)
        |
        v
./deploy_rack_nccl_test.sh rack01.sh --bootstrap
  -> prompts once for root password
  -> seeds node-00's key to all 18 nodes
  -> continues with normal deploy [0/4] -> [1/4] -> [2/4] -> [3/4]
        |
        v
All future runs -- no --bootstrap needed
./deploy_rack_nccl_test.sh rack01.sh
  -> [0/4] preflight confirms all nodes passwordlessly reachable
  -> [2/4] mesh re-syncs key after any node swap
```

Re-run `--bootstrap` only if a node is **reimaged from scratch** (wiping
its `authorized_keys`); the `[0/4]` preflight will flag it first.

### `[0/4]` preflight SSH key-check (v0.8+)

Before touching any rack files or SSH connections, `deploy_rack_nccl_test.sh`
now runs a fast `BatchMode=yes` SSH check against every node:

```
--- [0/4] Preflight: verifying passwordless root SSH to all 18 node(s) ---
    all nodes reachable -- OK
```

If any node fails:
```
ERROR: passwordless SSH failed for 2 node(s):
  root@192.168.14.154
  root@192.168.14.152

Fix: run 'ssh-copy-id root@<ip>' for each node above, then re-run this script.
```

`BatchMode=yes` is the same non-interactive SSH mode `mpirun` uses — so
this catches exactly the nodes that would cause a silent hang during the
actual test. The check also runs after `--bootstrap` seeding to confirm
the seed worked before proceeding with the deploy.
