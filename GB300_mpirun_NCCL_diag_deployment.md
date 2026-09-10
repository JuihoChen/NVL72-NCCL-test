# GB300 NVL72 — Slurm-free Multi-Node NCCL Diag Deployment

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
pipeline entirely (the diag team unpacks the binaries/libs they need from
the container manually), ship it as a small `.tar.gz`, and have the
deployment script just distribute and persist that tarball. No container
runtime is involved at test time at all.

## What's actually in the pack

`nccl-test-pack-arm64.tar.gz` mirrors a standard Open MPI install prefix —
`bin/`, `lib/`, `share/` — plus the nccl-tests binaries dropped into that
same `bin/`:

```
bin/      all_reduce_perf, alltoall_perf, all_gather_perf, broadcast_perf,
          gather_perf, hypercube_perf, reduce_perf, reduce_scatter_perf,
          scatter_perf, sendrecv_perf
          mpirun, orterun, orted, ompi_info   <- custom Open MPI build
lib/      libmpi.so.40, libopen-rte.so.40, libopen-pal.so.40, libpmix.so.2,
          libnccl.so.2, libcudart.so.13, libhwloc.so.15, libevent_core...
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
| `deploy_rack_nccl_test.sh` | The whole pipeline for one rack: parallel pack distribution, SSH mesh, staging, optional auto-run. Takes `rackXX.sh` as its primary argument. |
| `test_all_racks.sh` | Thin loop over every `rackXX.sh` found next to it, calling `deploy_rack_nccl_test.sh` per rack and writing a pass/fail summary under `results/`. |

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
