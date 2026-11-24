# VR200: NCCL tests across compute trays

The GB300 flow, on VR200: build a portable test pack once, push it to every
tray from the jumper, run from node-00. The GB300 files at the top of the
repo are untouched; everything VR200 lives in this folder.

| | GB300 | VR200 |
|---|---|---|
| Pack build | `nccl_arm64.Dockerfile` (in `../GB300_mpirun_NCCL_diag_deployment.md`) | `pack/build_vr_pack.sh`, natively on a VR200 tray (no Docker) |
| Pack | `nccl-test-pack-arm64.tar.gz` | `nccl-test-pack-vr-arm64.tar.gz` |
| Deploy, from the jumper | `../deploy_rack_nccl_test.sh` | `deploy_vr_nccl_test.sh` |
| Test, on node-00 | generated `run_nccl_test.sh` | `run_vr_nccl_test.sh` (+ `vr_nccl.conf`) |
| Open MPI | 4.1.6 `--without-slurm`, in the pack | same |
| NCCL | 2.28.7 (built from tag), in the pack | stock 2.32.6-1+cuda13.5 release `.deb`s (or the `v2.32.6-1` tag), fetched by the build, in the pack |
| CUDA runtime | 13.0, in the pack | 13.5, in the pack |
| nccl-tests | v2.17.8, in the pack | your checkout (or `--nccl-tests-ref`), built for the tray's GPU arch |
| Default data dir | `/root/portable-nccl` | same |

Only the GPU driver and IMEX come from the trays. Nothing on the trays is
patched or installed, and the NCCL is NVIDIA's release build as is. The
NCCL settings and mpirun flags match GB300 (`--mca schizo ompi`, ob1/tcp,
MNNVL auto, NVLS on, IB off, `NCCL_P2P_SCHEDULE_GROUP_SIZE=1`).

Identical hostnames on all trays are fine again: they only hung the system
Open MPI 5 used during bring-up (finding 3), not the pack's Open MPI 4.
Pre-flight lines are labelled `nNN/IP` like GB300.

## 1. Build the pack (once, natively on one VR200 tray)

The pack is built on a tray with the CUDA 13.5 toolkit. It needs no root,
no Docker and no extra packages, and changes nothing on the system (the
tray's own libnccl2, e.g. a 2.32.4 dev drop, is neither used nor touched):

```bash
# on the build tray (e.g. node136, as nvidia)
cd ~ && git clone -b vr200-deploy https://github.com/JuihoChen/NVL72-NCCL-test.git
bash ~/NVL72-NCCL-test/vr200/pack/build_vr_pack.sh --nccl-tests-src ~/nccl-tests
#   -> ~/vr-pack-build/nccl-test-pack-vr-arm64.tar.gz
```

`pack/build_vr_pack.sh`:

1. Checks the tray: aarch64, `nvcc`, the GPUs' compute capability.
2. Fetches NCCL `2.32.6-1+cuda13.5` (`NCCL_VERSION`) into
   `~/vr-pack-build/nccl` (`include/` + `lib/`). First match wins:
   `--nccl-debs <dir>` (libnccl2 + libnccl-dev `.deb`s, e.g. from the
   release drop), then `apt-get download` from the tray's apt lists, then
   the `.deb`s from NVIDIA's CUDA repo (`NCCL_REPO_URLS`, ubuntu2604 then
   ubuntu2404 sbsa), all unpacked with `dpkg-deb -x` (no root). If none has
   it, the stock `NVIDIA/nccl` tag `v2.32.6-1` is built for the tray's GPU
   arch (slow; `--nccl-source` forces it). The header and the library must
   both be 2.32.6.
3. Builds Open MPI 4.1.6 `--without-slurm` (as GB300) into its own prefix
   `~/vr-pack-build/pack`, with Open MPI's bundled PMIx / libevent / hwloc,
   so it cannot clash with the system Open MPI 5 and needs no `-dev`
   packages. UCX is used when `libucx-dev` is installed. `-std=gnu17` keeps
   GCC 15 (Ubuntu 26.04) from compiling the 4.1 sources as C23. The
   download needs internet; otherwise `--ompi-tarball <file>`. It is built
   once and reused (`--clean` rebuilds).
4. Copies `--nccl-tests-src` (default `~/nccl-tests`; your checkout and its
   `build/` are left alone; `--nccl-tests-ref <tag>` picks a tag) and builds
   it with `MPI=1` against that Open MPI and the fetched NCCL
   (`NCCL_HOME=~/vr-pack-build/nccl`, not `/usr`). The GPU
   architecture comes from this tray's GPUs (`nvidia-smi --query-gpu=compute_cap`,
   SASS + PTX), so the Rubin GPUs are targeted; `NVCC_GENCODE` overrides.
5. It copies `libnccl.so.2` (the fetched 2.32.6),
   `libcudart.so.13` and every other library the binaries and Open MPI
   plugins need into `lib/`, except the base-OS floor the GB300 audit found
   (glibc, `libstdc++`, `libgcc_s`, `libz`, `libudev`) and the driver
   (`libcuda`, `libnvidia-*`). It fails if anything else is unresolved, or if
   a test binary would load a non-pack NCCL. It writes `PACK_INFO`
   (versions, checked by the pre-flight) and `PACK_LIBS` (the `ldd` audit).
6. Smoke run: the pack's own `mpirun` runs `all_reduce_perf` on the 4 GPUs
   of the build tray with only the pack's libraries (`--no-smoke` skips it).
   With `NCCL_DEBUG=VERSION`, the loaded NCCL must report 2.32.6. The
   tarball is only written if that passes.

`build_vr_pack.sh` is a single self-contained file, so copying just that
file to a tray is enough.

Then copy the tarball next to `deploy_vr_nccl_test.sh` on the jumper.

First build (node136, Nov 24 2026): Ubuntu 26.04, GCC 15.2, CUDA 13.5.37,
NCCL 2.32.6 from the release `.deb`s, nccl-tests v2.22.0, compute
capability 10.7 (`sm_107`). Pack 208 MB. Smoke run on 4 GPUs: All-Reduce
busbw ~630 GB/s at 128 MB, `#wrong` 0. Without `NCCL_IB_DISABLE=1` each
rank printed `libibverbs: Warning: couldn't load driver 'libionic-rdmav59.so'`
(an rdma-core provider listed in `/etc/libibverbs.d` but not installed);
harmless, and the test itself runs with IB disabled.

Pack layout (same as GB300, plus the two info files):

```
bin/      *_perf (nccl-tests), mpirun, orterun, orted, ompi_info, ...
lib/      libmpi.so.40, libopen-rte/-pal, libpmix, libhwloc, libevent*, libucx*,
          libnccl.so.2 (2.32.6), libcudart.so.13, lib/openmpi/ (MCA plugins)
share/    openmpi help files, wrapper data
PACK_INFO nccl=2.32.6-1+cuda13.5, cuda=13.5.x, openmpi=4.1.6, nccl_tests=<git describe>, gencode, built_on, ...
PACK_LIBS ldd of all_reduce_perf against the pack
```

## 2. Deploy (from the jumper)

```bash
cd vr200
./deploy_vr_nccl_test.sh rackNN.sh                  # CM_IPS from a GB300-style rack file
./deploy_vr_nccl_test.sh --hosts 192.168.101.136,192.168.101.137   # or a list, node-00 first
./deploy_vr_nccl_test.sh rackNN.sh my-pack.tar.gz   # default ./nccl-test-pack-vr-arm64.tar.gz
./deploy_vr_nccl_test.sh rackNN.sh --only <ip>      # re-push to a swapped tray only
./deploy_vr_nccl_test.sh rackNN.sh --bootstrap [password]
./deploy_vr_nccl_test.sh rackNN.sh --auto [-- 1 --info]   # deploy, then run the test
./deploy_vr_nccl_test.sh rackNN.sh --dry-run
```

node-00 is the LAST entry of `CM_IPS` (same reversal as GB300) or the first
of `--hosts`. `NCCL_MNNVL_UUID` is derived from the rack file name like GB300
(`rack17.sh` -> `0x17`, `--uuid` overrides; unset with `--hosts`, so NCCL
picks it). Steps:

0. Passwordless SSH jumper -> every tray (`--bootstrap` seeds the key with
   `sshpass`). Stale `known_hosts` entries are cleared (tray swap).
1. The pack is pushed to every tray **in parallel** and extracted under
   `--data-dir`. A tray whose cached tar.gz has the same sha256 is not
   copied again (GB300 compares sizes). The old `bin/ lib/ share/` are
   replaced, so no library of an older pack is left behind.
2. SSH mesh node-00 -> every tray, itself included, verified from node-00.
3. `run_vr_nccl_test.sh` and `vr_nccl.conf` (hosts, pack paths, UUID,
   pinned versions) are staged in `<data-dir>/run/` on node-00.
4. `run_vr_nccl_test.sh --check` runs the full pre-flight on node-00; the
   deploy exits non-zero if it fails (`--no-check` skips it).

`--user <name>` runs everything as a normal user instead of root (data dir
`~/portable-nccl`, repairs through `sudo -n`).

A reboot needs no re-deploy: the test repairs IMEX / channel0, clears stale
processes and re-extracts the pack from its local cache if it is damaged,
every run. Re-deploy after a tray swap, a reimage or a new pack.

## 3. Test (on node-00)

```bash
cd /root/portable-nccl/run
bash run_vr_nccl_test.sh --check        # pre-flight only, prints the mpirun lines
bash run_vr_nccl_test.sh 1              # node-00 only (4 GPUs)
bash run_vr_nccl_test.sh                # all trays
bash run_vr_nccl_test.sh --info         # NCCL INFO -> /tmp/nccl_info/ on each tray
bash run_vr_nccl_test.sh --no-nvls      # skip NVLink-switch multicast (NVLS)
bash run_vr_nccl_test.sh --no-mnnvl     # control run: no NVLink between trays (TCP, slow)
bash run_vr_nccl_test.sh --max-size 8G
```

Settings: built-in defaults < `vr_nccl.conf` < environment < command line.
Logs and hostfiles go to `<data-dir>/run/results/`.

### Target versions

Matching is by substring. Each can be overridden in the environment (on the
jumper it is then written to `vr_nccl.conf`); `""` skips that check.
`--versions-warn` (either script) turns FAIL into WARN, to try a newer drop.
All trays must also agree with each other on driver, IMEX and pack.

| Component | Where | Variable | Default | Mismatch |
|---|---|---|---|---|
| Driver, user space = kernel module | tray | `EXPECT_DRIVER` | `620.70.08` | FAIL |
| IMEX (`nvidia-imex*` package / `--version`) | tray | `EXPECT_IMEX` | `2.0.0` | FAIL on multi-tray runs |
| NCCL | pack `PACK_INFO` | `EXPECT_NCCL` + `EXPECT_NCCL_CUDA` | `2.32.6` + `cuda13.5` | FAIL |
| CUDA runtime | pack `PACK_INFO` | `EXPECT_CUDA` | `13.5.0` | FAIL |
| NVSHMEM (`nvshmem*` package) | tray | `EXPECT_NVSHMEM` | `3.8.0` | WARN |
| DOCA Host (`doca-host` package) | tray | `EXPECT_DOCA` | `3.3.3` | WARN |

### Pre-flight checks (per tray, over SSH)

- versions (table above); driver user space **and** kernel module must
  agree (finding 1); `nvidia-smi` works, 4 GPUs visible
- pack complete (else re-extracted from the cache); `*_perf`, `mpirun`,
  `orted` resolve every library, and NCCL / MPI come from the pack
- bootstrap interface: each tray reports its own (names differ per tray, e.g. `enx<MAC>`)
- multi-tray runs: IMEX node list, `nvidia-imex` active, IMEX channel0, fabric `State`
- fabric **health**: `Summary: Healthy`, `Route Unhealthy: False`
- `nvidia-smi topo -p2p r` is `OK` for every GPU pair
- stale test processes (`*_perf`, `orted`, `mpirun`) cleared
- across trays: same driver, IMEX and pack (sha256), same `ClusterUUID`

## Findings from the first two trays (bring-up, Nov 2026)

These were seen on the earlier dev stack (driver 620.62, NCCL 2.32.4gr100).

1. **Driver 615.62 -> 620.62.** 615.62 was installed with apt; installing the
   620.62 `.run` on top left both drivers on disk and the initramfs kept loading
   the 615 module (`NVRM version: 615.62`, NVML 620.62, "version mismatch").
   Fix: remove the `.run` driver, `apt purge` the 615.62 packages and its local
   repo, reboot, install 620.62 from `nvidia-driver-local-repo-ubuntu2604-620.62`.
   A "GPU failed probe-time init" after that was cleared by uninstall -> reboot ->
   reinstall.
2. **Fabric `Route Unhealthy: True`, P2P `NS` on every GPU pair, even inside one
   tray.** `dmesg`: `knvlinkCheckNvswitchP2pConfig` failed. On the GFM switch the
   default partition 32766 (`location_based`, 72 GPUs) was `unhealthy`: the rack
   is missing trays (slots 26, 27 and two GPUs of slot 18 show UUID 0). Routes
   later recovered without a partition change; cause not confirmed.

   **Observation worth repeating:** the recovery was first seen right after these
   two commands were run on the GFM switch (Nov 20 2026, 03:12:41 UTC), while
   backing up the controller state:

   ```bash
   nv action generate sdn state app nmx-controller type topology
   nv show sdn state app nmx-controller type topology files
   ```

   The `generate` action produced **one new topology state file** (it appears in
   the `show ... files` list), so it is not purely a no-op on the switch: the
   controller exported its current topology view to disk at that moment:

   ```
   /host/cluster_infra/app_state/nmx-controller/topology/nmx-controller_topology_20261120_031241
   ```

   Size 5,520,246 bytes (~5.5 MB): a full dump of the controller's view of the
   fabric (trays, GPUs, switch ports, links), not a small summary. Keep this file: if routes go unhealthy again, generate a new one and give both
   to NVIDIA for comparison.

   About seven minutes later (first check at 03:19:29 UTC) both trays showed `Health: Healthy`, `Route Unhealthy: False`,
   P2P `OK` on every GPU pair and the same `CliqueId`, and the default partition
   32766 went from `unhealthy` to `healthy`. Generating a state file is not expected
   to change routing, so this may be coincidence (the GFM recovering on its own, or
   someone else acting on the switch at the same time), but nothing else was
   changed on our side. If `Route Unhealthy: True` comes back, run these two
   commands first, note the time, and check `nv show sdn partition` and
   `nvidia-smi -q | grep -E "Summary|Route Unhealthy"` a few minutes later, so the
   next occurrence can confirm or rule out the link. `nv action update sdn
   partition 32766 reroute` is not an option: it returns `NMX_ST_NOT_SUPPORTED`
   on the default partition.
3. **Identical hostnames hang Open MPI 5** before NCCL starts (cloned OS image ->
   every tray is `localhost`). Open MPI 4 / HPC-X (rack pack, NGC containers)
   does not have this problem. Fix then: unique hostnames. Since the switch
   to the pack's Open MPI 4.1.6 this no longer matters.
4. **Cross-tray memory import refused: NVLink `bwMode` mismatch.** Across trays
   NCCL failed at the first transfer: `ncclP2pImportShareableBuffer` CUDA 401/400
   (`--no-nvls`), and with NVLS `ncclMcGroupBuildPartitions` CUDA 999. IMEX logged
   `OBJECT_NOT_FOUND` / `NV_ERR_CYCLE_DETECTED`; the real cause was in `dmesg`:
   `memoryfabricimportv2IsGpuMapAllowed_IMPL: bwMode does not match: owner 0,
   mapper 293` - the two trays' GPUs carried different fabric bandwidth modes
   (slot 2 = 293, slot 1 = 0) although driver, firmware, module options and
   `nvidia-smi -q` fabric sections were identical, and `nvidia-smi nvlink -gBwMode`
   is "not supported". `nv action update sdn partition 32766 reroute` returns
   `NMX_ST_NOT_SUPPORTED` on the default partition. **Rebooting both trays after
   the partition had become healthy fixed it** (both re-registered with the same
   mode). Check with `sudo dmesg -T | grep -i bwMode` after a cross-tray failure.

## Baselines (bring-up stack: driver 620.62, NCCL 2.32.4gr100, Nov 20 2026)

To be re-measured on the 1.0.0 RC stack.


| Run | All-Reduce busbw @ 32 GB | All-to-All busbw @ 32 GB |
|---|---|---|
| 1 tray, 4 GPUs | ~1080 GB/s | ~1100 GB/s |
| 2 trays, 8 GPUs, MNNVL + NVLS | **~1485 GB/s** | **~1099 GB/s** |

Two-tray `--info` run: `via P2P/MNNVL` between trays, "NVLS multicast support is
available", no `bwMode`/multicast errors in `dmesg`. `#wrong` 0 everywhere.

## GFM switch commands used

```bash
nv show cluster apps                     # nmx-controller version / state
nv show sdn partition                    # partitions + health
nv show sdn partition 32766 location     # GPU locations (chassis.slot.host.gpu) + UUIDs
nv action update sdn partition 32766 reroute
# logs: /var/log/nmx/nmx-c/fabricmanager.log, nvlsm.log  (use grep -a)
```
