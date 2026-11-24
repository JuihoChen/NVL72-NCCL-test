# VR200: NCCL tests across compute trays

Two scripts, the VR200 counterparts of the GB300 flow:

| | GB300 | VR200 |
|---|---|---|
| Deploy, from the jumper | `../deploy_rack_nccl_test.sh` | `deploy_vr_nccl_test.sh` |
| Test, on node-00 / tray 1 | generated `run_nccl_test.sh` | `run_vr_nccl_test.sh` (+ `vr_nccl.conf`) |
| NCCL | pack's own `lib/` | system `libnccl2`, stock release (2.32.6+cuda13.5) |
| MPI | pack's Open MPI 4 (`orted`) | Ubuntu Open MPI 5 (`prterun`) |
| nccl-tests | pack's `bin/` | stock source, built once on tray 1 with `MPI=1`, copied to every tray |
| User | root | root, or a normal user with `--user` (repairs then use `sudo -n`) |

Nothing on the trays is patched: no test pack, no `LD_LIBRARY_PATH`
overrides, no rebuilt NCCL. The NCCL settings match the rack script
(MNNVL auto, NVLS on, IB off).

## Target stack (VR NVL72 1.0.0 RC)

The pre-flight checks each tray against these defaults. Matching is by
substring, and each one can be overridden in the environment (on the
jumper, it is then written to `vr_nccl.conf`; set it to `""` to skip it).

| Component | Variable | Default | Mismatch |
|---|---|---|---|
| Driver (user space and kernel module) | `EXPECT_DRIVER` | `620.70.08` | FAIL |
| CUDA (`$CUDA_HOME/version.json`) | `EXPECT_CUDA` | `13.5.0` | FAIL |
| `libnccl2` | `EXPECT_NCCL` + `EXPECT_NCCL_CUDA` | `2.32.6` + `cuda13.5` | FAIL |
| IMEX (`nvidia-imex*` package / `--version`) | `EXPECT_IMEX` | `2.0.0` | FAIL on multi-tray runs |
| NVSHMEM (`nvshmem*` package) | `EXPECT_NVSHMEM` | `3.8.0` | WARN (not used by nccl-tests) |
| DOCA Host (`doca-host` package) | `EXPECT_DOCA` | `3.3.3` | WARN (not used by nccl-tests) |

`--versions-warn` (on either script) turns the FAIL mismatches into
warnings, for trying a newer drop. All trays must still match each other
(driver, CUDA, NCCL, IMEX).

## Deploy (from the jumper)

```bash
./deploy_vr_nccl_test.sh --hosts 192.168.101.136,192.168.101.137   # tray 1 first
./deploy_vr_nccl_test.sh rackNN.sh                  # or CM_IPS from a GB300-style rack file
./deploy_vr_nccl_test.sh rackNN.sh --set-hostnames vr200   # cloned image: rename to vr200-nNN
./deploy_vr_nccl_test.sh rackNN.sh --nccl-tests-src nccl-tests.tar.gz   # trays without internet
./deploy_vr_nccl_test.sh rackNN.sh --user nvidia    # SSH and run as nvidia instead of root
./deploy_vr_nccl_test.sh rackNN.sh --auto -- 1      # deploy, then run the test on 1 tray
./deploy_vr_nccl_test.sh rackNN.sh --dry-run        # print what would be done
```

Tray 1 (first in `--hosts`, or the LAST entry of `CM_IPS`, the same
reversal as the GB300 script) is the launch node. Steps:

0. Passwordless SSH jumper -> every tray (`--bootstrap [password]` seeds
   the jumper's key with `sshpass` first). Stale `known_hosts` entries are
   cleared, so a swapped tray on a reused IP is fine.
1. SSH mesh tray 1 -> every tray, itself included, verified from tray 1
   (mpirun and the pre-flight use it; a missing link would hang mpirun).
2. Hostnames must be unique: identical hostnames hang Open MPI 5 (finding 3
   below). `--set-hostnames <prefix>` sets `<prefix>-n00`, `-n01`, ... in
   tray order and writes a marked `# vr-nccl` block with all trays into
   `/etc/hosts`.
3. nccl-tests: the stock source (cloned on tray 1 from
   `NCCL_TESTS_REPO`, optionally at `NCCL_TESTS_REF`, or pushed with
   `--nccl-tests-src <dir|tar.gz>`) is built once on tray 1 with
   `make MPI=1 MPI_HOME=/usr/lib/aarch64-linux-gnu/openmpi CUDA_HOME=/usr/local/cuda`
   against the system NCCL (`libnccl-dev` must match `libnccl2`), then
   `build/` is copied to the same path on every tray. A stamp
   (NCCL / nvcc / Open MPI versions + source checksum) skips the build and
   the copies when nothing changed, so after a stack upgrade a re-deploy
   rebuilds by itself. `--rebuild` forces it; `NVCC_GENCODE` is passed to
   make when set.
4. `run_vr_nccl_test.sh` and `vr_nccl.conf` (host list, paths, optional
   `--uuid` and pinned versions) are staged on tray 1.
5. `run_vr_nccl_test.sh --check` runs the full pre-flight on tray 1. The
   deploy exits non-zero if it fails (`--no-check` skips it).

Layout on the trays (`--data-dir`, default `<user's home>/vr-nccl`):

```
vr-nccl/nccl-tests/            source + build/ (every tray, same path)
vr-nccl/run/run_vr_nccl_test.sh
vr-nccl/run/vr_nccl.conf
vr-nccl/run/results/           logs + hostfiles (tray 1)
```

A reboot needs no re-deploy: the test repairs IMEX / channel0 and clears
stale processes every time. Re-deploy after a tray swap, a reimage or a
stack upgrade.

## Test (on tray 1)

```bash
cd ~/vr-nccl/run
bash run_vr_nccl_test.sh --check        # pre-flight only, prints the mpirun lines
bash run_vr_nccl_test.sh 1              # first tray only (4 GPUs)
bash run_vr_nccl_test.sh                # all trays
bash run_vr_nccl_test.sh --info         # NCCL INFO -> /tmp/nccl_info/ on each tray
bash run_vr_nccl_test.sh --no-nvls      # skip NVLink-switch multicast (NVLS)
bash run_vr_nccl_test.sh --no-mnnvl     # control run: no NVLink between trays (TCP, slow)
bash run_vr_nccl_test.sh --max-size 8G
```

Settings come from built-in defaults < `vr_nccl.conf` < environment <
command line. Without a conf (the bring-up way) pass `--hosts ip1,ip2`
and the script looks for nccl-tests in `$NCCL_TESTS_DIR`, `~/nccl-tests`,
then `/home/nvidia/nccl-tests`; logs then go to `~/nccl-results/`.

### Pre-flight checks (per tray, over SSH)

- stack versions (table above); driver user space **and** kernel module
  must agree (finding 1)
- `nvidia-smi` works, 4 GPUs visible
- nccl-tests built with MPI and linked to the system `libnccl.so.2`
- bootstrap interface: each tray reports its own (names differ per tray, e.g. `enx<MAC>`)
- multi-tray runs: IMEX node list, `nvidia-imex` active, IMEX channel0, fabric `State`
- fabric **health**: `Summary: Healthy`, `Route Unhealthy: False`
- `nvidia-smi topo -p2p r` is `OK` for every GPU pair
- stale test processes cleared
- across trays: same driver, CUDA, NCCL and IMEX, same `ClusterUUID`, **unique hostnames**

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
   does not have this problem. Fix here: unique hostnames.
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
