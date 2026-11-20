# VR200 bring-up: NCCL test across compute trays

`run_vr_nccl_test.sh` runs nccl-tests (All-Reduce + All-to-All) across one or
more VR200 compute trays, launched from the first tray. It is derived from
`../deploy_rack_nccl_test.sh` (its `run_nccl_test.sh` part) but uses what is
installed on the trays instead of a test pack:

| | GB300 rack script | `run_vr_nccl_test.sh` |
|---|---|---|
| NCCL | pack's own `lib/` | system `libnccl2` (dev drop, e.g. `2.32.4gr100`) |
| MPI | pack's Open MPI 4 (`orted`) | Ubuntu Open MPI 5 (`prterun`) |
| nccl-tests | pack's `bin/` | built on each tray with `MPI=1` |
| Jumper / SSH mesh | yes | no: run on tray 1, needs SSH to every tray incl. itself |
| User | root | root or a normal user (repairs then use `sudo -n`) |

The NCCL settings match the rack script (MNNVL auto, NVLS on, IB off).

## Usage

On the first tray (first IP in `--hosts`, default `192.168.101.136,192.168.101.137`):

```bash
bash run_vr_nccl_test.sh --dry-run      # pre-flight only, prints the mpirun lines
bash run_vr_nccl_test.sh 1              # first tray only (4 GPUs)
bash run_vr_nccl_test.sh                # all trays
bash run_vr_nccl_test.sh --info         # NCCL INFO -> /tmp/nccl_info/ on each tray
bash run_vr_nccl_test.sh --no-nvls      # skip NVLink-switch multicast (NVLS)
bash run_vr_nccl_test.sh --no-mnnvl     # control run: no NVLink between trays (TCP, slow)
bash run_vr_nccl_test.sh --hosts ip1,ip2,ip3 --max-size 8G
```

nccl-tests is looked for in `$NCCL_TESTS_DIR`, `~/nccl-tests`, then
`/home/nvidia/nccl-tests`. Logs go to `~/nccl-results/`.

Build nccl-tests on every tray at the same path:

```bash
git clone https://github.com/NVIDIA/nccl-tests.git && cd nccl-tests
make -j MPI=1 MPI_HOME=/usr/lib/aarch64-linux-gnu/openmpi CUDA_HOME=/usr/local/cuda
```

## Pre-flight checks (per tray, over SSH)

- driver version in user space **and** kernel module (`EXPECT_DRIVER`, default 620.62)
- `nvidia-smi` works, 4 GPUs visible
- `libnccl2` version (`EXPECT_NCCL`, default 2.32.4gr100)
- nccl-tests built with MPI and linked to the system `libnccl.so.2`
- bootstrap interface: each tray reports its own (names differ per tray, e.g. `enx<MAC>`)
- multi-tray runs: IMEX node list, `nvidia-imex` active, IMEX channel0, fabric `State`
- fabric **health**: `Summary: Healthy`, `Route Unhealthy: False`
- `nvidia-smi topo -p2p r` is `OK` for every GPU pair
- stale test processes cleared
- across trays: same driver and NCCL, same `ClusterUUID`, **unique hostnames**

## Findings from the first two trays (Nov 2026)

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
   two commands were run on the GFM switch (Nov 20 2026, ~03:1x UTC), while backing
   up the controller state:

   ```bash
   nv action generate sdn state app nmx-controller type topology
   nv show sdn state app nmx-controller type topology files
   ```

   The `generate` action produced **one new topology state file** (it appears in
   the `show ... files` list), so it is not purely a no-op on the switch: the
   controller exported its current topology view to disk at that moment.

   Minutes later both trays showed `Health: Healthy`, `Route Unhealthy: False`,
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

## Baselines (driver 620.62, NCCL 2.32.4gr100, Nov 20 2026)

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
