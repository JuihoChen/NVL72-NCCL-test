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
3. **Identical hostnames hang Open MPI 5** before NCCL starts (cloned OS image ->
   every tray is `localhost`). Open MPI 4 / HPC-X (rack pack, NGC containers)
   does not have this problem. Fix here: unique hostnames.
4. **Open:** across trays, NCCL fails at the first transfer importing the peer
   tray's memory: `ncclP2pImportShareableBuffer` CUDA 401/400 (`--no-nvls`), and
   with NVLS `ncclMcGroupBuildPartitions` CUDA 999. Same tray works (All-Reduce
   ~1080 GB/s, All-to-All ~1100 GB/s busbw at 32 GB). Suspects: the still
   unhealthy partition 32766, and IMEX 2.0.0.rel47 (trays) vs GFM 2.0.0.rel51.

## GFM switch commands used

```bash
nv show cluster apps                     # nmx-controller version / state
nv show sdn partition                    # partitions + health
nv show sdn partition 32766 location     # GPU locations (chassis.slot.host.gpu) + UUIDs
nv action update sdn partition 32766 reroute
# logs: /var/log/nmx/nmx-c/fabricmanager.log, nvlsm.log  (use grep -a)
```
