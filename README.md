# NVL72-NCCL-test
Deployment scripts and diagnostics for NCCL tests on 18-node/72-GPU GB300 racks — hostfile setup, IMEX pre-flight checks, and fixes for socket/NVLink hangs.

## VR200 (Vera Rubin NVL72) compute trays

`vr200/` applies the same flow to VR200 trays, without changing the GB300
files above. `vr200/pack/build_vr_pack.sh`, run on one VR200 tray, builds a portable pack
(`nccl-test-pack-vr-arm64.tar.gz`: Open MPI 4.1.6 `--without-slurm`, stock
NCCL 2.32.6+cuda13.5, CUDA 13.5 runtime, nccl-tests).
`vr200/deploy_vr_nccl_test.sh` pushes it to every tray from the jumper and
generates `run_vr_nccl_test.sh` on node-00. The jumper needs only that one
script and the tarball. See [vr200/README.md](vr200/README.md).

```bash
bash vr200/pack/build_vr_pack.sh        # on one VR200 tray; copy the tarball to vr200/
cd vr200
./deploy_vr_nccl_test.sh rackNN.sh --auto
```

## No rack file? `--hosts` with an IP range or a file

Both deploy scripts take `--hosts` instead of `rackXX.sh`, e.g. at a field
site: `--hosts 10.0.0.11-28` is 18 nodes (10.0.0.11 = node-00 ... 10.0.0.28),
and `--hosts nodes.txt` reads IPs and/or ranges from a file (one or more per
line, `#` comments). Items can be mixed with commas
(`10.0.0.11-26,10.0.0.30,10.0.0.31`); the order given is the node order, so
node-00 is the first IP. `--only` takes the same forms.

```bash
./deploy_rack_nccl_test.sh --hosts 10.0.0.11-28 --uuid 0x17      # GB300 (give --uuid)
vr200/deploy_vr_nccl_test.sh --hosts 10.0.0.11-28                # VR200
```

## Ranking all GPUs by speed in one run

`gpu_rank/` ranks every GPU of the rack by NCCL point-to-point bandwidth from
**one** `mpirun` launch. All-Reduce / All-to-All can't do this: every GPU
waits for the slowest one, so all ranks report the same busbw.
`nccl_pairwise_rank` instead runs N-1 rounds; in round k every GPU sends to
GPU r+k and receives from GPU r-k at the same time, and times its own
transfer. After the last round every GPU pair has been measured, and each
GPU's score (mean GB/s of every measurement it took part in) sorts the rack
slowest -> fastest. A slow GPU drags all of its measurements down, while a
healthy GPU meets it in only a few, so it stands out clearly; healthy GPUs
also name it as their "worst partner".

How the rounds, the score and the output files work, with a diagram:
[gpu_rank/README.md](gpu_rank/README.md).

It is separate from the normal test: `run_nccl_test.sh` and
`deploy_rack_nccl_test.sh` are unchanged, and an already-staged
`run_nccl_test.sh` works as is. `run_gpu_rank.sh` reuses the same
mpirun flags and NCCL env (plus `NCCL_COMM_ID` for the NCCL bootstrap), builds
the binary on node-00 with plain `gcc` on first use (no CUDA/NCCL/MPI headers:
it `dlopen()`s the pack's libnccl/libcudart), and copies it to every node.

From the jumper (node-00 = the LAST IP in `CM_IPS`):

```bash
scp gpu_rank/nccl_pairwise_rank.c gpu_rank/run_gpu_rank.sh root@<node-00>:/root/portable-nccl/run/
ssh root@<node-00> bash /root/portable-nccl/run/run_gpu_rank.sh --preflight   # after a reboot
ssh root@<node-00> bash /root/portable-nccl/run/run_gpu_rank.sh               # otherwise
```

After a reboot, add `--preflight` instead of running `run_nccl_test.sh`
first. It runs only the pre-flight part of the staged `run_nccl_test.sh`
(SSH mesh check, IMEX node list, nvidia-imex/channel0 repair, stale-process
cleanup, pack check), cut out of that file up to its
`=== Pre-flight complete ===` line, and skips the All-Reduce / All-to-All
sweep. Without `--preflight` the script only checks node state, as before.
It then goes straight on to the GPU ranking in the same run. Pre-flight
lines are labelled `nNN/IP` (e.g. `[n07/192.168.41.147]`, n00 = node-00),
the same form as the ranking table, since every node of a rack can share
one hostname.

Output goes to the screen and to `run/results/<timestamp>_<N>node_gpu_rank.log`,
with the full pair matrix in `..._gpu_rank_matrix.csv`. Options pass through
after `--`, e.g. `run_gpu_rank.sh -- -b 2G -n 20` (bytes per transfer, timed
iterations per round). `nccl_pairwise_rank --selftest [N] [slow_rank] [factor]`
shows the ranking on synthetic data without any GPU.

The output has three tables: the speed ranking (slowest first, with each GPU's
worst partner), the same GPUs **by location** (rank 0..N-1 = node x GPU, with
each GPU's speed position `#`), and the **per-node mean** score. Hosts are
labelled `nNN/<ip>` in hostfile order, since all nodes may share one hostname.

**Healthy reference** (known-good GB300 NVL72 rack, NCCL 2.28.7, 1 GiB, 3 runs;
All-Reduce ~927 GB/s, All-to-All ~670 GB/s): rack median ~607-610 GB/s, every
GPU within ~94-108% of the median, per-node means ~96-107%, and no GPU
repeatedly named as worst partner. The tray order is stable across runs, so a
tray at ~96% is normal spread. Neighbour rounds (r+-1, r+-2) are naturally
slower (~210-500 GB/s shared) and are not a fault. On a bad (NG) rack, look for
a GPU or node well below this band (e.g. < 90%) that healthy GPUs share as
their worst partner.

## Collecting NCCL INFO logs from all nodes

A quick hard-fault check for a bad rack, before the pairwise ranking above.
NCCL INFO only covers setup (links, fabric, NVLS, channels), so it cannot rank
GPU speed, but a fault there is obvious: on a healthy GB300 NVL72 rack every
GPU shows `NVL[721.8]` (18 links), the same MNNVL clique of 72, NVLS
available and 32 channels; a GPU with a down link shows e.g. `NVL[681.7]`.

`run_nccl_test.sh` (generated by `deploy_rack_nccl_test.sh` v0.22+) takes an
opt-in `--info` switch. **Without it nothing changes** -- same mpirun argv
(`NCCL_DEBUG=WARN`), same output, same log name. With it, pre-flight empties
`/tmp/nccl_info` on every node, NCCL writes one INFO file per rank to
`/tmp/nccl_info/%h.%p.log` (subsystems `INIT,GRAPH,NVLS,ENV`; `COLL`/`ALL`
are left out because they lower the scores), and the log name gets an
`_info` suffix.

From the jumper, in the folder with `rackXX.sh` (node-00 = the LAST IP in
`CM_IPS`):

```bash
./deploy_rack_nccl_test.sh rack17.sh                      # re-stage run_nccl_test.sh (pack copy is skipped)
ssh root@<node-00> bash /root/portable-nccl/run/run_nccl_test.sh --info
bash collect_nccl_info.sh rack17.sh                       # pull logs from all nodes, in parallel
```

The collector prints files / lines / WARN count per node and writes
`results/nccl_info_<timestamp>/<node-ip>/*.log` plus one
`results/nccl_info_<timestamp>.tar.gz` next to the script. On node-00 it
also works without the rack file (it reads the IMEX node list).
