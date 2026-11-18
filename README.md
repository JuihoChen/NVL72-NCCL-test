# NVL72-NCCL-test
Deployment scripts and diagnostics for NCCL tests on 18-node/72-GPU GB300 racks — hostfile setup, IMEX pre-flight checks, and fixes for socket/NVLink hangs.

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

It is separate from the normal test: `run_nccl_test.sh` and
`deploy_rack_nccl_test.sh` are unchanged. `run_gpu_rank.sh` reuses the same
mpirun flags and NCCL env (plus `NCCL_COMM_ID` for the NCCL bootstrap), builds
the binary on node-00 with plain `gcc` on first use (no CUDA/NCCL/MPI headers:
it `dlopen()`s the pack's libnccl/libcudart), and copies it to every node.

From the jumper (node-00 = the LAST IP in `CM_IPS`):

```bash
scp gpu_rank/nccl_pairwise_rank.c gpu_rank/run_gpu_rank.sh root@<node-00>:/root/portable-nccl/run/
ssh root@<node-00> bash /root/portable-nccl/run/run_nccl_test.sh 1   # only if rebooted since the last test
ssh root@<node-00> bash /root/portable-nccl/run/run_gpu_rank.sh
```

Output goes to the screen and to `run/results/<timestamp>_<N>node_gpu_rank.log`,
with the full pair matrix in `..._gpu_rank_matrix.csv`. Options pass through
after `--`, e.g. `run_gpu_rank.sh -- -b 2G -n 20` (bytes per transfer, timed
iterations per round). `nccl_pairwise_rank --selftest [N] [slow_rank] [factor]`
shows the ranking on synthetic data without any GPU.
