# NVL72-NCCL-test
Deployment scripts and diagnostics for NCCL tests on 18-node/72-GPU GB300 racks — hostfile setup, IMEX pre-flight checks, and fixes for socket/NVLink hangs.

## Finding slow nodes

`find_slow_nodes.sh` isolates the tray(s) behind a low full-rack NCCL score
(below ~927 GB/s All-Reduce / ~670 GB/s All-to-All out-of-place busbw). It is
not staged by `deploy_rack_nccl_test.sh`; copy it next to `run_nccl_test.sh`
on node-00 and run it there after `run_nccl_test.sh` has run once since boot:

```bash
scp find_slow_nodes.sh root@<node-00>:/root/portable-nccl/run/
ssh root@<node-00> bash /root/portable-nccl/run/find_slow_nodes.sh            # single + pair
ssh root@<node-00> bash /root/portable-nccl/run/find_slow_nodes.sh --phases loo
```

Phases: `single` (one 4-GPU job per node), `pair` (2-node ring, a node is
suspect only when both its pairs are slow), `loo` (leave-one-out N-1 node
jobs, for slowdowns that only appear at full scale). Each job's peak
out-of-place busbw is compared to its phase median (`--threshold`, default
0.90). Jobs on disjoint nodes run concurrently, so the default single + pair
check is 6 short rounds (all 18 single-node jobs at once, then the ring's
pairs in two halves, for each collective); `--serial` turns that off. Output: a per-node PASS/SUSPECT table plus `results.csv` and per-job
logs under `run/results/find_slow_nodes_<timestamp>/`. Exits 2 when any node
is flagged.
