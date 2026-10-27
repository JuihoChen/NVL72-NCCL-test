# NVL72-NCCL-test
Deployment scripts and diagnostics for NCCL tests on 18-node/72-GPU GB300 racks — hostfile setup, IMEX pre-flight checks, and fixes for socket/NVLink hangs.

## Ranking all 72 GPUs from one INFO run

nccl-tests only prints rack-wide bandwidth, so a slow GPU drags every rank
down without being named. `rank_gpus_from_info.py` ranks each GPU (worst
first) from the NCCL INFO logs of one full-rack run instead: missing or
incomplete init, MNNVL fabric state/health, NVLS availability, channel
counts, non-P2P (SHM/NET) fallback connections, WARN lines and init time,
each compared against the rest of the rack. It is a health ranking, not a
bandwidth one: INFO says how a GPU set up, not how fast it ran.

1. Run the full rack with INFO going to per-rank files (until a switch
   exists, edit the `-x NCCL_DEBUG=WARN` line of the generated
   `run/run_nccl_test.sh` on node-00, and `rm -rf /tmp/nccl_info` on every
   node first):
   `-x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,GRAPH,NVLS,ENV -x NCCL_DEBUG_FILE=/tmp/nccl_info/%h.%p.log`
   Avoid the `COLL`/`ALL` subsystems: they log every collective call and do
   lower the scores.
2. Copy `collect_nccl_info.sh` and `rank_gpus_from_info.py` into
   `/root/portable-nccl/run/` on node-00 and run
   `bash collect_nccl_info.sh`. It pulls `/tmp/nccl_info/*.log` from all 18
   nodes in parallel into `run/results/nccl_info_<timestamp>/` and writes
   `gpu_ranking.txt` / `gpu_ranking.csv` there.
