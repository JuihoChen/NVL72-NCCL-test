# NVL72-NCCL-test
Deployment scripts and diagnostics for NCCL tests on 18-node/72-GPU GB300 racks — hostfile setup, IMEX pre-flight checks, and fixes for socket/NVLink hangs.

## Collecting NCCL INFO logs from all nodes

Step 1 of ranking GPUs from NCCL INFO output: collect the per-rank INFO logs
of one full-rack run back to node-00 so they can be analyzed. No ranking
happens here.

1. Run the full rack with INFO going to per-rank files. Until a switch
   exists, run `rm -rf /tmp/nccl_info` on every node, then edit the
   `-x NCCL_DEBUG=WARN` line of the generated `run/run_nccl_test.sh` on
   node-00 to:
   `-x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,GRAPH,NVLS,ENV -x NCCL_DEBUG_FILE=/tmp/nccl_info/%h.%p.log`
   Avoid the `COLL`/`ALL` subsystems: they log every collective call and do
   lower the scores. Put the line back to `WARN` afterwards.
2. Copy `collect_nccl_info.sh` into `/root/portable-nccl/run/` on node-00
   and run `bash collect_nccl_info.sh`. It pulls `/tmp/nccl_info/*.log`
   from all 18 nodes in parallel into
   `run/results/nccl_info_<timestamp>/<node-ip>/`, prints files / lines /
   WARN count per node, and packs everything into
   `run/results/nccl_info_<timestamp>.tar.gz`.
