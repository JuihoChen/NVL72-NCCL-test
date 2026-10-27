# NVL72-NCCL-test
Deployment scripts and diagnostics for NCCL tests on 18-node/72-GPU GB300 racks — hostfile setup, IMEX pre-flight checks, and fixes for socket/NVLink hangs.

## Collecting NCCL INFO logs from all nodes

Step 1 of ranking GPUs from NCCL INFO output: collect the per-rank INFO logs
of one full-rack run so they can be analyzed. No ranking happens here, and
`deploy_rack_nccl_test.sh` / normal runs are unchanged. From the jumper, in
the folder with `rackXX.sh` (node-00 = the LAST IP in `CM_IPS`):

```bash
bash collect_nccl_info.sh rack17.sh --clean        # empty /tmp/nccl_info on all nodes

# switch node-00's run_nccl_test.sh to INFO (keeps a .warn backup)
ssh root@<node-00> "sed -i.warn 's#-x NCCL_DEBUG=WARN#-x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,GRAPH,NVLS,ENV -x NCCL_DEBUG_FILE=/tmp/nccl_info/%h.%p.log#' /root/portable-nccl/run/run_nccl_test.sh"
ssh root@<node-00> bash /root/portable-nccl/run/run_nccl_test.sh

# put WARN back -- restores the original file byte-for-byte
ssh root@<node-00> mv /root/portable-nccl/run/run_nccl_test.sh.warn /root/portable-nccl/run/run_nccl_test.sh

bash collect_nccl_info.sh rack17.sh                # pull logs from all nodes, in parallel
```

The collector prints files / lines / WARN count per node and writes
`results/nccl_info_<timestamp>/<node-ip>/*.log` plus one
`results/nccl_info_<timestamp>.tar.gz` next to the script. On node-00 it
also works without the rack file (it reads the IMEX node list). Avoid the
`COLL`/`ALL` debug subsystems: they log every collective call and do lower
the scores.
