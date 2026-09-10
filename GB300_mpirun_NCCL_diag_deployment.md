# GB300 NVL72 — Slurm-free Multi-Node NCCL Diag Deployment

## Purpose

The existing SOP (`GB300_NVL72_NCCL_SOP_v6.md`) runs multi-node NCCL tests
through Slurm + Pyxis/Enroot (`sbatch` → `srun --mpi=pmix`). That's too much
overhead for the diag team to install and operate on every rack they touch.

This deployment replaces that path with a **bare `mpirun` + hostfile**
workflow — closer to NVIDIA's own field diag tooling — driven entirely from
one script, runnable rack-by-rack across the fleet from a single jump host.

No Slurm, no Ansible, no per-rack cluster setup. Just SSH, NFS, `enroot`,
and `mpirun`.

## Topology

- **Jumper**: `nv_ae@bu18-nv-ae-01`, e.g. working directory `~/carlonext`.
  Holds the rack inventory files and the compiled nccl-tests container
  image. Orchestrates everything via SSH — no GPU, no enroot needed here.
- **Rack inventory files**: `rackXX.sh`, one per physical rack, each
  defining bash arrays of IPs by role (`CM_IPS`, `CM_BMC_IPS`, `SW_BMC_IPS`,
  `IPS`, `PMC_BMC_IPS`). The NCCL test targets **`CM_IPS`** — the 18
  compute-tray host-OS IPs (Compute Module / "Compute Trays OS", root
  login). `CM_IPS` is listed high-IP-first in the file; the scripts reverse
  it so index 0 ("IP0") is treated as **node-00**, the launch/anchor node.
- **node-00**: the lowest-IP compute tray in the rack. Acts as the mpirun
  launch node, the NFS server for the container image, and the SSH
  "jump-in-one" point the diag team uses to actually fire the test.
- **The container**: `compiled-nccl-test-image+latest.sqsh`, an
  Enroot-importable image with nccl-tests built at `/var/nccl-tests/build`.

## What changed from the Slurm SOP

| | Slurm SOP | This deployment |
|---|---|---|
| Scheduler | `sbatch` + `srun --mpi=pmix` | bare `mpirun` over SSH |
| Container runtime | Pyxis (`--container-image`) | plain `enroot create` / `enroot start` |
| Node list | Slurm partition | `CM_IPS` array in `rackXX.sh` |
| Setup needed on diag team's side | Slurm client, Pyxis, partition config | SSH + NFS + enroot (already on the nodes) |
| Privilege | Slurm job user | root + `--allow-run-as-root` (mpirun's safety override; GB300 IMEX/device access needs root anyway, so this avoids fighting permissions for no real security gain on a single-tenant diag rack) |

## Production-line / customer-site split

Deployment and testing don't necessarily happen at the same time or place:
**`deploy_rack_nccl_test.sh` runs once on the production line**; the rack
then ships, and **`run_nccl_test.sh` runs later at the customer's site**,
potentially after multiple power cycles in transit. That timing split
drives two different requirements for the two things that get set up:

| | Survives reboot/shipping? | Where it's handled |
|---|---|---|
| Unpacked container (slow — multi-GB unsquash) | **Must persist** | Once, at deploy time, on a persistent disk |
| `nvidia-imex` / `channel0` (fast — a `mknod`) | **Cannot persist** (kernel resets it every boot, unconditionally) | Every time, at test time |

## End-to-end flow (`deploy_rack_nccl_test.sh`)

One script, one rack file as its argument, does all of the following:

1. **Image distribution** — copies the `.sqsh` **once** to node-00 (not all
   18 nodes), NFS-exports `/mnt/nccl-share` from node-00, and mounts it
   read-only on every sibling. Skips the copy entirely if node-00 already
   has a same-size copy (cheap re-runs).
2. **Per-node container creation, in parallel** — every node runs
   `enroot create` against that shared `.sqsh` to materialize its **own
   local** container, all 18 nodes at once (background jobs + `wait`, not
   one-at-a-time) since this is the genuinely slow step (unsquashing a
   multi-GB image). The test binaries run **inside** this container via
   `enroot start`; nothing is extracted onto the bare host filesystem, so
   the node's own CUDA/NCCL/OpenMPI environment is never touched.
   Idempotent — skipped if the container already exists.

   **Where the container actually lives matters.** `enroot`'s default
   storage location is `~/.local/share/enroot` — for root, that's under
   `/root/...`, which **is** the persistent system/OS disk. Since that
   disk must not be modified, every node's storage is redirected via
   `ENROOT_DATA_PATH` / `ENROOT_RUNTIME_PATH` / `ENROOT_CACHE_PATH` to a
   **separate, persistent data/scratch volume** supplied via the required
   `--data-dir <path>` flag. This is deliberately **not** `/tmp` (tmpfs) —
   tmpfs would solve the "don't touch the OS disk" problem but reintroduce
   the "vanishes on reboot" problem, which is exactly what can't happen
   here given the production-line-to-customer-site timing.

   Every node **verifies this at runtime**, not just by convention: it
   runs `findmnt` on both `/` and `--data-dir` and refuses to proceed on
   that node if they resolve to the same underlying filesystem. A typo'd
   `--data-dir` fails loudly instead of silently writing onto the OS disk.
3. **SSH key meshing** — node-00 generates a keypair (if it doesn't have
   one) and gets it appended to every sibling's `authorized_keys`. mpirun
   launches node→node, not jumper→node, so this is what lets node-00 reach
   the rest of the rack. This same mesh is also what lets node-00 run the
   pre-flight (below) against every sibling at test time.
4. **Staging** — generates and pushes `run_nccl_test.sh` straight onto
   node-00 at `/mnt/nccl-share/run_nccl_test.sh`. By the time
   `deploy_rack_nccl_test.sh` finishes, node-00 is **fully ready** — no
   second script needed, and nothing about the container needs to be
   redone later, including at the customer's site.
5. **(optional, `--auto`)** — SSHes into node-00 and runs it immediately,
   streaming output back to the jumper and saving a local log. Not
   typically used here since deploy and test happen at different times/
   places, but available for same-site testing.

### Pre-flight is checked at *test* time, not deploy time

Three things get checked — and repaired if needed — **inside
`run_nccl_test.sh`**, every single time it's invoked, against **every
node** in the rack (node-00 SSHes out to itself and each sibling, using the
key mesh from step 3):

1. Write `/etc/nvidia-imex/nodes_config.cfg` (idempotent, same content
   every time).
2. If `nvidia-imex` isn't `active`, restart it and wait (retries) for it
   to reach `active` state.
3. If `/dev/nvidia-caps-imex-channels/channel0` doesn't exist, pull the
   major number from `/proc/devices` and recreate it:
   ```bash
   MAJOR=$(cat /proc/devices | grep nvidia-caps-imex-channels | awk '{print $1}')
   mkdir -p /dev/nvidia-caps-imex-channels
   rm -f /dev/nvidia-caps-imex-channels/channel0
   mknod /dev/nvidia-caps-imex-channels/channel0 c $MAJOR 0
   chmod 0666 /dev/nvidia-caps-imex-channels/channel0
   ```
4. **Safety net only**: if the container is somehow missing from
   `--data-dir` (e.g. the data volume itself was swapped), recreate it —
   this should normally never trigger, since the container persists
   across reboots by design.

Only after every node passes does it move on to the actual `mpirun`
invocation. This means **no redeploy is needed after a reboot** at the
customer's site — `run_nccl_test.sh` self-heals the fast-changing pieces
(`nvidia-imex`/`channel0`) every time, while the slow piece (the
container) was already handled once, permanently, back at the factory.

## What actually runs on node-00 (`run_nccl_test.sh`)

```bash
mpirun -np 72 -N 4 --hostfile /etc/nvidia-imex/nodes_config.cfg \
  --bind-to none --oversubscribe \
  -x NCCL_DEBUG=WARN -x NCCL_MNNVL_ENABLE=2 -x NCCL_NVLS_ENABLE=1 \
  -x NCCL_P2P_DISABLE=0 -x NCCL_IB_DISABLE=1 -x CUDA_IPC_HANDLE_SHARING_SUPPORT=1 \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 -x NCCL_MNNVL_UUID=0x17 -x NCCL_MIN_CTAS=32 \
  -x ENROOT_DATA_PATH -x ENROOT_RUNTIME_PATH -x ENROOT_CACHE_PATH \
  --mca btl tcp,self --mca btl_tcp_if_include <auto-detected iface> --allow-run-as-root \
  --mca coll_hcoll_enable 0 \
  enroot start --root \
    --mount /dev/nvidia-caps-imex-channels:/dev/nvidia-caps-imex-channels \
    --mount /dev/nvidia-caps:/dev/nvidia-caps \
    nccltest -- /var/nccl-tests/build/all_reduce_perf -b 8 -e 32G -f 2 -g 1
```

Notable points:
- **`--hostfile` is `/etc/nvidia-imex/nodes_config.cfg` itself** — same file
  used by IMEX, same convention as the original reference diag log. One
  less file to keep in sync.
- The outbound interface for `--mca btl_tcp_if_include` is auto-detected at
  run time via `ip route get <next node> | sed ...`, not hardcoded.
- Each rank's actual program is `enroot start ... -- <binary>` — mpirun's
  `orted` daemons run natively on the host, but the NCCL test process
  itself executes inside the container.
- `ENROOT_DATA_PATH`/`RUNTIME_PATH`/`CACHE_PATH` are forwarded via `-x` (no
  value — picked up from node-00's own environment) so every rank's
  `enroot start` looks for the container at the same persistent
  `--data-dir` location it was created in at deploy time.
- Runs both **All-Reduce** and **All-to-All** passes, logs to
  `/mnt/nccl-share/results/<timestamp>.log` on node-00, pullable via
  `scp root@<node-00>:/mnt/nccl-share/results/*.log`.

## `NCCL_MNNVL_UUID`

Not a magic constant — it's an explicit override of the MNNVL NVLink-domain
"clique" tag, normally assigned automatically by Fabric Manager. Setting it
lets you soft-partition jobs: NCCL only treats ranks sharing the same UUID
as one NVLink domain. The reference log's `0x1969` had no special meaning;
this deployment instead **auto-derives it from the rack filename**
(`rack17.sh` → `0x17`, `rack8.sh` → `0x08`) so every run is traceable back
to a specific rack. Override anytime with `--uuid 0xNNNN`.

## Re-run / node-swap safety

Built specifically for the diag team's workflow of swapping a node after a
field error and re-testing:

- **Idempotent by default** — NFS export/mount, container creation, and SSH
  key meshing are all check-before-act; running `deploy_rack_nccl_test.sh` again on
  an unchanged rack does almost nothing.
- **Stale SSH host keys are cleared automatically** — swapped hardware on a
  reused IP gets a new host key, which `ssh` normally refuses to reconnect
  to. The script runs `ssh-keygen -R` on every node it's about to touch
  before doing anything else.
- **`--only <ip1,ip2,...>`** limits the *container/NFS rebuild* steps to
  just the swapped node(s), so re-running deploy after one node's hardware
  changes doesn't touch the rest of the rack. If node-00 itself is in that
  list, the script automatically falls back to a full rack pass (it's the
  NFS/image source for everyone else). Note this only affects the deploy
  steps — the IMEX pre-flight inside `run_nccl_test.sh` always covers the
  full rack on every test run regardless, since it's cheap and needs to
  catch reboots on *any* node, not just recently-swapped ones.
- If a replacement node gets a **new IP**, just update `CM_IPS` in the
  rack's `rackXX.sh` file first — everything downstream is driven off that
  array.

## Usage

```bash
# One rack, default sqsh (./compiled-nccl-test-image+latest.sqsh)
./deploy_rack_nccl_test.sh rack17.sh

# Then, on the diag side:
ssh root@<node-00 ip>
bash /mnt/nccl-share/run_nccl_test.sh

# Override the UUID explicitly
./deploy_rack_nccl_test.sh rack17.sh --uuid 0x1969

# Have the jumper run the test for you instead of doing it by hand
./deploy_rack_nccl_test.sh rack17.sh --auto

# Preview every action without touching any node
./deploy_rack_nccl_test.sh rack17.sh --dry-run

# Check which version of the script you're running
./deploy_rack_nccl_test.sh --version

# After swapping one node, redeploy just that node
./deploy_rack_nccl_test.sh rack17.sh --only 192.168.14.191

# Whole fleet, one rack at a time
./test_all_racks.sh                 # all rack*.sh next to the scripts
./test_all_racks.sh --auto          # unattended, runs mpirun on every rack too
```

## Versioning

Both scripts carry a `SCRIPT_VERSION` and respond to `--version`
(short-circuits everything else, no rack file required). Currently
**`0.1`** for both — bump this in the script header as changes land, so
the diag team can confirm which build is on the jumper.

## Files delivered

| File | Purpose |
|---|---|
| `deploy_rack_nccl_test.sh` | The whole pipeline for one rack: image distribution, container creation, IMEX bring-up, SSH mesh, staging, optional auto-run. Takes `rackXX.sh` as its primary argument. |
| `test_all_racks.sh` | Thin loop over every `rackXX.sh` found next to it, calling `deploy_rack_nccl_test.sh` per rack and writing a pass/fail summary under `results/`. |

Both expect to live in the same directory as the rack inventory files and
the `.sqsh` image (e.g. `~/carlonext` on the jumper).

## Known assumptions / caveats

- **`apt-get` access**: `nfs-kernel-server`/`nfs-common` installs assume the
  compute trays can reach a package mirror (or it's already installed).
  Airgapped racks will need that swapped for a pre-staged package or a
  hard failure with a clear message instead.
- **`nodes_config.cfg` ownership**: the script overwrites
  `/etc/nvidia-imex/nodes_config.cfg` directly. If anything else on these
  nodes manages that file independently, this will clobber it.
- **`nvidia-imex` restarts only when needed**: the pre-flight inside
  `run_nccl_test.sh` only restarts the service if it's found *not* active —
  it doesn't unconditionally bounce it on every run. A healthy rack sees no
  fabric disruption at all when the test is re-run.
- **GPU device passthrough into the container**: only
  `/dev/nvidia-caps-imex-channels` and `/dev/nvidia-caps` are mounted
  explicitly. GPU devices themselves (`/dev/nvidia0`, `/dev/nvidiactl`,
  etc.) are assumed to be injected automatically via the standard NVIDIA
  enroot hook (typical for NGC-derived images). If the `.sqsh` wasn't built
  with that hook configured, the container won't see the GPUs and
  additional `--mount` lines will be needed.
- **`--only` does not change `nodes_config.cfg`'s membership** — it assumes
  the swapped node kept its IP/role. If the rack's node *count* changes,
  run a full pass (no `--only`) so every node's copy of the fabric config
  stays in sync.
