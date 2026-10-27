# GB300 NCCL Diag Rack — SSH Mesh Investigation Findings

**Scope:** `deploy_rack_nccl_test.sh` / `run_nccl_test.sh`, v0.15 → v0.21
**Topology assumption confirmed for this fleet:** the jumper is always network-reachable
to every node in a rack, for the life of the rack. No disconnected field/customer
sites where the jumper cannot reach the rack at test time.

---

## 1. Original symptom

Running `run_nccl_test.sh` on node-00 required manually typing the root password
36 times (18 siblings × 2 SSH calls each in the pre-flight loop) and then hung
indefinitely at the `mpirun` stage with no further output.

## 2. Root causes identified

| # | Finding | Why it happened |
|---|---|---|
| 2.1 | `--bootstrap` seeds the **jumper's** own key to all nodes, not node-00's key, despite doc wording implying otherwise. | The jumper can't fetch a key from node-00 before any trust exists — bootstrapping necessarily uses whatever key is already local to the machine invoking the script. |
| 2.2 | Node-00's own key mesh (`[2/4]`) was **never verified** — only pushed. A broken/incomplete push could silently report success and only surface later as an `mpirun` hang. | No equivalent of `[0/4]`'s preflight check existed for the node-00→sibling direction. |
| 2.3 | The node-00→sibling mesh **does not reliably survive** reboot/reimage on this hardware — confirmed via a live customer-site transcript where 17 of 18 siblings needed a password, and node-00's own self-entry did not. | Consistent with the same class of state-loss already known for `nvidia-imex`/`channel0` ("lost on last reboot, as expected"). Root cause of the *loss itself* is outside these scripts (site provisioning/reimaging), not something deploy-time code can prevent. |
| 2.4 | `authorized_keys` on at least one node (.137) contained **duplicate copies** of node-00's key, interleaved with unrelated factory/AE-provisioning keys (`root@bu18-nv-ae-01`, `nv_ae@bu18-nv-ae-01`). | The original `grep -qxF ... \|\| echo ... >>` check-then-append pattern is not atomic and can be bypassed by slightly different repeat invocations. |

## 3. Design decisions made (and why alternatives were rejected)

- **Keep `--bootstrap` (password auth) on the jumper only.** Never move it to
  node-00. Reasoning: the rack root password must never touch a node that
  ships to a customer; moving it to node-00 would also require bundling
  `sshpass` into the pack for no net reduction in password-conveyance
  complexity (still needs to reach *some* machine non-interactively).
- **Add `[2b/4]`**: after `[2/4]` pushes node-00's key, the jumper has node-00
  itself test SSH to every sibling (`BatchMode=yes`) before staging anything.
  Turns a silent bad-mesh into an immediate, clear deploy-time failure.
- **`run_nccl_test.sh` gets its own fail-fast mesh check**, run first, before
  the IMEX/channel0 preflight loop. Password-free — on failure it names the
  repair script rather than attempting anything itself, and exits in
  seconds instead of hanging at `mpirun` with no way to prompt (no TTY,
  `BatchMode`-equivalent).
- **`run_nccl_test.sh --bootstrap <password>`** (two-process pattern, decided
  with the diag team): a second, deliberate password entrypoint *specifically
  for the case where a jumper round-trip isn't the fastest path* — reuses
  node-00's existing key, only invokes `sshpass` if the passwordless check
  already failed.
- **Split into `repair_ssh_mesh.sh` + unmodified `run_nccl_test.sh`** rather
  than one script with a `--bootstrap` flag: keeps "does this file ever
  handle a password" answerable by filename alone, and lets the diag team's
  tooling call the repair path independently of running a test.
- **Dedupe-on-write** (`{ cat existing; echo newkey; } | awk '!seen[$0]++' >
  tmp && mv`) replacing the old check-then-append, applied to all three
  places `authorized_keys` is touched (bootstrap, `[2/4]`, and
  `repair_ssh_mesh.sh`'s push). Collapses exact-duplicate lines — including
  the pre-existing factory keys — without removing any identity's access.
  Confirmed present and correctly escaped through both heredoc-generation
  layers; validated end-to-end on a fresh rack03 deploy with no manual
  password entry.

## 4. Current recommendation, given confirmed topology

**Since the jumper is always reachable, `repair_ssh_mesh.sh` is functionally
redundant with `./deploy_rack_nccl_test.sh rackXX.sh --bootstrap <password>`
run again from the jumper.** Both fix the same underlying problem (a broken
node-00↔sibling mesh); redeploy is simply the jumper-side path to the same
repair, and under this topology there's no scenario where it's unreachable.

Given that, the two realistic options are:

- **A — Retire `repair_ssh_mesh.sh`.** Simplify `run_nccl_test.sh`'s fail-fast
  message to point back at the jumper's `--bootstrap` redeploy instead of a
  second script. Removes one file, one code path, and one place `sshpass`
  needs to exist (node-00 no longer needs it in the pack at all).
- **B — Keep it, but only as a targeted/faster option**, not a required part
  of the flow: a full rack redeploy re-touches `[0/4]`–`[2b/4]` on all 18
  nodes to fix what's usually one node; `repair_ssh_mesh.sh --only <ip>`
  is scoped and faster, and doubles as the field same-IP-swap tool
  (`--redeploy` re-pushes the pack from node-00's local cache). This is a
  convenience/speed tradeoff, not a reachability one, now that reachability
  is off the table as a justification.

**Recommendation: (A) is the more consistent choice once reachability is
guaranteed** — it removes a redundant code path and a redundant password
surface (node-00 no longer needs `sshpass` bundled at all) for a real, if
modest, simplification. (B) is defensible only if the diag team specifically
values the speed of a single-node fix over the small maintenance cost of a
second script — worth asking them directly rather than deciding unilaterally,
since they're the ones operating it.

**Decision (2026-10-27): A.** `repair_ssh_mesh.sh` has never been used in
practice because the deploy script's own mesh handling works reliably.
The script change itself is still pending (see open items).

## 5. Open items

- [x] Decide A vs. B above with the diag team's input. → **A** (see above).
- [ ] If (A): remove `repair_ssh_mesh.sh` generation/staging from
      `deploy_rack_nccl_test.sh`, drop `sshpass` from the pack manifest,
      update `run_nccl_test.sh`'s fail-fast message to reference jumper
      redeploy instead.
- [ ] Diff a freshly-deployed sibling's `authorized_keys` (post-v0.17) against
      the .137 dump to confirm the dedupe fix leaves exactly one `root@root`
      line on a clean run. Confirmed by inspecting a node's `authorized_keys`.
- [ ] Point the diag team's automation at v0.17 specifically.
