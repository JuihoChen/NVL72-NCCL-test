#!/usr/bin/env python3
"""rank_gpus_from_info.py -- rank every GPU in a rack from NCCL INFO logs.

Input: a folder of per-rank NCCL debug files, as written by
NCCL_DEBUG=INFO + NCCL_DEBUG_FILE=/tmp/nccl_info/%h.%p.log and pulled back
by collect_nccl_info.sh (any layout under the folder; every *.log is read).

NCCL INFO says how each GPU *set up* (fabric state, transports, NVLS,
channels, init time), not how fast it ran, so the result is a health
ranking: worst GPU first. Every line NCCL writes starts with
"host:pid:tid [cudaDev]", which is how each line is tied to a GPU.

Scoring (higher = worse). A host with fewer than 4 GPUs in the logs gets
a +200 "never logged" row for each missing cudaDev. Comparisons are against the rack majority/median,
so a rack-wide setting never flags every GPU at once:
  +100  a communicator on this GPU never reached "Init COMPLETE"
  + 50  MNNVL fabric state not COMPLETED (3) or healthMask != 0
  + 50  MNNVL cliqueId / fabric UUID differs from the majority
  + 50  communicator came up with MNNVL 0 while the majority has MNNVL 1
  + 30  NVLS multicast not available while the majority has it
  + 20  coll/nvls/p2p channel counts differ from the majority
  +  5  per channel connection NOT over P2P (SHM / NET fallback)
  + 10  per NCCL WARN line
  + 10 x (init time / median - 1), only when slower than 1.5x median

  python3 rank_gpus_from_info.py <folder> [--csv out.csv]
"""

import argparse
import collections
import csv
import os
import re
import statistics
import sys

GPUS_PER_NODE = 4

PREFIX = re.compile(r'^(?P<host>[^\s:]+):(?P<pid>\d+):(?P<tid>\d+) \[(?P<dev>\d+)\] NCCL (?P<lvl>INFO|WARN) (?P<msg>.*)$')

RE_MNNVL_FABRIC = re.compile(r'MNNVL busId (?P<bus>0x[0-9a-fA-F]+) fabric UUID (?P<uuid>\S+) cliqueId (?P<clique>\S+) state (?P<state>\d+) healthMask (?P<mask>\S+)')
RE_COMM = re.compile(r'comm 0x[0-9a-f]+ rank (?P<rank>\d+) nRanks (?P<nranks>\d+) nNodes (?P<nnodes>\d+) localRanks (?P<lr>\d+) localRank (?P<lrank>\d+) MNNVL (?P<mnnvl>\d+)')
RE_INIT_DONE = re.compile(r'ncclCommInitRank(?:Config|Func)? comm 0x[0-9a-f]+ rank (?P<rank>\d+) nranks (?P<nranks>\d+) cudaDev (?P<dev>\d+) nvmlDev (?P<nvml>\d+) busId (?P<bus>[0-9a-fA-F]+).*Init COMPLETE')
RE_INIT_TIME = re.compile(r'Init timings.*?rank (?P<rank>\d+) nranks \d+ total (?P<total>[0-9.]+)')
RE_NVLS = re.compile(r'NVLS multicast support is (?P<not>not )?available on dev (?P<dev>\d+)')
RE_CHANNELS = re.compile(r'(?P<coll>\d+) coll channels, (?:(?P<collnet>\d+) collnet channels, )?(?:(?P<nvls>\d+) nvls channels, )?(?P<p2p>\d+) p2p channels')
RE_CONN = re.compile(r'Channel \d+/\d+ : \d+\[\d+\] -> \d+\[\d+\](?: \[(?:send|receive)\])? via (?P<via>\S+)')


class Gpu:
    def __init__(self, host, dev):
        self.host, self.dev = host, dev
        self.bus = None
        self.ranks = set()
        self.comms_started = 0      # "comm ... rank ... MNNVL" lines
        self.comms_done = 0         # "Init COMPLETE" lines
        self.fabric = []            # (uuid, clique, state, mask)
        self.mnnvl = []             # MNNVL flag per communicator
        self.nvls = None            # True/False/None
        self.channels = []          # (coll, nvls, p2p) per communicator
        self.via = collections.Counter()
        self.warns = []
        self.init_times = []
        self.files = set()

    @property
    def key(self):
        return (self.host, self.dev)


def parse(folder):
    gpus = {}
    nfiles = 0
    for root, _, names in os.walk(folder):
        for name in sorted(names):
            if not name.endswith('.log'):
                continue
            nfiles += 1
            path = os.path.join(root, name)
            with open(path, errors='replace') as fh:
                for line in fh:
                    m = PREFIX.match(line.rstrip('\n'))
                    if not m:
                        continue
                    host, dev, lvl, msg = m['host'], int(m['dev']), m['lvl'], m['msg']
                    g = gpus.get((host, dev))
                    if g is None:
                        g = gpus[(host, dev)] = Gpu(host, dev)
                    g.files.add(path)
                    if lvl == 'WARN':
                        g.warns.append(msg.strip())
                        continue
                    if (x := RE_CONN.search(msg)):
                        g.via[x['via'].split('/')[0]] += 1
                    elif (x := RE_MNNVL_FABRIC.search(msg)):
                        g.bus = g.bus or x['bus']
                        g.fabric.append((x['uuid'], x['clique'], int(x['state']), x['mask']))
                    elif (x := RE_COMM.search(msg)):
                        g.comms_started += 1
                        g.ranks.add(int(x['rank']))
                        g.mnnvl.append(int(x['mnnvl']))
                    elif (x := RE_INIT_DONE.search(msg)):
                        g.comms_done += 1
                        g.ranks.add(int(x['rank']))
                        g.bus = g.bus or ('0x' + x['bus'].lower())
                    elif (x := RE_INIT_TIME.search(msg)):
                        g.init_times.append(float(x['total']))
                    elif (x := RE_NVLS.search(msg)):
                        g.nvls = (x['not'] is None) if g.nvls is not False else False
                    elif (x := RE_CHANNELS.search(msg)):
                        g.channels.append((int(x['coll']), int(x['nvls'] or 0), int(x['p2p'])))
    return gpus, nfiles


def majority(values):
    values = [v for v in values if v is not None]
    return collections.Counter(values).most_common(1)[0][0] if values else None


def score(gpus):
    all_gpus = list(gpus.values())
    maj_fabric = majority([(f[0], f[1]) for g in all_gpus for f in g.fabric])
    maj_mnnvl = majority([m for g in all_gpus for m in g.mnnvl])
    maj_nvls = majority([g.nvls for g in all_gpus])
    maj_chan = majority([c for g in all_gpus for c in g.channels])
    times = [max(g.init_times) for g in all_gpus if g.init_times]
    med_time = statistics.median(times) if times else 0.0

    rows = []
    for g in all_gpus:
        pts, why = 0.0, []
        if not g.files:
            pts += 200; why.append('never logged (process missing or died before NCCL init)')
        elif g.comms_started > g.comms_done or (g.comms_started == 0 and g.comms_done == 0):
            pts += 100; why.append(f'init incomplete ({g.comms_done}/{g.comms_started} comms)')
        bad_state = [f for f in g.fabric if f[2] != 3 or f[3] not in ('0x0', '0', '0x00')]
        if bad_state:
            pts += 50; why.append(f'MNNVL state {bad_state[0][2]} healthMask {bad_state[0][3]}')
        if maj_fabric and any((f[0], f[1]) != maj_fabric for f in g.fabric):
            pts += 50; why.append('fabric UUID/cliqueId differs from rack')
        if maj_mnnvl == 1 and any(m != 1 for m in g.mnnvl):
            pts += 50; why.append('communicator without MNNVL')
        if maj_nvls is True and g.nvls is False:
            pts += 30; why.append('NVLS not available')
        if maj_chan and any(c != maj_chan for c in g.channels):
            odd = next(c for c in g.channels if c != maj_chan)
            pts += 20; why.append(f'channels coll/nvls/p2p {odd[0]}/{odd[1]}/{odd[2]} vs {maj_chan[0]}/{maj_chan[1]}/{maj_chan[2]}')
        non_p2p = sum(n for via, n in g.via.items() if via != 'P2P')
        if non_p2p:
            pts += 5 * non_p2p
            why.append('non-P2P connections: ' + ', '.join(f'{v}={n}' for v, n in g.via.items() if v != 'P2P'))
        if g.warns:
            pts += 10 * len(g.warns); why.append(f'{len(g.warns)} WARN (first: {g.warns[0][:80]})')
        t = max(g.init_times) if g.init_times else None
        if t is not None and med_time > 0 and t > 1.5 * med_time:
            pts += 10 * (t / med_time - 1); why.append(f'init {t:.2f}s vs median {med_time:.2f}s')
        rows.append({
            'host': g.host, 'cudaDev': g.dev, 'busId': g.bus or '?',
            'rank': ','.join(str(r) for r in sorted(g.ranks)) or '?',
            'score': round(pts, 1),
            'init_s': f'{t:.2f}' if t is not None else '?',
            'p2p_conns': g.via.get('P2P', 0),
            'nvls': {True: 'yes', False: 'NO', None: '?'}[g.nvls],
            'warns': len(g.warns),
            'reasons': '; '.join(why) or 'clean',
        })
    # Worst first; among equal scores, slower init first (the only
    # continuous signal left once everything else is clean).
    rows.sort(key=lambda r: (-r['score'], -(float(r['init_s']) if r['init_s'] != '?' else 0.0)))
    return rows, med_time


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('folder')
    ap.add_argument('--csv', help='also write the ranking to this CSV file')
    ap.add_argument('--expect', type=int, default=72, help='expected GPU count (default 72)')
    a = ap.parse_args()

    gpus, nfiles = parse(a.folder)
    if not gpus:
        sys.exit(f'ERROR: no NCCL INFO/WARN lines found in {nfiles} .log file(s) under {a.folder}')
    for host in {h for h, _ in gpus}:
        for dev in range(GPUS_PER_NODE):
            gpus.setdefault((host, dev), Gpu(host, dev))
    rows, med_time = score(gpus)

    hosts = sorted({r['host'] for r in rows})
    nmissing = sum(1 for g in gpus.values() if not g.files)
    print(f'=== {len(rows)} GPU(s) ({nmissing} never logged) on {len(hosts)} host(s) from {nfiles} file(s) | median init {med_time:.2f}s ===')
    if len(rows) != a.expect:
        print(f'WARNING: expected {a.expect} GPUs, found {len(rows)}')
        if len(hosts) * GPUS_PER_NODE < a.expect:
            print(f'         only {len(hosts)} host(s) logged anything -- whole nodes are missing')
    print(f'{"#":>3} {"host":<22} {"dev":>3} {"busId":<12} {"rank":<7} {"score":>6} {"init_s":>7} {"p2p":>5} {"nvls":>4} {"warn":>4}  reasons')
    for i, r in enumerate(rows, 1):
        print(f'{i:>3} {r["host"]:<22} {r["cudaDev"]:>3} {r["busId"]:<12} {r["rank"]:<7} {r["score"]:>6} {r["init_s"]:>7} '
              f'{r["p2p_conns"]:>5} {r["nvls"]:>4} {r["warns"]:>4}  {r["reasons"]}')

    if a.csv:
        with open(a.csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'CSV: {a.csv}')


if __name__ == '__main__':
    main()
