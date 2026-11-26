#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch nuPlan's 10 Hz camera images and log databases for a list of OpenScene logs.

OpenScene is nuPlan's public sensor subset resampled to 2 Hz; nuPlan itself records its 8 cameras at 10 Hz.
For each listed log this fetches, from nuPlan v1.1 (you must have accepted the nuPlan terms at nuplan.org):

  <out>/nuplan_db/<log>.db                               the log database (image table: every camera frame with
                                                         its timestamp; ego_pose; camera calibration)
  <out>/sensor_blobs/sweeps/<log>/<CAM>/<token>.jpg      every camera frame that is not already an OpenScene
                                                         2 Hz frame in <out>/sensor_blobs/trainval/<log>/<CAM>/

Only the listed logs' bytes are downloaded, not whole archives:
- The log archives are zips: their central directory gives each .db's offset; one range request per .db.
- The camera archives (``*_camera_<N>.zip``) are tar files despite the name, with each log's images stored
  together. The script reads a small window every ``--probe-step-mb`` to see which log sits where, bisects to
  where each wanted log starts, then streams from there and stops at the next log.

Probe results go to <out>/nuplan_fetch/ and finished logs are marked there, so a rerun resumes. Files are written
to ``*.part`` and renamed. At the end every log is checked against its database's image table and a summary is
written to <out>/nuplan_fetch/summary.json.

Standard library only (Python >= 3.8); copy this file and the log list to the download machine:

    python3 fetch_openscene_sweeps.py --logs openscene_lvpgh_logs.txt --out ~/data/openscene --dry-run
    python3 fetch_openscene_sweeps.py --logs openscene_lvpgh_logs.txt --out ~/data/openscene

``--dry-run`` downloads the databases and probes the camera archives, then prints where each log is and the
estimated download size without fetching images.
"""

import argparse
import concurrent.futures as cf
import json
import os
import sqlite3
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

BASE = "https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1"
DB_ARCHIVES = ["nuplan-v1.1_train_pittsburgh.zip", "nuplan-v1.1_train_boston.zip", "nuplan-v1.1_train_singapore.zip"] \
    + [f"nuplan-v1.1_train_vegas_{i}.zip" for i in range(1, 7)] + ["nuplan-v1.1_val.zip", "nuplan-v1.1_test.zip"]
CAMERA_PARTS = {"train": 43, "val": 12, "test": 12}
CAMS = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0"]
LOCK = threading.Lock()


def log(msg):
    with LOCK:
        print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------------------------------------------- HTTP
def _open(url, start, end=None, timeout=120):
    rng = f"bytes={start}-" if end is None else f"bytes={start}-{end}"
    return urllib.request.urlopen(urllib.request.Request(url, headers={"Range": rng}), timeout=timeout)


def get_range(url, start, end, tries=8):
    for k in range(tries):
        try:
            with _open(url, start, end) as r:
                data = r.read()
            if len(data) == end - start + 1:
                return data
            raise IOError(f"short read {len(data)} of {end - start + 1}")
        except (urllib.error.URLError, OSError):
            if k == tries - 1:
                raise
            time.sleep(min(60, 2 ** k))


def content_length(url):
    for k in range(8):
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=60) as r:
                return int(r.headers["Content-Length"])
        except (urllib.error.URLError, OSError):
            if k == 7:
                raise
            time.sleep(min(60, 2 ** k))


# ---------------------------------------------------------------------------------------------------- .db (zip)
def zip_members(url):
    """{name: (local header offset, compressed size, method, crc)} from the zip's central directory."""
    n = content_length(url)
    tail = get_range(url, max(0, n - 65536), n - 1)
    i = tail.rfind(b"PK\x05\x06")
    _, _, _, _, count, cd_size, cd_off, _ = struct.unpack("<4s4H2LH", tail[i:i + 22])
    j = tail.rfind(b"PK\x06\x06")
    if j >= 0:
        z = struct.unpack("<4sQ2H2L4Q", tail[j:j + 56])
        cd_size, cd_off = z[8], z[9]
    cd = get_range(url, cd_off, cd_off + cd_size - 1)
    out, p = {}, 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        f = struct.unpack("<4s6H3L5H2L", cd[p:p + 46])
        method, crc, csize, usize, nl, el, cl, off = f[4], f[7], f[8], f[9], f[10], f[11], f[12], f[16]
        name = cd[p + 46:p + 46 + nl].decode()
        extra, q = cd[p + 46 + nl:p + 46 + nl + el], 0
        while q + 4 <= len(extra):  # zip64 extended information
            hid, hl = struct.unpack("<HH", extra[q:q + 4])
            d, k = extra[q + 4:q + 4 + hl], 0
            if hid == 1:
                for field in ("usize", "csize", "off"):
                    if {"usize": usize, "csize": csize, "off": off}[field] == 0xFFFFFFFF:
                        v = struct.unpack("<Q", d[k:k + 8])[0]
                        k += 8
                        usize, csize, off = (v, csize, off) if field == "usize" else \
                            (usize, v, off) if field == "csize" else (usize, csize, v)
            q += 4 + hl
        out[name] = (off, csize, method, crc)
        p += 46 + nl + el + cl
    return out


def fetch_db(url, member, dest):
    off, csize, method, crc = member
    head = get_range(url, off, off + 29)
    nl, el = struct.unpack("<HH", head[26:30])
    start = off + 30 + nl + el
    tmp = dest.with_suffix(".db.part")
    dec = zlib.decompressobj(-15) if method == 8 else None
    got, c = 0, 0
    with _open(url, start, start + csize - 1, timeout=300) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            got += len(chunk)
            data = dec.decompress(chunk) if dec else chunk
            c = zlib.crc32(data, c)
            f.write(data)
        if dec:
            tail = dec.flush()
            c = zlib.crc32(tail, c)
            f.write(tail)
    if got != csize or c != crc:
        tmp.unlink(missing_ok=True)
        raise IOError(f"{dest.name}: got {got} of {csize} bytes, crc {c:08x} vs {crc:08x}")
    tmp.rename(dest)


def fetch_dbs(logs, out, workers):
    db_dir = out / "nuplan_db"
    db_dir.mkdir(parents=True, exist_ok=True)
    want = {l for l in logs if not (db_dir / f"{l}.db").exists()}
    log(f"databases: {len(logs) - len(want)} present, {len(want)} to fetch")
    jobs = []
    for arch in DB_ARCHIVES:
        if not want:
            break
        url = f"{BASE}/{arch}"
        members = zip_members(url)
        for name, m in members.items():
            stem = os.path.basename(name)[:-3] if name.endswith(".db") else None
            if stem in want:
                jobs.append((url, m, db_dir / f"{stem}.db"))
                want.discard(stem)
        log(f"  {arch}: {sum(1 for j in jobs if j[0] == url)} wanted")
    with cf.ThreadPoolExecutor(workers) as ex:
        for fut in cf.as_completed([ex.submit(fetch_db, *j) for j in jobs]):
            fut.result()
    if want:
        log(f"WARNING: no database found for {len(want)} logs: {sorted(want)[:5]}")
    return want


# ---------------------------------------------------------------------------------------------------- camera (tar)
def _octal(b):
    if b and b[0] & 0x80:  # base-256
        return int.from_bytes(b[1:], "big")
    s = b.split(b"\0")[0].strip()
    return int(s, 8) if s else 0


def tar_header(h):
    """(path, size, type) of a valid ustar header block, else None."""
    if len(h) < 512 or h[257:262] != b"ustar":
        return None
    try:
        chk = _octal(h[148:156])
    except ValueError:
        return None
    if chk != sum(h[:148]) + 256 + sum(h[156:512]):
        return None
    name, prefix = h[:100].split(b"\0")[0].decode(), h[345:500].split(b"\0")[0].decode()
    return (f"{prefix}/{name}" if prefix else name), _octal(h[124:136]), h[156:157]


def log_of(path):
    """<archive root>/<log>/<CAM>/<file> -> <log> (None for the root entry)."""
    parts = path.strip("/").split("/")
    return parts[1] if len(parts) >= 2 else None


def first_header(url, off, span, size):
    """First valid tar header at or after ``off``: (offset, path), or None before the end of the archive."""
    off -= off % 512
    while off < size:
        buf = get_range(url, off, min(off + span, size) - 1)
        for p in range(0, len(buf) - 511, 512):
            h = tar_header(buf[p:p + 512])
            if h:
                return off + p, h[0]
        off += len(buf) - 512 - (len(buf) - 512) % 512
    return None


def probe_archive(url, size, step, span, workers, cache):
    if cache.exists():
        return json.loads(cache.read_text())
    offs = list(range(0, size, step))
    with cf.ThreadPoolExecutor(workers) as ex:
        hits = list(ex.map(lambda o: first_header(url, o, span, size), offs))
    probes = [[o, h[0], log_of(h[1])] for o, h in zip(offs, hits) if h]
    cache.write_text(json.dumps({"url": url, "size": size, "probes": probes}))
    return {"url": url, "size": size, "probes": probes}


def runs_of(probes, wanted):
    """Maximal runs of consecutive probes on one wanted log: (log, lo, hi, est_bytes), lo/hi = probe offsets of the
    last probe before the run (or 0) and of its first probe."""
    runs, i = [], 0
    while i < len(probes):
        lg = probes[i][2]
        j = i
        while j + 1 < len(probes) and probes[j + 1][2] == lg:
            j += 1
        if lg in wanted:
            lo = probes[i - 1][0] if i > 0 else 0
            end = probes[j + 1][0] if j + 1 < len(probes) else probes[j][0]
            runs.append((lg, lo, probes[i][0], end - probes[i][0]))
        i = j + 1
    return runs


def find_start(url, target, lo, hi, span, size):
    """Header offset at or before the first member of ``target`` in (lo, hi]: the header at or after lo is not
    ``target``, the one at or after hi is."""
    while hi - lo > 4 * span:
        mid = (lo + hi) // 2
        h = first_header(url, mid, span, size)
        if h is None or h[0] >= hi or log_of(h[1]) == target:
            hi = mid
        else:
            lo = mid
    h = first_header(url, lo, span, size)
    return h[0] if h else lo


def stream_log(url, target, start, out, keep_keyframes, limit):
    """Stream tar members from header offset ``start``; write ``target``'s images; stop at the next log."""
    sweeps, keys = out / "sensor_blobs" / "sweeps" / target, out / "sensor_blobs" / "trainval" / target
    off, seen, n_new, n_key, n_have = start, False, 0, 0, 0
    pending_path = None
    for attempt in range(20):
        try:
            with _open(url, off, timeout=300) as r:
                while True:
                    hb = r.read(512)
                    if len(hb) < 512 or hb == b"\0" * 512:
                        return n_new, n_key, n_have
                    h = tar_header(hb)
                    if h is None:
                        raise IOError(f"bad tar header at {off}")
                    path, size, typ = h
                    padded = (size + 511) // 512 * 512
                    data = r.read(padded) if padded else b""
                    if len(data) < padded:
                        raise IOError("short read")
                    if typ in (b"x", b"L"):  # pax / GNU long name for the next member
                        text = data[:size]
                        if typ == b"L":
                            pending_path = text.split(b"\0")[0].decode()
                        else:
                            for rec in text.decode(errors="replace").split("\n"):
                                if " path=" in rec:
                                    pending_path = rec.split(" path=", 1)[1]
                        off += 512 + padded
                        continue
                    if pending_path:
                        path, pending_path = pending_path, None
                    lg = log_of(path)
                    if lg != target:
                        if seen:
                            return n_new, n_key, n_have
                        off += 512 + padded
                        continue
                    seen = True
                    off += 512 + padded
                    if typ not in (b"0", b"\0") or not path.lower().endswith(".jpg"):
                        continue
                    cam, fname = path.strip("/").split("/")[-2:]
                    if not keep_keyframes and (keys / cam / fname).exists():
                        n_key += 1
                        continue
                    dest = sweeps / cam / fname
                    if dest.exists():
                        n_have += 1
                    else:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        tmp = dest.with_suffix(".jpg.part")
                        tmp.write_bytes(data[:size])
                        tmp.rename(dest)
                        n_new += 1
                    if limit and n_new >= limit:
                        return n_new, n_key, n_have
        except (urllib.error.URLError, OSError) as e:
            log(f"  {target}: {type(e).__name__} {e}; resuming at byte {off} (attempt {attempt + 1})")
            time.sleep(min(60, 2 ** attempt))
    raise IOError(f"{target}: giving up at byte {off}")


# ---------------------------------------------------------------------------------------------------- check
def check(logs, out):
    summary = {}
    for lg in logs:
        db = out / "nuplan_db" / f"{lg}.db"
        row = {"db": db.exists()}
        if db.exists():
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            names = [r[0] for r in con.execute("SELECT filename_jpg FROM image")]
            con.close()
            exp = {}
            for n in names:
                cam, fn = n.split("/")[-2:]
                exp.setdefault(cam, set()).add(fn)
            for cam in CAMS:
                e = exp.get(cam, set())
                sw = out / "sensor_blobs" / "sweeps" / lg / cam
                kf = out / "sensor_blobs" / "trainval" / lg / cam
                have = {p.name for d in (sw, kf) if d.is_dir() for p in d.iterdir() if p.suffix == ".jpg"}
                row[cam] = {"db_images": len(e), "on_disk": len(have & e), "missing": len(e - have)}
        summary[lg] = row
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--logs", required=True, help="text file, one OpenScene / nuPlan log name per line")
    ap.add_argument("--out", default="~/data/openscene", help="OpenScene root (default ~/data/openscene)")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=list(CAMERA_PARTS),
                    help="nuPlan camera archive sets to search (default train val)")
    ap.add_argument("--workers", type=int, default=8, help="parallel log streams / db downloads (default 8)")
    ap.add_argument("--probe-workers", type=int, default=16)
    ap.add_argument("--probe-step-mb", type=int, default=256,
                    help="probe spacing; a log smaller than this can be missed (a 10 Hz log is > 400 MB)")
    ap.add_argument("--probe-span-kb", type=int, default=256, help="bytes read per probe (images are ~200 KB)")
    ap.add_argument("--keep-keyframes", action="store_true", help="also write frames already in trainval/")
    ap.add_argument("--dry-run", action="store_true", help="databases + probing only; print the plan")
    ap.add_argument("--limit-files", type=int, default=0, help="debug: stop each log after N new files")
    a = ap.parse_args()

    out = Path(os.path.expanduser(a.out))
    logs = [l.strip() for l in open(a.logs) if l.strip() and not l.startswith("#")]
    state = out / "nuplan_fetch"
    state.mkdir(parents=True, exist_ok=True)
    log(f"{len(logs)} logs -> {out}")

    missing_db = fetch_dbs(logs, out, a.workers)

    done_dir = state / "done"
    done_dir.mkdir(exist_ok=True)
    todo = {l for l in logs if not (done_dir / l).exists()}
    plan, found = [], set()
    for split in a.splits:
        for i in range(CAMERA_PARTS[split]):
            if not todo - found:
                break
            arch = f"nuplan-v1.1_{split}_camera_{i}.zip"
            url = f"{BASE}/sensor_blobs/{split}_set/{arch}"
            idx = probe_archive(url, content_length(url), a.probe_step_mb << 20, a.probe_span_kb << 10,
                                a.probe_workers, state / f"{arch}.probes.json")
            runs = runs_of(idx["probes"], todo)
            for r in runs:
                plan.append((url, arch, idx["size"]) + r)
                found.add(r[0])
            log(f"probed {arch} ({idx['size'] / 1e9:.0f} GB, {len(idx['probes'])} probes): "
                f"{len(runs)} wanted runs, {len(todo - found)} logs still unlocated")
    unlocated = sorted(todo - found)
    est = sum(r[6] for r in plan)
    log(f"plan: {len(plan)} runs over {len({p[1] for p in plan})} archives, ~{est / 1e9:.0f} GB to stream; "
        f"{len(unlocated)} logs not located")
    (state / "plan.json").write_text(json.dumps({"runs": plan, "unlocated": unlocated}, indent=1))
    if a.dry_run:
        return

    def work(p):
        url, arch, size, lg, lo, hi, _ = p
        start = find_start(url, lg, lo, hi, a.probe_span_kb << 10, size)
        n = stream_log(url, lg, start, out, a.keep_keyframes, a.limit_files)
        log(f"{lg} ({arch}): {n[0]} new, {n[1]} keyframes skipped, {n[2]} already there")
        return lg

    by_log = {}
    for p in plan:
        by_log.setdefault(p[3], []).append(p)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(lambda ps: [work(p) for p in ps], ps): lg for lg, ps in by_log.items()}
        for fut in cf.as_completed(futs):
            lg = futs[fut]
            try:
                fut.result()
                if not a.limit_files:
                    (done_dir / lg).touch()
            except Exception as e:  # keep going; a rerun retries this log
                log(f"FAILED {lg}: {type(e).__name__} {e}")

    summary = check(logs, out)
    (state / "summary.json").write_text(json.dumps(summary, indent=1))
    incomplete = [lg for lg, r in summary.items() if not r["db"] or any(r[c]["missing"] for c in CAMS if c in r)]
    log(f"check: {len(logs) - len(incomplete)} of {len(logs)} logs complete against their database "
        f"(details: {state / 'summary.json'}); unlocated: {len(unlocated)}; no database: {len(missing_db)}")
    sys.exit(1 if incomplete else 0)


if __name__ == "__main__":
    main()
