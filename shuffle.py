#!/usr/bin/env python3
"""Shuffle NPZ training data for KataGo nano training.

Simplified version of train/shuffle.py: no windowing/tapering, just full shuffle
of all data using the Shardify + Sequential-Merge-Repack two-stage approach.

Output guarantees:
  - Every output file has exactly --rows-per-file rows, except train's last file.
  - Val: all files exactly --rows-per-file rows. Remainder (< rows_per_file) moves
    to train (not discarded). Val is processed first.
  - Train: all files exactly --rows-per-file rows, except the last file which holds
    whatever is left (including val's remainder).
  - Set --rows-per-file to a power of 2 so it's divisible by batch_size * world_size.

Usage:
    python3 shuffle.py <input-dirs...> \\
        --num-processes 8 \\
        [--rows-per-file 131072] \\
        --split "train:0.00:0.95:/out/train:/tmp/train" \\
        --split "val:0.95:1.00:/out/val:/tmp/val"
"""

import argparse
import heapq
import hashlib
import itertools
import json
import multiprocessing
import os
import random
import shutil
import sqlite3
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass

import numpy as np

REQUIRED_KEYS = [
    "binaryInputNCHWPacked",
    "globalInputNC",
    "policyTargetsNCMove",
    "globalTargetsNC",
    "scoreDistrN",
    "valueTargetsNCHW",
]
OPTIONAL_KEYS = []

POS_LEN = 19
PACKED_BYTES = (POS_LEN * POS_LEN + 7) // 8  # 46
SCAN_CACHE_KEY_NONE = -1
SCAN_CACHE_PROCESS_CHUNK_SIZE = 8192
SCAN_CACHE_QUERY_BATCH_SIZE = 512
MANIFEST_SEPARATOR = "\0"
SHARD_CACHE_VERSION = 1


def format_duration(seconds):
    """Format seconds as a compact human-readable duration."""
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def format_wall_time(ts=None):
    """Format a wall-clock timestamp for log lines."""
    if ts is None:
        ts = time.time()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def log(message, *, file=sys.stdout):
    """Print one timestamped log line."""
    print(f"[{format_wall_time()}] {message}", file=file, flush=True)


def atomic_save_json(path, data):
    """Write JSON atomically to avoid leaving partial state on crashes."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def atomic_save_npz(path, arrays, compressed=False):
    """Write an NPZ atomically."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    save_fn = np.savez_compressed if compressed else np.savez
    with open(tmp_path, "wb") as handle:
        save_fn(handle, **arrays)
    os.replace(tmp_path, path)


class ProgressLogger:
    """Periodic progress logger with throughput and ETA."""

    def __init__(self, desc, total, unit, interval_sec=30.0):
        self.desc = desc
        self.total = max(0, int(total))
        self.unit = unit
        self.interval_sec = max(1.0, float(interval_sec))
        self.t0 = time.time()
        self.last_report_time = self.t0
        self.last_completed = 0

    def maybe_report(self, completed, extra=""):
        now = time.time()
        if completed < self.total and now - self.last_report_time < self.interval_sec:
            return
        self._report(completed, now, extra)

    def final_report(self, completed, extra=""):
        now = time.time()
        if completed == self.last_completed and now - self.last_report_time < 1.0:
            return
        self._report(completed, now, extra)

    def _report(self, completed, now, extra):
        elapsed = max(1e-9, now - self.t0)
        avg_rate = completed / elapsed
        interval_elapsed = max(1e-9, now - self.last_report_time)
        recent_rate = (completed - self.last_completed) / interval_elapsed
        if self.total > 0:
            pct = 100.0 * completed / self.total
            if completed > 0 and completed < self.total:
                eta = (self.total - completed) / max(1e-9, recent_rate)
                eta_text = f", ETA {format_duration(eta)}"
            else:
                eta_text = ""
            total_text = f"/{self.total}"
            pct_text = f" ({pct:.1f}%)"
        else:
            total_text = ""
            pct_text = ""
            eta_text = ""

        rate_text = ""
        if avg_rate > 0:
            rate_text = f", avg {avg_rate:.2f} {self.unit}/s"
        if recent_rate > 0:
            rate_text += f", recent {recent_rate:.2f} {self.unit}/s"
        extra_text = f", {extra}" if extra else ""
        log(f"  Progress [{self.desc}]: {completed}{total_text} {self.unit}{pct_text}"
            f"{rate_text}{eta_text}{extra_text}")
        self.last_report_time = now
        self.last_completed = completed


def write_manifest_entry(handle, filename, num_rows):
    """Write one (filename, num_rows) entry to a manifest file."""
    handle.write(f"{filename}{MANIFEST_SEPARATOR}{num_rows}\n")


def iter_manifest_entries(manifest_path):
    """Yield (filename, num_rows) pairs from a manifest file."""
    with open(manifest_path, "r", encoding="utf-8", newline="") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            filename, num_rows_str = line.rsplit(MANIFEST_SEPARATOR, 1)
            yield filename, int(num_rows_str)


class FixedValSelector:
    """Streaming selector for a random val prefix by cumulative row count."""

    def __init__(self, target_rows):
        self.target_rows = max(0, int(target_rows))
        self.total_rows = 0
        self._heap = []  # max-heap by random key via negated key

    def add(self, filename, num_rows):
        """Consider one file for the selected val set."""
        if self.target_rows <= 0:
            return

        # Use a stable per-file hash so fixed val selection is reproducible
        # across reruns of the same dataset and can participate in caching.
        rand_key = int.from_bytes(
            hashlib.blake2b(filename.encode("utf-8"), digest_size=8).digest(),
            byteorder="big",
            signed=False,
        )
        heapq.heappush(self._heap, (-rand_key, num_rows, filename))
        self.total_rows += num_rows

        while self._heap and self.total_rows - self._heap[0][1] >= self.target_rows:
            _, popped_rows, _ = heapq.heappop(self._heap)
            self.total_rows -= popped_rows

    def selected_paths(self):
        """Return the selected file paths as a set."""
        return {filename for _, _, filename in self._heap}


def board_size_cache_key(board_size):
    """Encode board_size for cache storage."""
    return SCAN_CACHE_KEY_NONE if board_size is None else int(board_size)


def open_scan_cache(path):
    """Open or create the SQLite scan cache."""
    cache_dir = os.path.dirname(path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scan_cache (
            board_size INTEGER NOT NULL,
            path TEXT NOT NULL,
            num_rows INTEGER NOT NULL,
            ok INTEGER NOT NULL,
            PRIMARY KEY (board_size, path)
        )
        """
    )
    # Older versions used 0 to represent board_size=None. Migrate in place so
    # existing caches keep working after switching the sentinel to -1.
    conn.execute(
        "UPDATE scan_cache SET board_size = ? WHERE board_size = 0",
        (SCAN_CACHE_KEY_NONE,),
    )
    conn.commit()
    return conn


def fetch_scan_cache_entries(conn, board_size, paths):
    """Fetch cached scan results for a list of file paths."""
    if not paths:
        return {}

    cache_key = board_size_cache_key(board_size)
    cached = {}
    for i in range(0, len(paths), SCAN_CACHE_QUERY_BATCH_SIZE):
        batch = paths[i:i + SCAN_CACHE_QUERY_BATCH_SIZE]
        placeholders = ",".join("?" for _ in batch)
        query = (
            f"SELECT path, num_rows, ok FROM scan_cache "
            f"WHERE board_size = ? AND path IN ({placeholders})"
        )
        rows = conn.execute(query, [cache_key, *batch]).fetchall()
        for path, num_rows, ok in rows:
            num_rows = None if num_rows == -1 else num_rows
            cached[path] = (num_rows, bool(ok))
    return cached


def store_scan_cache_entries(conn, board_size, scan_results):
    """Store scan results into the cache."""
    if not scan_results:
        return

    cache_key = board_size_cache_key(board_size)
    rows = []
    for filename, num_rows, ok in scan_results:
        stored_rows = -1 if num_rows is None else int(num_rows)
        rows.append((cache_key, filename, stored_rows, 1 if ok else 0))

    conn.executemany(
        """
        INSERT INTO scan_cache (board_size, path, num_rows, ok)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(board_size, path) DO UPDATE SET
            num_rows = excluded.num_rows,
            ok = excluded.ok
        """,
        rows,
    )
    conn.commit()


def apply_scan_result(filename, num_rows, ok, total_rows, bad_files, filtered_by_board):
    """Accumulate one scan result into summary counters."""
    if num_rows is None or num_rows <= 0:
        bad_files += 1
    elif not ok:
        filtered_by_board += 1
    else:
        total_rows += num_rows

    return total_rows, bad_files, filtered_by_board


def choose_scan_tmp_root(splits):
    """Choose a temp root directory for scan manifests."""
    if not splits:
        return tempfile.gettempdir()

    tmp_dirs = [os.path.abspath(split.tmp_dir) for split in splits]
    try:
        common = os.path.commonpath(tmp_dirs)
    except ValueError:
        common = os.path.dirname(tmp_dirs[0])
    if os.path.isdir(common):
        return common
    return os.path.dirname(common)


@dataclass
class SplitConfig:
    name: str
    md5_lbound: float
    md5_ubound: float
    out_dir: str
    tmp_dir: str
    file_rows: list | None = None
    manifest_path: str | None = None
    num_files: int = 0
    total_rows: int = 0


def split_cache_dir(split):
    """Return the persistent cache/checkpoint directory for one split."""
    return os.path.join(split.out_dir, ".shuffle_cache")


def split_cache_state_path(split):
    """Return the JSON checkpoint path for one split."""
    return os.path.join(split_cache_dir(split), "state.json")


def split_cache_remainder_dir(split):
    """Return the directory storing committed bucket remainders."""
    return os.path.join(split_cache_dir(split), "remainders")


def split_cache_final_remainder_path(split):
    """Return the persisted final remainder path for keep_remainder splits."""
    return os.path.join(split_cache_dir(split), "final_remainder.npz")


def build_shard_cache_config(split, num_files, total_rows, num_buckets, rows_per_file,
                             worker_group_size, shard_chunk_size, compress_shards,
                             keep_remainder, max_active_worker_groups):
    """Build a stable config payload for validating cached shard checkpoints."""
    return {
        "version": SHARD_CACHE_VERSION,
        "split_name": split.name,
        "num_files": int(num_files),
        "total_rows": int(total_rows),
        "num_buckets": int(num_buckets),
        "rows_per_file": int(rows_per_file),
        "worker_group_size": int(worker_group_size),
        "shard_chunk_size": int(shard_chunk_size),
        "compress_shards": bool(compress_shards),
        "keep_remainder": bool(keep_remainder),
        "max_active_worker_groups": (
            None if max_active_worker_groups is None else int(max_active_worker_groups)
        ),
    }


def load_shard_cache_state(split, expected_config):
    """Load and validate a persisted shard cache checkpoint if present."""
    state_path = split_cache_state_path(split)
    if not os.path.exists(state_path):
        return None

    with open(state_path, "r", encoding="utf-8") as handle:
        state = json.load(handle)

    if state.get("config") != expected_config:
        raise ValueError(
            f"Shard cache config mismatch for split '{split.name}'. "
            f"Delete {split_cache_dir(split)} or use a new output directory."
        )
    return state


def save_shard_cache_state(split, state):
    """Persist shard cache state atomically."""
    atomic_save_json(split_cache_state_path(split), state)


def cleanup_stale_wave_stage_dirs(split):
    """Delete any uncommitted staging dirs left by an interrupted previous run."""
    cache_dir = split_cache_dir(split)
    if not os.path.isdir(cache_dir):
        return

    with os.scandir(cache_dir) as entries:
        for entry in entries:
            if entry.is_dir() and entry.name.startswith("wave_stage_"):
                shutil.rmtree(entry.path, ignore_errors=True)


def collect_bucket_part_files(out_dir, num_buckets):
    """Collect committed bucket temp outputs sorted by bucket-local part index."""
    bucket_files = [[] for _ in range(num_buckets)]
    if not os.path.isdir(out_dir):
        return bucket_files

    with os.scandir(out_dir) as entries:
        for entry in entries:
            name = entry.name
            if not entry.is_file() or not name.endswith(".npz") or not name.startswith("bucket"):
                continue
            try:
                bucket_part, _suffix = name[:-4], ".npz"
                bucket_text, part_text = bucket_part.split(".part", 1)
                bucket_idx = int(bucket_text[len("bucket"):])
                part_idx = int(part_text)
            except (ValueError, IndexError):
                continue
            if 0 <= bucket_idx < num_buckets:
                bucket_files[bucket_idx].append((part_idx, entry.path))

    return [
        [path for _part_idx, path in sorted(files)]
        for files in bucket_files
    ]


def scan_file(args):
    """Get row count + optional board size check in a single pass.

    Args:
        args: (filename, board_size) where board_size may be None.

    Returns:
        (filename, num_rows, ok) where ok is True if the file passes the
        board size check (or if no check is requested).
    """
    filename, board_size = args
    try:
        npheaders = get_numpy_npz_headers(filename)
    except (PermissionError, zipfile.BadZipFile) as e:
        log(f"WARNING: {e}: {filename}")
        return (filename, None, False)
    if npheaders is None or len(npheaders) == 0:
        return (filename, None, False)

    num_rows = None
    for key in ["binaryInputNCHWPacked", "binaryInputNCHWPacked.npy"]:
        if key in npheaders:
            num_rows = npheaders[key][0][0]
            break
    if num_rows is None:
        return (filename, None, False)

    if board_size is None:
        return (filename, num_rows, True)

    # Board size check
    try:
        with np.load(filename) as npz:
            packed = npz["binaryInputNCHWPacked"]
            if packed.shape[2] != PACKED_BYTES:
                return (filename, num_rows, False)
            ch0 = np.unpackbits(packed[0:1, 0:1, :], axis=2)
            ch0 = ch0[0, 0, :POS_LEN * POS_LEN].reshape(POS_LEN, POS_LEN)
            ok = (int(ch0[:board_size, :board_size].sum()) == board_size * board_size
                  and int(ch0.sum()) == board_size * board_size)
            return (filename, num_rows, ok)
    except Exception:
        return (filename, num_rows, False)


def get_numpy_npz_headers(filename):
    """Read NPZ headers without loading array data."""
    with zipfile.ZipFile(filename) as z:
        npzheaders = {}
        for subfilename in z.namelist():
            npyfile = z.open(subfilename)
            try:
                version = np.lib.format.read_magic(npyfile)
            except ValueError:
                log(f"WARNING: bad array in {filename}: {subfilename}")
                return None
            (shape, is_fortran, dtype) = np.lib.format._read_array_header(npyfile, version)
            npzheaders[subfilename] = (shape, is_fortran, dtype)
        return npzheaders


def get_header_entry(npheaders, key):
    """Get NPZ header entry, allowing for the '.npy' suffix inside zip files."""
    if key in npheaders:
        return npheaders[key]
    key_npy = f"{key}.npy"
    if key_npy in npheaders:
        return npheaders[key_npy]
    return None


def estimate_required_bytes_per_row(filename):
    """Estimate bytes/row for the arrays this script actually reads."""
    try:
        npheaders = get_numpy_npz_headers(filename)
    except (PermissionError, zipfile.BadZipFile):
        return None

    if not npheaders:
        return None

    packed_header = get_header_entry(npheaders, "binaryInputNCHWPacked")
    if packed_header is None:
        return None

    num_rows = packed_header[0][0]
    if num_rows <= 0:
        return None

    total_bytes = 0
    for key in REQUIRED_KEYS + OPTIONAL_KEYS:
        entry = get_header_entry(npheaders, key)
        if entry is None:
            if key in REQUIRED_KEYS:
                return None
            continue
        shape, _, dtype = entry
        itemsize = np.dtype(dtype).itemsize
        num_items = 1
        for dim in shape:
            num_items *= dim
        total_bytes += itemsize * num_items

    return total_bytes / num_rows


def format_bytes(num_bytes):
    """Format bytes as a human-readable binary size."""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0


def get_total_memory_bytes():
    """Best-effort total system RAM detection."""
    page_names = ("SC_PAGE_SIZE", "SC_PAGESIZE")
    for page_name in page_names:
        try:
            page_size = os.sysconf(page_name)
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            return int(page_size) * int(phys_pages)
        except (AttributeError, OSError, ValueError):
            continue
    return None


def log_memory_estimates(sample_file, worker_group_size, rows_per_file,
                         shard_processes, merge_processes):
    """Print rough memory estimates for current shuffle settings."""
    bytes_per_row = estimate_required_bytes_per_row(sample_file)
    if bytes_per_row is None:
        log("Memory estimate: unavailable (could not read sample NPZ headers)")
        return

    shard_raw = bytes_per_row * worker_group_size
    merge_raw = bytes_per_row * rows_per_file

    # Rough peak multipliers from the current algorithm:
    # shardify ~= loaded file arrays + concatenated arrays + shuffled copy
    # merge ~= buffered arrays + concatenated chunk + shuffled output chunk
    shard_peak_per_worker = shard_raw * 3.25
    merge_peak_per_worker = merge_raw * 4.0
    shard_peak_total = shard_peak_per_worker * shard_processes
    merge_peak_total = merge_peak_per_worker * merge_processes

    log("Memory estimate (rough, for required arrays only):")
    log(f"  Sample file: {sample_file}")
    log(f"  Bytes per row: {bytes_per_row:.0f} ({format_bytes(bytes_per_row)})")
    log(f"  Shardify per worker: ~{format_bytes(shard_peak_per_worker)} "
        f"(worker_group_size={worker_group_size})")
    log(f"  Shardify total: ~{format_bytes(shard_peak_total)} "
        f"({shard_processes} workers)")
    log(f"  Merge per worker: ~{format_bytes(merge_peak_per_worker)} "
        f"(rows_per_file={rows_per_file})")
    log(f"  Merge total: ~{format_bytes(merge_peak_total)} "
        f"({merge_processes} workers)")

    total_mem = get_total_memory_bytes()
    if total_mem is not None:
        log(f"  Host RAM: {format_bytes(total_mem)}")
        if shard_peak_total > total_mem * 0.70:
            log("WARNING: shardify memory estimate exceeds 70% of host RAM. "
                "Reduce --worker-group-size and/or --shard-processes.")
        if merge_peak_total > total_mem * 0.70:
            log("WARNING: merge memory estimate exceeds 70% of host RAM. "
                "Reduce --merge-processes and/or --rows-per-file.")


def md5_hash_float(s):
    """Hash a string to a float in [0, 1) using MD5."""
    return int("0x" + hashlib.md5(s.encode("utf-8")).hexdigest()[:13], 16) / 2**52


def load_npz_arrays(filename):
    """Load all relevant arrays from an NPZ file. Returns dict or None."""
    try:
        with np.load(filename) as npz:
            data = {}
            for key in REQUIRED_KEYS:
                if key not in npz:
                    log(f"WARNING: missing key {key} in {filename}")
                    return None
                data[key] = npz[key]
            for key in OPTIONAL_KEYS:
                if key in npz:
                    data[key] = npz[key]
            return data
    except Exception as e:
        log(f"WARNING: error loading {filename}: {e}")
        return None


def joint_shuffle(arrs, n=None):
    """Jointly shuffle a list of arrays along axis 0, optionally taking first n."""
    total = len(arrs[0])
    perm = np.random.permutation(total)
    if n is not None:
        perm = perm[:n]
    return [arr[perm] for arr in arrs]


def assign_chunk_ranges_to_buckets(num_rows, num_out_files, shard_chunk_size):
    """Assign contiguous row ranges to shard buckets and return per-bucket ranges."""
    num_chunks = (num_rows + shard_chunk_size - 1) // shard_chunk_size
    chunk_assignments = np.random.randint(num_out_files, size=num_chunks)
    bucket_ranges = [[] for _ in range(num_out_files)]
    counts = np.zeros(num_out_files, dtype=np.int64)

    for chunk_idx, bucket_idx in enumerate(chunk_assignments):
        start = chunk_idx * shard_chunk_size
        end = min(start + shard_chunk_size, num_rows)
        bucket_idx = int(bucket_idx)
        bucket_ranges[bucket_idx].append((start, end))
        counts[bucket_idx] += end - start

    return bucket_ranges, counts


def shardify(input_idx, file_group, num_out_files, out_tmp_dirs,
             compress_shards=False, shard_chunk_size=16384):
    """Load a group of files, shuffle, and distribute rows to shard temp dirs."""
    np.random.seed([int.from_bytes(os.urandom(4), byteorder="little") for _ in range(4)])

    # Load and concatenate all files in the group
    all_data = {}
    for fpath in file_group:
        data = load_npz_arrays(fpath)
        if data is None:
            continue
        for key, arr in data.items():
            if key not in all_data:
                all_data[key] = []
            all_data[key].append(arr)

    if not all_data or "binaryInputNCHWPacked" not in all_data:
        return np.zeros(num_out_files, dtype=np.int64)

    merged = {key: np.concatenate(arrs) for key, arrs in all_data.items()}
    num_rows = merged["binaryInputNCHWPacked"].shape[0]

    # Shuffle
    keys = list(merged.keys())
    arrays = [merged[k] for k in keys]
    arrays = joint_shuffle(arrays)
    merged = dict(zip(keys, arrays))

    # Distribute to shards by chunk: assign each contiguous chunk of shuffled rows
    # to one bucket, then concatenate those chunk slices per bucket. This avoids
    # building a full per-row assignment array and argsorting it for every group.
    bucket_ranges, counts = assign_chunk_ranges_to_buckets(
        num_rows, num_out_files, shard_chunk_size
    )

    save_fn = np.savez_compressed if compress_shards else np.savez
    for out_idx in range(num_out_files):
        ranges = bucket_ranges[out_idx]
        if not ranges:
            continue
        if len(ranges) == 1:
            start, end = ranges[0]
            shard = {key: merged[key][start:end] for key in keys}
        else:
            shard = {
                key: np.concatenate([merged[key][start:end] for start, end in ranges])
                for key in keys
            }
        out_path = os.path.join(out_tmp_dirs[out_idx], f"{input_idx}.npz")
        save_fn(out_path, **shard)

    return counts


def shardify_star(args):
    """Helper for imap_unordered with shardify."""
    return shardify(*args)


def sequential_merge_repack(num_shards, out_tmp_dirs, out_dir, rows_per_file,
                            keep_remainder=False, extra_rows=None):
    """Sequentially read all shards, buffer rows, and write exact-sized output files.

    Args:
        num_shards: Number of sharding worker groups (= shard files per tmp dir).
        out_tmp_dirs: List of shard temp directories.
        out_dir: Final output directory.
        rows_per_file: Exact number of rows per output file.
        keep_remainder: If True, don't write the last partial chunk; return it instead.
        extra_rows: Optional dict of arrays to inject into buffer as initial data.

    Returns:
        (written_files, remainder)
        - written_files: list of (filepath, num_rows)
        - remainder: dict of arrays (< rows_per_file) or None
    """
    np.random.seed([int.from_bytes(os.urandom(4), byteorder="little") for _ in range(5)])

    buffer = {}  # key -> list of arrays
    buffer_rows = 0
    out_file_idx = 0
    written_files = []

    def flush_buffer(n_rows):
        nonlocal buffer, buffer_rows, out_file_idx

        merged = {key: np.concatenate(arrs) for key, arrs in buffer.items()}
        keys = list(merged.keys())

        take = {key: merged[key][:n_rows] for key in keys}
        remain = {key: merged[key][n_rows:] for key in keys}

        # Shuffle the chunk before writing
        arrays = joint_shuffle([take[k] for k in keys])
        take = dict(zip(keys, arrays))

        out_path = os.path.join(out_dir, f"data{out_file_idx}.npz")
        np.savez_compressed(out_path, **take)
        written_files.append((out_path, n_rows))
        out_file_idx += 1

        remain_rows = remain[keys[0]].shape[0]
        if remain_rows > 0:
            buffer = {key: [arr] for key, arr in remain.items()}
        else:
            buffer = {}
        buffer_rows = remain_rows

    # Inject extra rows (e.g. val remainder) into buffer
    if extra_rows is not None:
        n = extra_rows["binaryInputNCHWPacked"].shape[0]
        if n > 0:
            for key, arr in extra_rows.items():
                buffer[key] = [arr]
            buffer_rows = n
            log(f"  Injected {n} extra rows into buffer")

    # Collect all shard file paths and randomize read order
    all_shard_files = []
    for tmp_dir in out_tmp_dirs:
        for shard_idx in range(num_shards):
            shard_path = os.path.join(tmp_dir, f"{shard_idx}.npz")
            if os.path.exists(shard_path):
                all_shard_files.append(shard_path)
    np.random.shuffle(all_shard_files)

    for shard_path in all_shard_files:
        try:
            with np.load(shard_path) as npz:
                for key in npz.keys():
                    if key not in buffer:
                        buffer[key] = []
                    buffer[key].append(npz[key])
                buffer_rows += npz["binaryInputNCHWPacked"].shape[0]
        except Exception as e:
            log(f"WARNING: error reading shard {shard_path}: {e}")
            continue

        # Flush full files as buffer accumulates
        while buffer_rows >= rows_per_file:
            flush_buffer(rows_per_file)

    # Handle final remainder
    remainder = None
    if buffer_rows > 0:
        if keep_remainder:
            remainder = {key: np.concatenate(arrs) for key, arrs in buffer.items()}
            log(f"  Remainder: {buffer_rows} rows -> train")
        else:
            flush_buffer(buffer_rows)

    return written_files, remainder


def merge_one_bucket(bucket_idx, tmp_dir, num_shards, out_dir,
                     out_file_idx, rows_per_file, remainder_path,
                     extra_rows_path=None, progress_interval_sec=30.0):
    """Merge all shard files in one bucket, write output files, save remainder.

    Uses streaming buffer: loads one shard at a time and flushes when the buffer
    reaches rows_per_file, so memory stays O(rows_per_file + single_shard_size)
    instead of O(entire_bucket_size).

    Args:
        bucket_idx: Index of this bucket (for logging).
        tmp_dir: The temp directory for this bucket (e.g., tmp.shuf3/).
        num_shards: Number of shard files to look for (0..num_shards-1).
        out_dir: Final output directory.
        out_file_idx: Starting output file index for this bucket.
        rows_per_file: Exact number of rows per output file.
        remainder_path: Path to save remainder as NPZ (if any leftover rows).
        extra_rows_path: Optional path to NPZ file with extra rows to inject.

    Returns:
        (bucket_idx, written_files, remainder_rows)
        - written_files: list of (filepath, num_rows)
        - remainder_rows: int, number of leftover rows saved to remainder_path
    """
    np.random.seed([int.from_bytes(os.urandom(4), byteorder="little") for _ in range(5)])

    buffer = {}  # key -> list of arrays
    buffer_rows = 0
    file_idx = out_file_idx
    written_files = []

    def flush_buffer(n_rows):
        nonlocal buffer, buffer_rows, file_idx

        merged = {key: np.concatenate(arrs) for key, arrs in buffer.items()}
        keys = list(merged.keys())

        take = {key: merged[key][:n_rows] for key in keys}
        remain = {key: merged[key][n_rows:] for key in keys}

        arrays = joint_shuffle([take[k] for k in keys])
        take = dict(zip(keys, arrays))

        out_path = os.path.join(out_dir, f"data{file_idx}.npz")
        np.savez_compressed(out_path, **take)
        written_files.append((out_path, n_rows))
        file_idx += 1

        remain_rows = remain[keys[0]].shape[0]
        if remain_rows > 0:
            buffer = {key: [arr] for key, arr in remain.items()}
        else:
            buffer = {}
        buffer_rows = remain_rows

    # Inject extra rows (e.g. val remainder) if provided
    if extra_rows_path is not None and os.path.exists(extra_rows_path):
        with np.load(extra_rows_path) as npz:
            for key in npz.keys():
                buffer[key] = [npz[key]]
            buffer_rows = npz["binaryInputNCHWPacked"].shape[0]
            log(f"  Bucket {bucket_idx}: injected {buffer_rows} extra rows")

    # Read all shard files in this bucket (randomized order)
    shard_files = []
    for shard_idx in range(num_shards):
        path = os.path.join(tmp_dir, f"{shard_idx}.npz")
        if os.path.exists(path):
            shard_files.append(path)
    np.random.shuffle(shard_files)
    total_shard_files = len(shard_files)
    last_progress_time = time.time()

    for processed_shards, path in enumerate(shard_files, start=1):
        try:
            with np.load(path) as npz:
                for key in npz.keys():
                    if key not in buffer:
                        buffer[key] = []
                    buffer[key].append(npz[key])
                buffer_rows += npz["binaryInputNCHWPacked"].shape[0]
        except Exception as e:
            log(f"WARNING: error reading shard {path}: {e}")
            continue

        # Flush as buffer accumulates
        while buffer_rows >= rows_per_file:
            flush_buffer(rows_per_file)

        now = time.time()
        if processed_shards < total_shard_files and now - last_progress_time >= progress_interval_sec:
            pct = 100.0 * processed_shards / total_shard_files if total_shard_files > 0 else 100.0
            log(f"  Bucket {bucket_idx}: read {processed_shards}/{total_shard_files} shard files "
                f"({pct:.1f}%), wrote {len(written_files)} files, buffered {buffer_rows} rows")
            last_progress_time = now

    # Save remainder to temp file (avoid pickling large arrays through IPC)
    if buffer_rows > 0:
        remainder = {key: np.concatenate(arrs) for key, arrs in buffer.items()}
        np.savez(remainder_path, **remainder)  # uncompressed, temp file

    log(f"  Bucket {bucket_idx}: finished {total_shard_files}/{total_shard_files} shard files, "
        f"wrote {len(written_files)} files, remainder {buffer_rows} rows")
    return bucket_idx, written_files, buffer_rows


def merge_one_bucket_star(args):
    """Helper for imap_unordered with merge_one_bucket."""
    return merge_one_bucket(*args)


def merge_one_bucket_incremental(bucket_idx, shard_files, parts_dir, out_file_idx,
                                 rows_per_file, remainder_in_path, remainder_out_path,
                                 progress_interval_sec=30.0):
    """Merge one wave of shard files into bucket-local temp outputs.

    Existing bucket remainder (if any) is loaded from remainder_in_path and merged
    with the current wave's shard files. Full output chunks are written to
    bucket-scoped temporary files in parts_dir so later waves can continue
    appending without knowing final global file indices yet.
    """
    np.random.seed([int.from_bytes(os.urandom(4), byteorder="little") for _ in range(5)])

    buffer = {}
    buffer_rows = 0
    file_idx = out_file_idx
    written_files = []

    def flush_buffer(n_rows):
        nonlocal buffer, buffer_rows, file_idx

        merged = {key: np.concatenate(arrs) for key, arrs in buffer.items()}
        keys = list(merged.keys())

        take = {key: merged[key][:n_rows] for key in keys}
        remain = {key: merged[key][n_rows:] for key in keys}

        arrays = joint_shuffle([take[k] for k in keys])
        take = dict(zip(keys, arrays))

        out_path = os.path.join(parts_dir, f"bucket{bucket_idx}.part{file_idx}.npz")
        atomic_save_npz(out_path, take, compressed=True)
        written_files.append((out_path, n_rows))
        file_idx += 1

        remain_rows = remain[keys[0]].shape[0]
        if remain_rows > 0:
            buffer = {key: [arr] for key, arr in remain.items()}
        else:
            buffer = {}
        buffer_rows = remain_rows

    if os.path.exists(remainder_in_path):
        try:
            with np.load(remainder_in_path) as npz:
                for key in npz.keys():
                    buffer[key] = [npz[key]]
                buffer_rows = npz["binaryInputNCHWPacked"].shape[0]
        except Exception as e:
            log(f"WARNING: error reading bucket remainder {remainder_in_path}: {e}")
            buffer = {}
            buffer_rows = 0

    shard_files = list(shard_files)
    np.random.shuffle(shard_files)
    total_shard_files = len(shard_files)
    last_progress_time = time.time()

    for processed_shards, path in enumerate(shard_files, start=1):
        try:
            with np.load(path) as npz:
                for key in npz.keys():
                    if key not in buffer:
                        buffer[key] = []
                    buffer[key].append(npz[key])
                buffer_rows += npz["binaryInputNCHWPacked"].shape[0]
        except Exception as e:
            log(f"WARNING: error reading shard {path}: {e}")
            continue

        while buffer_rows >= rows_per_file:
            flush_buffer(rows_per_file)

        now = time.time()
        if processed_shards < total_shard_files and now - last_progress_time >= progress_interval_sec:
            pct = 100.0 * processed_shards / total_shard_files if total_shard_files > 0 else 100.0
            log(f"  Bucket {bucket_idx}: read {processed_shards}/{total_shard_files} shard files "
                f"({pct:.1f}%), wrote {len(written_files)} files, buffered {buffer_rows} rows")
            last_progress_time = now

    if buffer_rows > 0:
        remainder = {key: np.concatenate(arrs) for key, arrs in buffer.items()}
        atomic_save_npz(remainder_out_path, remainder, compressed=False)
    elif os.path.exists(remainder_out_path):
        os.remove(remainder_out_path)

    log(f"  Bucket {bucket_idx}: finished {total_shard_files}/{total_shard_files} shard files, "
        f"wrote {len(written_files)} files, remainder {buffer_rows} rows")
    return bucket_idx, written_files, buffer_rows


def merge_one_bucket_incremental_star(args):
    """Helper for imap_unordered with merge_one_bucket_incremental."""
    return merge_one_bucket_incremental(*args)


def parallel_merge_repack(num_shards, out_tmp_dirs, out_dir, rows_per_file,
                          num_processes, keep_remainder=False,
                          extra_rows=None, tmp_base_dir=None,
                          bucket_row_counts=None, progress_interval_sec=30.0):
    """Parallel merge: each bucket processed independently by a worker.

    Args:
        num_shards: Number of shard files per bucket (= num worker groups).
        out_tmp_dirs: List of bucket temp directories.
        out_dir: Final output directory.
        rows_per_file: Exact number of rows per output file.
        num_processes: Number of parallel workers.
        keep_remainder: If True, don't write the last partial chunk; return it.
        extra_rows: Optional dict of arrays to inject (e.g. val remainder for train).
        tmp_base_dir: Temp directory for remainder files and extra_rows file.
        bucket_row_counts: Pre-computed list of row counts per bucket (from shardify).

    Returns:
        (written_files, remainder) -- same contract as sequential_merge_repack
    """
    num_buckets = len(out_tmp_dirs)

    # Use pre-computed bucket row counts from shardify (skip header scanning)
    if bucket_row_counts is None:
        bucket_row_counts = []
        for tmp_dir in out_tmp_dirs:
            total = 0
            for shard_idx in range(num_shards):
                path = os.path.join(tmp_dir, f"{shard_idx}.npz")
                if os.path.exists(path):
                    try:
                        headers = get_numpy_npz_headers(path)
                        if headers:
                            for key in headers:
                                if key in ("binaryInputNCHWPacked",
                                           "binaryInputNCHWPacked.npy"):
                                    total += headers[key][0][0]
                                    break
                    except Exception:
                        pass
            bucket_row_counts.append(total)

    # Save extra_rows to temp file for IPC
    extra_rows_path = None
    extra_count = 0
    if extra_rows is not None:
        extra_count = extra_rows["binaryInputNCHWPacked"].shape[0]
        if extra_count > 0:
            extra_rows_path = os.path.join(tmp_base_dir, "extra_rows.npz")
            np.savez(extra_rows_path, **extra_rows)

    # --- Phase 2: Pre-allocate output file indices ---
    # Add extra_rows to bucket 0
    effective_counts = list(bucket_row_counts)
    if extra_count > 0:
        effective_counts[0] += extra_count

    out_file_starts = []
    cumulative = 0
    for count in effective_counts:
        out_file_starts.append(cumulative)
        cumulative += count // rows_per_file

    # --- Phase 3: Parallel merge ---
    remainder_dir = os.path.join(tmp_base_dir, "tmp.remainders")
    os.makedirs(remainder_dir, exist_ok=True)

    tasks = []
    for bucket_idx, tmp_dir in enumerate(out_tmp_dirs):
        remainder_path = os.path.join(remainder_dir, f"rem_{bucket_idx}.npz")
        inject_path = extra_rows_path if bucket_idx == 0 else None
        tasks.append((
            bucket_idx, tmp_dir, num_shards, out_dir,
            out_file_starts[bucket_idx], rows_per_file,
            remainder_path, inject_path, progress_interval_sec,
        ))

    with multiprocessing.Pool(num_processes) as pool:
        results = []
        progress = ProgressLogger(
            desc="merge buckets",
            total=len(tasks),
            unit="buckets",
            interval_sec=progress_interval_sec,
        )
        total_written_files = 0
        for completed_buckets, result in enumerate(pool.imap_unordered(merge_one_bucket_star, tasks), start=1):
            bucket_idx, written_files, rem_rows = result
            results.append(result)
            total_written_files += len(written_files)
            bucket_rows = bucket_row_counts[bucket_idx] if bucket_idx < len(bucket_row_counts) else 0
            progress.maybe_report(
                completed_buckets,
                extra=f"bucket {bucket_idx} done, files_written={total_written_files}, "
                      f"bucket_rows={bucket_rows}",
            )

    # --- Phase 4: Collect results and handle remainders ---
    all_written = []
    all_remainder_paths = []
    total_remainder_rows = 0

    for bucket_idx, written_files, rem_rows in results:
        all_written.extend(written_files)
        if rem_rows > 0:
            rem_path = os.path.join(remainder_dir, f"rem_{bucket_idx}.npz")
            all_remainder_paths.append(rem_path)
            total_remainder_rows += rem_rows

    # Determine next file index
    next_file_idx = cumulative  # from the pre-allocation

    # Merge all remainders
    remainder = None
    if total_remainder_rows > 0:
        combined = {}
        for rem_path in all_remainder_paths:
            try:
                with np.load(rem_path) as npz:
                    for key in npz.keys():
                        if key not in combined:
                            combined[key] = []
                        combined[key].append(npz[key])
            except Exception as e:
                log(f"WARNING: error reading remainder {rem_path}: {e}")

        if combined:
            merged_rem = {key: np.concatenate(arrs) for key, arrs in combined.items()}
            keys = list(merged_rem.keys())
            total_rem = merged_rem[keys[0]].shape[0]

            # Produce full files from merged remainders
            offset = 0
            while offset + rows_per_file <= total_rem:
                chunk = {key: merged_rem[key][offset:offset + rows_per_file] for key in keys}
                arrays = joint_shuffle([chunk[k] for k in keys])
                chunk = dict(zip(keys, arrays))
                out_path = os.path.join(out_dir, f"data{next_file_idx}.npz")
                np.savez_compressed(out_path, **chunk)
                all_written.append((out_path, rows_per_file))
                next_file_idx += 1
                offset += rows_per_file

            # Final remainder
            final_rem_rows = total_rem - offset
            if final_rem_rows > 0:
                if keep_remainder:
                    remainder = {key: merged_rem[key][offset:] for key in keys}
                    log(f"  Remainder: {final_rem_rows} rows -> train")
                else:
                    chunk = {key: merged_rem[key][offset:] for key in keys}
                    arrays = joint_shuffle([chunk[k] for k in keys])
                    chunk = dict(zip(keys, arrays))
                    out_path = os.path.join(out_dir, f"data{next_file_idx}.npz")
                    np.savez_compressed(out_path, **chunk)
                    all_written.append((out_path, final_rem_rows))

    # Cleanup temp files
    shutil.rmtree(remainder_dir, ignore_errors=True)
    if extra_rows_path and os.path.exists(extra_rows_path):
        os.remove(extra_rows_path)

    # Sort by file path for consistent ordering
    all_written.sort(key=lambda x: x[0])

    return all_written, remainder


def iter_worker_group_waves(split, worker_group_size, group_shuffle_rng,
                            max_active_worker_groups, skip_groups=0):
    """Yield bounded batches of worker groups for wave-style sharding."""
    group_iter = iter_worker_groups(split, worker_group_size, group_shuffle_rng)
    if skip_groups > 0:
        group_iter = itertools.islice(group_iter, skip_groups, None)
    while True:
        wave = list(itertools.islice(group_iter, max_active_worker_groups))
        if not wave:
            break
        yield wave


def collect_bucket_shard_files(out_tmp_dirs):
    """Collect current-wave shard files from each bucket temp directory."""
    bucket_shard_files = []
    for tmp_dir in out_tmp_dirs:
        shard_files = []
        with os.scandir(tmp_dir) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(".npz"):
                    shard_files.append(entry.path)
        bucket_shard_files.append(shard_files)
    return bucket_shard_files


def reset_tmp_dirs(out_tmp_dirs):
    """Drop all bucket temp shard directories and recreate them empty."""
    for d in out_tmp_dirs:
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)


def run_incremental_merge_wave(merge_pool, split_name, wave_idx, num_waves,
                               bucket_shard_files, parts_dir, rows_per_file,
                               bucket_output_counts, bucket_remainder_paths,
                               stage_remainder_dir,
                               progress_interval_sec=30.0):
    """Merge one wave of per-bucket shard files using a shared worker pool."""
    tasks = []
    active_bucket_indices = []
    for bucket_idx, shard_files in enumerate(bucket_shard_files):
        if not shard_files:
            continue
        active_bucket_indices.append(bucket_idx)
        tasks.append((
            bucket_idx,
            shard_files,
            parts_dir,
            bucket_output_counts[bucket_idx],
            rows_per_file,
            bucket_remainder_paths[bucket_idx],
            os.path.join(stage_remainder_dir, f"rem_{bucket_idx}.npz"),
            progress_interval_sec,
        ))

    wave_written = [[] for _ in range(len(bucket_shard_files))]
    if not tasks:
        return wave_written, active_bucket_indices

    progress = ProgressLogger(
        desc=f"merge {split_name} wave {wave_idx}/{num_waves}",
        total=len(tasks),
        unit="buckets",
        interval_sec=progress_interval_sec,
    )
    total_written_files = 0
    for completed_buckets, result in enumerate(
        merge_pool.imap_unordered(merge_one_bucket_incremental_star, tasks), start=1
    ):
        bucket_idx, written_files, _rem_rows = result
        bucket_output_counts[bucket_idx] += len(written_files)
        wave_written[bucket_idx] = written_files
        total_written_files += len(written_files)
        progress.maybe_report(
            completed_buckets,
            extra=f"bucket {bucket_idx} done, files_written={total_written_files}",
        )

    progress.final_report(len(tasks), extra=f"files_written={total_written_files}")
    return wave_written, active_bucket_indices


def commit_wave_stage(stage_parts_dir, out_dir, active_bucket_indices,
                      stage_remainder_dir, bucket_remainder_paths):
    """Commit one fully merged wave into persistent bucket part files/remainders."""
    if os.path.isdir(stage_parts_dir):
        with os.scandir(stage_parts_dir) as entries:
            for entry in entries:
                if entry.is_file():
                    os.replace(entry.path, os.path.join(out_dir, entry.name))

    for bucket_idx in active_bucket_indices:
        stage_rem_path = os.path.join(stage_remainder_dir, f"rem_{bucket_idx}.npz")
        committed_path = bucket_remainder_paths[bucket_idx]
        if os.path.exists(stage_rem_path):
            os.replace(stage_rem_path, committed_path)
        elif os.path.exists(committed_path):
            os.remove(committed_path)


def finalize_incremental_bucket_outputs(num_buckets, bucket_remainder_paths,
                                        out_dir, rows_per_file, keep_remainder,
                                        progress_interval_sec=30.0,
                                        split_name=None):
    """Finalize bucket-local temp outputs into data*.npz files and merge remainders."""
    written_files = []
    next_file_idx = 0
    finalize_label = split_name if split_name is not None else "split"

    bucket_part_files = collect_bucket_part_files(out_dir, num_buckets)
    total_part_files = sum(len(bucket_files) for bucket_files in bucket_part_files)
    if total_part_files > 0:
        log(f"  Finalize: renaming {total_part_files} bucket part files for '{finalize_label}'")
        rename_progress = ProgressLogger(
            desc=f"finalize rename {finalize_label}",
            total=total_part_files,
            unit="files",
            interval_sec=progress_interval_sec,
        )
    else:
        rename_progress = None

    renamed_files = 0
    for bucket_files in bucket_part_files:
        for temp_path in bucket_files:
            with np.load(temp_path) as npz:
                num_rows = npz["binaryInputNCHWPacked"].shape[0]
            final_path = os.path.join(out_dir, f"data{next_file_idx}.npz")
            os.replace(temp_path, final_path)
            written_files.append((final_path, num_rows))
            next_file_idx += 1
            renamed_files += 1
            if rename_progress is not None:
                rename_progress.maybe_report(
                    renamed_files,
                    extra=f"last=data{next_file_idx - 1}.npz",
                )

    if rename_progress is not None:
        rename_progress.final_report(renamed_files, extra=f"renamed={renamed_files}")

    remainder = None
    all_remainder_paths = [path for path in bucket_remainder_paths if os.path.exists(path)]
    if all_remainder_paths:
        log(f"  Finalize: reading {len(all_remainder_paths)} bucket remainders for '{finalize_label}'")
        remainder_read_progress = ProgressLogger(
            desc=f"finalize read remainders {finalize_label}",
            total=len(all_remainder_paths),
            unit="files",
            interval_sec=progress_interval_sec,
        )
        combined = {}
        for completed_remainders, rem_path in enumerate(all_remainder_paths, start=1):
            try:
                with np.load(rem_path) as npz:
                    for key in npz.keys():
                        if key not in combined:
                            combined[key] = []
                        combined[key].append(npz[key])
            except Exception as e:
                log(f"WARNING: error reading remainder {rem_path}: {e}")
            remainder_read_progress.maybe_report(completed_remainders)

        remainder_read_progress.final_report(len(all_remainder_paths))

        if combined:
            merged_rem = {key: np.concatenate(arrs) for key, arrs in combined.items()}
            keys = list(merged_rem.keys())
            total_rem = merged_rem[keys[0]].shape[0]
            full_remainder_files = total_rem // rows_per_file
            tail_rows = total_rem % rows_per_file
            remainder_outputs = full_remainder_files
            if tail_rows > 0 and not keep_remainder:
                remainder_outputs += 1
            log(
                f"  Finalize: {total_rem} remainder rows across {len(all_remainder_paths)} buckets, "
                f"writing {remainder_outputs} output files"
            )
            if remainder_outputs > 0:
                remainder_write_progress = ProgressLogger(
                    desc=f"finalize write remainders {finalize_label}",
                    total=remainder_outputs,
                    unit="files",
                    interval_sec=progress_interval_sec,
                )
            else:
                remainder_write_progress = None
            written_remainder_files = 0

            offset = 0
            while offset + rows_per_file <= total_rem:
                chunk = {key: merged_rem[key][offset:offset + rows_per_file] for key in keys}
                arrays = joint_shuffle([chunk[k] for k in keys])
                chunk = dict(zip(keys, arrays))
                out_path = os.path.join(out_dir, f"data{next_file_idx}.npz")
                np.savez_compressed(out_path, **chunk)
                written_files.append((out_path, rows_per_file))
                next_file_idx += 1
                offset += rows_per_file
                written_remainder_files += 1
                if remainder_write_progress is not None:
                    remainder_write_progress.maybe_report(
                        written_remainder_files,
                        extra=f"rows={offset}/{total_rem}",
                    )

            final_rem_rows = total_rem - offset
            if final_rem_rows > 0:
                if keep_remainder:
                    remainder = {key: merged_rem[key][offset:] for key in keys}
                    log(f"  Remainder: {final_rem_rows} rows -> train")
                else:
                    chunk = {key: merged_rem[key][offset:] for key in keys}
                    arrays = joint_shuffle([chunk[k] for k in keys])
                    chunk = dict(zip(keys, arrays))
                    out_path = os.path.join(out_dir, f"data{next_file_idx}.npz")
                    np.savez_compressed(out_path, **chunk)
                    written_files.append((out_path, final_rem_rows))
                    written_remainder_files += 1
                    if remainder_write_progress is not None:
                        remainder_write_progress.maybe_report(
                            written_remainder_files,
                            extra=f"rows={total_rem}/{total_rem}",
                        )

            if remainder_write_progress is not None:
                remainder_write_progress.final_report(
                    written_remainder_files,
                    extra=f"rows={total_rem}",
                )

    return written_files, remainder


class Timer:
    def __init__(self, desc):
        self.desc = desc

    def __enter__(self):
        log(f"Beginning: {self.desc}")
        self.t0 = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.t0
        log(f"Finished: {self.desc} in {elapsed:.1f}s")


def iter_split_file_rows(split):
    """Yield (filename, num_rows) pairs for a split."""
    if split.manifest_path is not None:
        yield from iter_manifest_entries(split.manifest_path)
        return

    if split.file_rows is not None:
        yield from split.file_rows


def split_num_files(split):
    """Return the number of input files in a split."""
    if split.manifest_path is not None:
        return split.num_files
    if split.file_rows is not None:
        return len(split.file_rows)
    return 0


def split_total_rows(split):
    """Return the total input rows in a split."""
    if split.manifest_path is not None:
        return split.total_rows
    if split.file_rows is not None:
        return sum(nr for _, nr in split.file_rows)
    return 0


def count_worker_groups(split, worker_group_size):
    """Count how many worker groups a split will produce."""
    groups = 0
    group_rows = 0
    for _, num_rows in iter_split_file_rows(split):
        group_rows += num_rows
        if group_rows >= worker_group_size:
            groups += 1
            group_rows = 0
    if group_rows > 0:
        groups += 1
    return groups


def iter_worker_groups(split, worker_group_size, group_shuffle_rng):
    """Yield lists of filenames grouped by approximate row count."""
    group = []
    group_rows = 0
    for filename, num_rows in iter_split_file_rows(split):
        group.append(filename)
        group_rows += num_rows
        if group_rows >= worker_group_size:
            group_shuffle_rng.shuffle(group)
            yield group
            group = []
            group_rows = 0
    if group:
        group_shuffle_rng.shuffle(group)
        yield group


def append_split_entry(split, handle, filename, num_rows):
    """Append one file entry to a split manifest and update counters."""
    write_manifest_entry(handle, filename, num_rows)
    split.num_files += 1
    split.total_rows += num_rows


def process_split(split, shard_processes, merge_processes, rows_per_file, worker_group_size,
                  num_buckets=None, shard_chunk_size=16384,
                  keep_remainder=False, extra_rows=None, compress_shards=False,
                  progress_interval_sec=30.0, max_active_worker_groups=None,
                  enable_shard_cache=False):
    """Process a single split: shardify + parallel merge-repack + index.json.

    Args:
        split: SplitConfig with file_rows populated.
        shard_processes: Number of parallel workers for shardify.
        merge_processes: Number of parallel workers for merge/repack.
        rows_per_file: Exact rows per output file.
        worker_group_size: Target rows per sharding worker group.
        keep_remainder: If True, return leftover rows instead of writing them.
        extra_rows: Optional dict of arrays to inject (e.g. val remainder for train).
        progress_interval_sec: Seconds between progress updates.
        max_active_worker_groups: Max number of worker groups to keep on local
            temp storage before triggering an intermediate merge. None means
            shard the entire split before merging.
        enable_shard_cache: Persist and reuse wave-mode progress for reruns.

    Returns:
        remainder: dict of arrays (< rows_per_file) or None.
    """
    num_files = split_num_files(split)
    total_rows = split_total_rows(split)
    extra_count = extra_rows["binaryInputNCHWPacked"].shape[0] if extra_rows is not None else 0

    print("", flush=True)
    log(f"{'='*60}")
    log(f"Processing split '{split.name}': {num_files} files, {total_rows} rows")
    if extra_count > 0:
        log(f"  + {extra_count} extra rows from val remainder")
    log(f"  out_dir: {split.out_dir}")
    log(f"  tmp_dir: {split.tmp_dir}")

    if num_files == 0 and extra_count == 0:
        log(f"  No data for split '{split.name}', skipping.")
        return None

    os.makedirs(split.out_dir, exist_ok=enable_shard_cache)

    num_output_files = max(1, round(total_rows / rows_per_file)) if total_rows > 0 else 1
    if num_buckets is not None:
        num_shards_buckets = min(num_output_files, max(1, num_buckets))
    else:
        num_shards_buckets = min(num_output_files, max(1, merge_processes))
    out_tmp_dirs = [os.path.join(split.tmp_dir, f"tmp.shuf{i}") for i in range(num_shards_buckets)]

    log(f"  Intermediate shard buckets: {num_shards_buckets}")
    log(f"  Rows per file: {rows_per_file}")
    log(f"  Shard workers: {shard_processes}")
    log(f"  Merge workers: {merge_processes}")
    if max_active_worker_groups is None:
        log("  Max active worker groups: unlimited")
    else:
        log(f"  Max active worker groups: {max_active_worker_groups}")

    # Clean and create tmp dirs
    reset_tmp_dirs(out_tmp_dirs)

    # Group files for sharding in two streaming passes over the manifest/list:
    # one lightweight pass counts groups for progress and shard indexing, and
    # one pass yields the actual filename lists to workers.
    num_worker_groups = count_worker_groups(split, worker_group_size)
    log(f"  Grouped into {num_worker_groups} worker groups")

    use_wave_mode = (
        max_active_worker_groups is not None
        and num_worker_groups > max_active_worker_groups
    )
    cache_config = build_shard_cache_config(
        split=split,
        num_files=num_files,
        total_rows=total_rows,
        num_buckets=num_shards_buckets,
        rows_per_file=rows_per_file,
        worker_group_size=worker_group_size,
        shard_chunk_size=shard_chunk_size,
        compress_shards=compress_shards,
        keep_remainder=keep_remainder,
        max_active_worker_groups=max_active_worker_groups,
    )
    cache_state = None
    if enable_shard_cache:
        cache_state = load_shard_cache_state(split, cache_config)
        if cache_state is None and os.path.isdir(split.out_dir):
            with os.scandir(split.out_dir) as entries:
                existing_names = [entry.name for entry in entries]
            unexpected = [name for name in existing_names if name != ".shuffle_cache"]
            if unexpected:
                raise ValueError(
                    f"Output directory for split '{split.name}' already contains files but no "
                    f"matching shard cache state was found: {split.out_dir}"
                )
        if cache_state is not None and cache_state.get("status") == "complete":
            log(f"  Shard cache: hit for split '{split.name}', reusing completed outputs")
            final_remainder_path = split_cache_final_remainder_path(split)
            if keep_remainder and os.path.exists(final_remainder_path):
                with np.load(final_remainder_path) as npz:
                    return {key: npz[key] for key in npz.keys()}
            return None

    if use_wave_mode:
        num_waves = (num_worker_groups + max_active_worker_groups - 1) // max_active_worker_groups
        log(f"  Wave mode: enabled ({num_waves} waves)")
        if enable_shard_cache:
            cleanup_stale_wave_stage_dirs(split)
            remainder_dir = split_cache_remainder_dir(split)
            os.makedirs(remainder_dir, exist_ok=True)
        else:
            remainder_dir = os.path.join(split.tmp_dir, "tmp.remainders")
            shutil.rmtree(remainder_dir, ignore_errors=True)
            os.makedirs(remainder_dir, exist_ok=True)

        bucket_remainder_paths = [
            os.path.join(remainder_dir, f"rem_{bucket_idx}.npz")
            for bucket_idx in range(num_shards_buckets)
        ]

        resumed_groups = 0
        total_sharded = 0
        bucket_output_counts = [0 for _ in range(num_shards_buckets)]
        if cache_state is not None:
            if cache_state.get("mode") != "wave":
                raise ValueError(
                    f"Split '{split.name}' has non-wave shard cache state but now requires "
                    f"wave mode. Delete {split_cache_dir(split)} or use a new output directory."
                )
            resumed_groups = int(cache_state.get("completed_groups", 0))
            total_sharded = int(cache_state.get("total_sharded", 0))
            bucket_output_counts = list(cache_state.get("bucket_output_counts", bucket_output_counts))
            log(f"  Shard cache: resuming after {resumed_groups}/{num_worker_groups} groups")

        if extra_rows is not None and extra_count > 0 and resumed_groups == 0:
            atomic_save_npz(bucket_remainder_paths[0], extra_rows, compressed=False)

        remaining_groups = max(0, num_worker_groups - resumed_groups)
        if remaining_groups > 0:
            with Timer(f"Wave sharding+merge ({split.name})"):
                progress = ProgressLogger(
                    desc=f"wave pipeline {split.name}",
                    total=remaining_groups,
                    unit="groups",
                    interval_sec=progress_interval_sec,
                )
                processed_this_run = 0
                group_shuffle_rng = random.Random(int.from_bytes(os.urandom(16), byteorder="little"))
                group_waves = iter_worker_group_waves(
                    split,
                    worker_group_size,
                    group_shuffle_rng,
                    max_active_worker_groups,
                    skip_groups=resumed_groups,
                )
                starting_wave_idx = resumed_groups // max_active_worker_groups + 1
                with multiprocessing.Pool(shard_processes) as shard_pool, \
                        multiprocessing.Pool(merge_processes) as merge_pool:
                    for wave_offset, wave_groups in enumerate(group_waves):
                        wave_idx = starting_wave_idx + wave_offset
                        log(f"  Wave {wave_idx}/{num_waves}: sharding {len(wave_groups)} groups")
                        shard_tasks = (
                            (group_idx, group, num_shards_buckets, out_tmp_dirs,
                             compress_shards, shard_chunk_size)
                            for group_idx, group in enumerate(wave_groups)
                        )
                        for counts in shard_pool.imap_unordered(shardify_star, shard_tasks):
                            total_sharded += int(counts.sum())
                            processed_this_run += 1
                            progress.maybe_report(
                                processed_this_run,
                                extra=(
                                    f"overall={resumed_groups + processed_this_run}/{num_worker_groups}, "
                                    f"rows={total_sharded}, wave={wave_idx}/{num_waves}"
                                ),
                            )

                        bucket_shard_files = collect_bucket_shard_files(out_tmp_dirs)
                        active_buckets = sum(1 for shard_files in bucket_shard_files if shard_files)
                        log(f"  Wave {wave_idx}/{num_waves}: merging {active_buckets} active buckets")

                        if enable_shard_cache:
                            stage_root = os.path.join(split_cache_dir(split), f"wave_stage_{wave_idx}")
                        else:
                            stage_root = os.path.join(split.tmp_dir, f"wave_stage_{wave_idx}")
                        stage_parts_dir = os.path.join(stage_root, "parts")
                        stage_remainder_dir = os.path.join(stage_root, "remainders")
                        shutil.rmtree(stage_root, ignore_errors=True)
                        os.makedirs(stage_parts_dir, exist_ok=True)
                        os.makedirs(stage_remainder_dir, exist_ok=True)

                        _wave_written, active_bucket_indices = run_incremental_merge_wave(
                            merge_pool,
                            split.name,
                            wave_idx,
                            num_waves,
                            bucket_shard_files,
                            stage_parts_dir,
                            rows_per_file,
                            bucket_output_counts,
                            bucket_remainder_paths,
                            stage_remainder_dir,
                            progress_interval_sec=progress_interval_sec,
                        )
                        commit_wave_stage(
                            stage_parts_dir=stage_parts_dir,
                            out_dir=split.out_dir,
                            active_bucket_indices=active_bucket_indices,
                            stage_remainder_dir=stage_remainder_dir,
                            bucket_remainder_paths=bucket_remainder_paths,
                        )
                        shutil.rmtree(stage_root, ignore_errors=True)

                        if enable_shard_cache:
                            save_shard_cache_state(split, {
                                "config": cache_config,
                                "status": "in_progress",
                                "mode": "wave",
                                "completed_groups": resumed_groups + processed_this_run,
                                "total_sharded": total_sharded,
                                "bucket_output_counts": bucket_output_counts,
                            })

                        reset_tmp_dirs(out_tmp_dirs)

                progress.final_report(
                    processed_this_run,
                    extra=f"overall={num_worker_groups}/{num_worker_groups}, rows={total_sharded}",
                )
                log(f"  Sharded {total_sharded} rows")

        with Timer(f"Finalize merged outputs ({split.name})"):
            written_files, remainder = finalize_incremental_bucket_outputs(
                num_buckets=num_shards_buckets,
                bucket_remainder_paths=bucket_remainder_paths,
                out_dir=split.out_dir,
                rows_per_file=rows_per_file,
                keep_remainder=keep_remainder,
                progress_interval_sec=progress_interval_sec,
                split_name=split.name,
            )
        shutil.rmtree(remainder_dir, ignore_errors=True)
    else:
        if num_worker_groups > 0 and max_active_worker_groups is not None:
            log("  Wave mode: skipped (all worker groups fit in one wave)")
        if cache_state is not None and cache_state.get("status") != "complete":
            raise ValueError(
                f"Split '{split.name}' has an incomplete shard cache but this split is not in "
                f"wave mode. Delete {split_cache_dir(split)} or lower --max-active-worker-groups."
            )

        bucket_row_counts = np.zeros(num_shards_buckets, dtype=np.int64)
        if num_worker_groups > 0:
            with multiprocessing.Pool(shard_processes) as pool:
                with Timer(f"Sharding ({split.name})"):
                    progress = ProgressLogger(
                        desc=f"sharding {split.name}",
                        total=num_worker_groups,
                        unit="groups",
                        interval_sec=progress_interval_sec,
                    )
                    total_sharded = 0
                    group_shuffle_rng = random.Random(int.from_bytes(os.urandom(16), byteorder="little"))
                    shard_tasks = (
                        (group_idx, group, num_shards_buckets, out_tmp_dirs,
                         compress_shards, shard_chunk_size)
                        for group_idx, group in enumerate(
                            iter_worker_groups(split, worker_group_size, group_shuffle_rng)
                        )
                    )
                    for completed_groups, counts in enumerate(pool.imap_unordered(shardify_star, shard_tasks), start=1):
                        bucket_row_counts += counts
                        total_sharded += int(counts.sum())
                        progress.maybe_report(
                            completed_groups,
                            extra=f"rows={total_sharded}",
                        )
                    progress.final_report(num_worker_groups, extra=f"rows={total_sharded}")
                    log(f"  Sharded {total_sharded} rows")
        else:
            total_sharded = 0

        with Timer(f"Parallel merge+repack ({split.name})"):
            written_files, remainder = parallel_merge_repack(
                num_shards=num_worker_groups,
                out_tmp_dirs=out_tmp_dirs,
                out_dir=split.out_dir,
                rows_per_file=rows_per_file,
                num_processes=merge_processes,
                keep_remainder=keep_remainder,
                extra_rows=extra_rows,
                tmp_base_dir=split.tmp_dir,
                bucket_row_counts=bucket_row_counts.tolist(),
                progress_interval_sec=progress_interval_sec,
            )

    # Write index.json
    index_entries = []
    total_written = 0
    for fname, nr in written_files:
        index_entries.append({"name": os.path.basename(fname), "num_rows": nr})
        total_written += nr

    index = {
        "files": index_entries,
        "total_rows": total_written,
        "rows_per_file": rows_per_file,
    }
    index_path = os.path.join(split.out_dir, "index.json")
    atomic_save_json(index_path, index)

    # Summary
    if index_entries:
        row_counts = [e["num_rows"] for e in index_entries]
        if len(set(row_counts)) == 1:
            log(f"  {len(index_entries)} files, all {row_counts[0]} rows")
        else:
            full_count = sum(1 for r in row_counts[:-1] if r == row_counts[0])
            log(f"  {full_count} full files ({row_counts[0]} rows each), "
                f"last file: {row_counts[-1]} rows")

    remainder_count = remainder["binaryInputNCHWPacked"].shape[0] if remainder else 0
    total_input = total_sharded + extra_count
    discarded = total_input - total_written - remainder_count
    log(f"  Total rows written: {total_written}" +
        (f" (discarded: {discarded})" if discarded > 0 else ""))
    log(f"  Index written to: {index_path}")

    if enable_shard_cache:
        final_remainder_path = split_cache_final_remainder_path(split)
        if keep_remainder and remainder is not None:
            atomic_save_npz(final_remainder_path, remainder, compressed=False)
        elif os.path.exists(final_remainder_path):
            os.remove(final_remainder_path)
        save_shard_cache_state(split, {
            "config": cache_config,
            "status": "complete",
            "mode": "wave" if use_wave_mode else "non_wave",
            "completed_groups": num_worker_groups,
            "total_sharded": total_sharded,
            "bucket_output_counts": (
                bucket_output_counts if use_wave_mode else None
            ),
        })

    # Cleanup tmp dirs
    for d in out_tmp_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
    log(f"  Temp dirs cleaned up for '{split.name}'.")

    return remainder


def parse_split(s):
    """Parse a --split argument string 'name:md5_lo:md5_hi:out_dir:tmp_dir'."""
    parts = s.split(":")
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            f"--split requires 5 colon-separated fields "
            f"(name:md5_lo:md5_hi:out_dir:tmp_dir), got {len(parts)}: '{s}'"
        )
    name, md5_lo, md5_hi, out_dir, tmp_dir = parts
    try:
        md5_lo = float(md5_lo)
        md5_hi = float(md5_hi)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"md5_lo and md5_hi must be floats, got '{parts[1]}' and '{parts[2]}'"
        )
    return SplitConfig(
        name=name,
        md5_lbound=md5_lo,
        md5_ubound=md5_hi,
        out_dir=out_dir,
        tmp_dir=tmp_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="Shuffle NPZ training data for nano.")
    parser.add_argument("dirs", nargs="+", help="Input directories containing NPZ files")
    parser.add_argument("--num-processes", type=int, required=True, help="Number of parallel workers")
    parser.add_argument("--shard-processes", type=int, default=None,
                        help="Parallel workers for shardify only (default: --num-processes)")
    parser.add_argument("--merge-processes", type=int, default=None,
                        help="Parallel workers for merge/repack only (default: --num-processes)")
    parser.add_argument("--rows-per-file", type=int, default=131072,
                        help="Exact rows per output file (default: 131072). "
                             "Use a power of 2 so it divides evenly by batch_size * world_size.")
    parser.add_argument("--worker-group-size", type=int, default=80000,
                        help="Target rows per sharding worker group (default: 80000)")
    parser.add_argument("--filter-board-size", type=int, default=None,
                        help="Keep only files matching this board size (e.g. 19 for 19x19)")
    parser.add_argument("--split", type=parse_split, action="append", dest="splits",
                        metavar="name:md5_lo:md5_hi:out_dir:tmp_dir",
                        help="Define a split (repeatable). Format: name:md5_lo:md5_hi:out_dir:tmp_dir")
    parser.add_argument("--val-num-files", type=int, default=None,
                        help="Fixed number of val output files (val total = N * rows-per-file). "
                             "Overrides MD5 bounds for val split.")
    parser.add_argument("--num-buckets", type=int, default=None,
                        help="Number of intermediate shard buckets (default: merge-processes)")
    parser.add_argument("--shard-chunk-size", type=int, default=16384,
                        help="Rows per chunk when distributing to buckets (default: 16384). "
                             "Larger = fewer shard files, slightly less uniform bucket sizes.")
    parser.add_argument("--max-active-worker-groups", type=int, default=32,
                        help="Max sharding worker groups to keep on local temp storage before "
                             "running an intermediate merge wave (default: 32). "
                             "Set <= 0 to disable wave mode.")
    parser.add_argument("--shard-cache", action="store_true", default=False,
                        help="Persist wave-mode sharding progress in the split output directory "
                             "and resume from completed waves on reruns.")
    parser.add_argument("--compress-shards", action="store_true", default=False,
                        help="Compress intermediate shard files (saves tmp disk space, slower)")
    parser.add_argument("--progress-interval-sec", type=float, default=30.0,
                        help="Seconds between progress updates during sharding/merge (default: 30)")
    parser.add_argument("--scan-cache", type=str, default=None,
                        help="Optional SQLite cache file for scan results. Reuses row counts "
                             "and board-size checks across reruns of the same dataset. "
                             "Cache entries are keyed by path and are not invalidated by mtime/size.")
    args = parser.parse_args()

    if not args.splits:
        log("ERROR: at least one --split is required.", file=sys.stderr)
        sys.exit(1)

    if args.val_num_files is not None:
        if args.val_num_files <= 0:
            log("ERROR: --val-num-files must be > 0.", file=sys.stderr)
            sys.exit(1)
        split_names = {s.name for s in args.splits}
        if "val" not in split_names or "train" not in split_names:
            log("ERROR: --val-num-files requires both 'val' and 'train' splits.", file=sys.stderr)
            sys.exit(1)

    if args.filter_board_size is not None and args.filter_board_size <= 0:
        log("ERROR: --filter-board-size must be > 0.", file=sys.stderr)
        sys.exit(1)

    # Pre-check: fail fast if any output directory already exists, unless shard
    # cache/resume is explicitly enabled.
    if not args.shard_cache:
        for split in args.splits:
            if os.path.exists(split.out_dir):
                log(f"ERROR: Output directory already exists: {split.out_dir}", file=sys.stderr)
                sys.exit(1)

    num_processes = args.num_processes
    shard_processes = args.shard_processes if args.shard_processes is not None else num_processes
    merge_processes = args.merge_processes if args.merge_processes is not None else num_processes
    rows_per_file = args.rows_per_file
    worker_group_size = args.worker_group_size
    num_buckets = args.num_buckets if args.num_buckets is not None else merge_processes
    shard_chunk_size = args.shard_chunk_size
    max_active_worker_groups = args.max_active_worker_groups
    if max_active_worker_groups <= 0:
        max_active_worker_groups = None
    shard_cache_enabled = args.shard_cache
    progress_interval_sec = args.progress_interval_sec
    scan_cache_path = args.scan_cache

    # --- Stage 1: Find all NPZ files ---
    all_files = []
    with Timer("Finding files"):
        for d in args.dirs:
            for root, dirs, files in os.walk(d, followlinks=True):
                for f in files:
                    if f.endswith(".npz"):
                        all_files.append(os.path.join(root, f))
    log(f"Found {len(all_files)} NPZ files")

    if not all_files:
        log("No files found, exiting.")
        sys.exit(0)

    scan_tmp_dir = tempfile.mkdtemp(
        prefix="shuffle.scan.",
        dir=choose_scan_tmp_root(args.splits),
    )
    master_manifest_path = os.path.join(scan_tmp_dir, "all_valid.manifest")

    # --- Stage 2: Scan files (row counts + optional board size filter) ---
    board_size = args.filter_board_size
    valid_files = 0
    total_rows = 0
    bad_files = 0
    filtered_by_board = 0
    sample_file = None
    fixed_val_selector = None
    if args.val_num_files is not None:
        fixed_val_selector = FixedValSelector(args.val_num_files * rows_per_file)
    if board_size is not None:
        scan_desc = f"Scanning files (row counts + {board_size}x{board_size} filter)"
    else:
        scan_desc = "Scanning files (row counts)"
    with Timer(scan_desc):
        cache_conn = None
        cache_hits = 0
        cache_misses = 0
        if scan_cache_path is not None:
            cache_conn = open_scan_cache(scan_cache_path)
            log(f"  Scan cache: {scan_cache_path}")

        try:
            with open(master_manifest_path, "w", encoding="utf-8", newline="") as master_manifest:
                with multiprocessing.Pool(num_processes) as pool:
                    progress = ProgressLogger(
                        desc="scan files",
                        total=len(all_files),
                        unit="files",
                        interval_sec=progress_interval_sec,
                    )
                    scanned_files = 0

                    def handle_scan_result(filename, num_rows, ok):
                        nonlocal valid_files, total_rows, bad_files, filtered_by_board, sample_file

                        is_valid = num_rows is not None and num_rows > 0 and ok
                        total_rows, bad_files, filtered_by_board = apply_scan_result(
                            filename, num_rows, ok,
                            total_rows, bad_files, filtered_by_board,
                        )
                        if is_valid:
                            valid_files += 1
                            if sample_file is None:
                                sample_file = filename
                            write_manifest_entry(master_manifest, filename, num_rows)
                            if fixed_val_selector is not None:
                                fixed_val_selector.add(filename, num_rows)

                    if cache_conn is None:
                        scan_iter = zip(all_files, itertools.repeat(board_size))
                        for filename, num_rows, ok in pool.imap_unordered(scan_file, scan_iter, chunksize=64):
                            handle_scan_result(filename, num_rows, ok)
                            scanned_files += 1
                            progress.maybe_report(
                                scanned_files,
                                extra=f"valid={valid_files}, bad={bad_files}, "
                                      f"filtered={filtered_by_board}, rows={total_rows}",
                            )
                    else:
                        for chunk_start in range(0, len(all_files), SCAN_CACHE_PROCESS_CHUNK_SIZE):
                            chunk_files = all_files[chunk_start:chunk_start + SCAN_CACHE_PROCESS_CHUNK_SIZE]
                            cached_results = fetch_scan_cache_entries(cache_conn, board_size, chunk_files)

                            for filename in chunk_files:
                                entry = cached_results.get(filename)
                                if entry is None:
                                    continue
                                num_rows, ok = entry
                                handle_scan_result(filename, num_rows, ok)
                                scanned_files += 1
                                cache_hits += 1
                                progress.maybe_report(
                                    scanned_files,
                                    extra=f"valid={valid_files}, bad={bad_files}, "
                                          f"filtered={filtered_by_board}, rows={total_rows}, "
                                          f"cache_hits={cache_hits}, cache_misses={cache_misses}",
                                )

                            missing = [filename for filename in chunk_files if filename not in cached_results]
                            if not missing:
                                continue

                            to_store = []
                            miss_iter = zip(missing, itertools.repeat(board_size))
                            for filename, num_rows, ok in pool.imap_unordered(scan_file, miss_iter, chunksize=64):
                                handle_scan_result(filename, num_rows, ok)
                                scanned_files += 1
                                cache_misses += 1
                                to_store.append((filename, num_rows, ok))
                                progress.maybe_report(
                                    scanned_files,
                                    extra=f"valid={valid_files}, bad={bad_files}, "
                                          f"filtered={filtered_by_board}, rows={total_rows}, "
                                          f"cache_hits={cache_hits}, cache_misses={cache_misses}",
                                )

                            store_scan_cache_entries(cache_conn, board_size, to_store)
        finally:
            if cache_conn is not None:
                cache_conn.close()

    log(f"Valid files: {valid_files}, bad/empty: {bad_files}")
    if board_size is not None:
        log(f"Filtered by board size ({board_size}x{board_size}): {filtered_by_board} files removed")
    log(f"Total rows: {total_rows}")
    if scan_cache_path is not None:
        log(f"Scan cache stats: {cache_hits} hits, {cache_misses} misses")
    if sample_file is not None:
        log_memory_estimates(
            sample_file=sample_file,
            worker_group_size=worker_group_size,
            rows_per_file=rows_per_file,
            shard_processes=shard_processes,
            merge_processes=merge_processes,
        )
    if fixed_val_selector is not None and total_rows < fixed_val_selector.target_rows:
        log(f"WARNING: total rows ({total_rows}) < val target ({fixed_val_selector.target_rows}). "
            f"Using all data for val.")

    if total_rows == 0:
        log("No rows found, exiting.")
        sys.exit(0)

    # --- Stage 3: Split files into val/train ---
    splits_by_name = {s.name: s for s in args.splits}
    for split in args.splits:
        split.file_rows = None
        split.num_files = 0
        split.total_rows = 0
        split.manifest_path = os.path.join(scan_tmp_dir, f"{split.name}.manifest")

    selected_val_paths = None
    if args.val_num_files is not None:
        selected_val_paths = fixed_val_selector.selected_paths() if fixed_val_selector is not None else set()

    with Timer("Assigning files to splits"):
        handles = {
            split.name: open(split.manifest_path, "w", encoding="utf-8", newline="")
            for split in args.splits
        }
        try:
            progress = ProgressLogger(
                desc="split manifests",
                total=valid_files,
                unit="files",
                interval_sec=progress_interval_sec,
            )
            for processed_files, (filename, num_rows) in enumerate(iter_manifest_entries(master_manifest_path), start=1):
                h = None

                if selected_val_paths is not None:
                    if filename in selected_val_paths:
                        append_split_entry(splits_by_name["val"], handles["val"], filename, num_rows)
                    else:
                        append_split_entry(splits_by_name["train"], handles["train"], filename, num_rows)

                    for split in args.splits:
                        if split.name in ("train", "val"):
                            continue
                        if h is None:
                            h = md5_hash_float(os.path.basename(filename))
                        if split.md5_lbound <= h < split.md5_ubound:
                            append_split_entry(split, handles[split.name], filename, num_rows)
                else:
                    for split in args.splits:
                        if h is None:
                            h = md5_hash_float(os.path.basename(filename))
                        if split.md5_lbound <= h < split.md5_ubound:
                            append_split_entry(split, handles[split.name], filename, num_rows)

                progress.maybe_report(processed_files)
        finally:
            for handle in handles.values():
                handle.close()

    if args.val_num_files is not None:
        val_split = splits_by_name["val"]
        train_split = splits_by_name["train"]
        log(f"Split 'val': {val_split.num_files}/{valid_files} files, "
            f"{val_split.total_rows}/{total_rows} rows "
            f"(fixed {args.val_num_files} output files)")
        log(f"Split 'train': {train_split.num_files}/{valid_files} files, "
            f"{train_split.total_rows}/{total_rows} rows")

        for split in args.splits:
            if split.name not in ("train", "val"):
                log(f"Split '{split.name}': {split.num_files}/{valid_files} files, "
                    f"{split.total_rows}/{total_rows} rows "
                    f"(MD5 [{split.md5_lbound:.2f}, {split.md5_ubound:.2f}))")
    else:
        for split in args.splits:
            log(f"Split '{split.name}': {split.num_files}/{valid_files} files, "
                f"{split.total_rows}/{total_rows} rows "
                f"(MD5 [{split.md5_lbound:.2f}, {split.md5_ubound:.2f}))")

    # --- Stage 4: Process splits (val first, then train) ---
    # Val is processed first so its remainder rows can be added to train.
    val_split = splits_by_name.get("val")
    train_split = splits_by_name.get("train")

    compress_shards = args.compress_shards

    try:
        val_remainder = None
        if val_split is not None:
            val_remainder = process_split(
                val_split, shard_processes, merge_processes, rows_per_file, worker_group_size,
                num_buckets=num_buckets, shard_chunk_size=shard_chunk_size,
                keep_remainder=True, compress_shards=compress_shards,
                progress_interval_sec=progress_interval_sec,
                max_active_worker_groups=max_active_worker_groups,
                enable_shard_cache=shard_cache_enabled,
            )

        if train_split is not None:
            process_split(
                train_split, shard_processes, merge_processes, rows_per_file, worker_group_size,
                num_buckets=num_buckets, shard_chunk_size=shard_chunk_size,
                extra_rows=val_remainder, compress_shards=compress_shards,
                progress_interval_sec=progress_interval_sec,
                max_active_worker_groups=max_active_worker_groups,
                enable_shard_cache=shard_cache_enabled,
            )

        # Process any other splits (neither train nor val)
        for split in args.splits:
            if split.name not in ("train", "val"):
                process_split(split, shard_processes, merge_processes, rows_per_file, worker_group_size,
                              num_buckets=num_buckets, shard_chunk_size=shard_chunk_size,
                              compress_shards=compress_shards,
                              progress_interval_sec=progress_interval_sec,
                              max_active_worker_groups=max_active_worker_groups,
                              enable_shard_cache=shard_cache_enabled)

        print("", flush=True)
        log("All splits done.")
    finally:
        shutil.rmtree(scan_tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
