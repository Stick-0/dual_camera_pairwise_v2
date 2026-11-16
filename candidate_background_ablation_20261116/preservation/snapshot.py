#!/usr/bin/env python3
"""Read-only preservation inventory; output is confined to this experiment."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time

sys.dont_write_bytecode = True
NEW = Path(__file__).resolve().parents[1]
ROOT = NEW.parent
HERE = NEW / 'preservation'
DATA_USED = Path('Data/real/Dual_UR/UR_dual_2596')
ENV_PARTS = {'.venv', 'venv', 'site-packages', 'node_modules'}
ARCHIVES = {'.zip', '.gz', '.tar', '.mp4', '.avi', '.mov', '.mkv'}
SANDBOX_MOUNT_ROOTS = {'.git', '.agents', '.aws', '.codex'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inventory():
    stack = [ROOT]
    while stack:
        parent = stack.pop()
        try:
            entries = sorted(os.scandir(parent), key=lambda p: p.name)
        except OSError as exc:
            yield {'path': str(parent.relative_to(ROOT)), 'error': repr(exc)}
            continue
        for entry in entries:
            path = Path(entry.path)
            if path == NEW:
                continue
            try:
                s = entry.stat(follow_symlinks=False)
                kind = ('directory' if stat.S_ISDIR(s.st_mode) else 'file' if stat.S_ISREG(s.st_mode)
                        else 'symlink' if stat.S_ISLNK(s.st_mode) else 'other')
                row = {'path': str(path.relative_to(ROOT)), 'kind': kind, 'size': s.st_size,
                       'mode': s.st_mode, 'uid': s.st_uid, 'gid': s.st_gid,
                       'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns,
                       'inode': s.st_ino, 'device': s.st_dev}
                if kind == 'symlink':
                    row['target'] = os.readlink(path)
                yield row
                if kind == 'directory':
                    stack.append(path)
            except OSError as exc:
                yield {'path': str(path.relative_to(ROOT)), 'error': repr(exc)}


def hash_scope(row):
    if row.get('kind') != 'file':
        return False, 'not_regular_file'
    p = Path(row['path'])
    if any(part in ENV_PARTS for part in p.parts):
        return False, 'environment_metadata_only'
    if p.parts[0] == 'Data' and not p.is_relative_to(DATA_USED):
        return False, 'unused_dataset_metadata_only'
    if p.suffix.lower() in ARCHIVES:
        return False, 'unused_archive_video_metadata_only'
    return True, 'sha256'


def read_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def before(workers):
    metadata = HERE / 'before_metadata.jsonl'
    if metadata.exists():
        rows = read_jsonl(metadata)
    else:
        rows = []
        with metadata.open('x') as f:
            for row in inventory():
                rows.append(row)
                f.write(json.dumps(row, separators=(',', ':')) + '\n')
    counts, bytes_by_group, selected = {}, {}, []
    for row in rows:
        yes, reason = hash_scope(row)
        counts[reason] = counts.get(reason, 0) + 1
        bytes_by_group[reason] = bytes_by_group.get(reason, 0) + row.get('size', 0)
        if yes:
            selected.append(row)
    summary = {'root': str(ROOT), 'excluded_new_tree': str(NEW), 'metadata_entries': len(rows),
               'counts_by_hash_scope': counts, 'bytes_by_hash_scope': bytes_by_group,
               'metadata_covers_all_discovered_original_workspace_entries': True,
               'metadata_errors': [r for r in rows if 'error' in r],
               'hash_scope': 'All regular workspace files except installed environments, unused Data subtrees, and archive/video files; all UR_dual_2596 dataset files and all original code/config/checkpoint/result/cache files outside those exceptions are hashed.',
               'access_times_intentionally_not_compared': 'Reading may update atime; content, size, mode, ownership, mtime, ctime, inode and symlink target are preserved.',
               'sandbox_managed_mount_roots': sorted(SANDBOX_MOUNT_ROOTS),
               'sandbox_mount_limit': 'The managed sandbox exposes these read-only mount roots with per-invocation mtime/ctime/device; changes limited to these three surface fields are separately reported. Hidden host-side mount contents are not claimed inventoried.',
               'started_unix': time.time(), 'hashes_complete': False}
    (HERE / 'before_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'metadata_complete': True, 'entries': len(rows),
                      'hash_files': len(selected), 'hash_bytes': bytes_by_group.get('sha256', 0)}), flush=True)
    hashes = HERE / 'before_hashes.jsonl'
    done = {r['path']: r for r in read_jsonl(hashes)} if hashes.exists() else {}
    pending = [r for r in selected if r['path'] not in done]
    def calculate(row):
        try:
            return {'path': row['path'], 'sha256': sha(ROOT / row['path']), 'size': row['size']}
        except OSError as exc:
            return {'path': row['path'], 'error': repr(exc)}
    with hashes.open('a') as f, concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(calculate, pending)):
            f.write(json.dumps(row, separators=(',', ':')) + '\n')
            done[row['path']] = row
            if i % 1000 == 0:
                f.flush()
    summary.update(hashes_complete=True, completed_unix=time.time(),
                   hashed_files=sum('sha256' in r for r in done.values()),
                   hash_errors=[r for r in done.values() if 'error' in r],
                   metadata_sha256=sha(metadata), hashes_manifest_sha256=sha(hashes))
    (HERE / 'before_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'before_complete': True, 'hashed_files': summary['hashed_files'],
                      'errors': len(summary['hash_errors'])}), flush=True)


def after(workers, label):
    before_rows = {r['path']: r for r in read_jsonl(HERE / 'before_metadata.jsonl')}
    current_rows = {r['path']: r for r in inventory()}
    missing = sorted(before_rows.keys() - current_rows.keys())
    added = sorted(current_rows.keys() - before_rows.keys())
    changed, surface_changes = [], []
    for p in before_rows.keys() & current_rows.keys():
        a, b = before_rows[p], current_rows[p]
        if a == b:
            continue
        changed_fields = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
        record = {'path': p, 'changed_fields': changed_fields, 'before': a, 'after': b}
        if p in SANDBOX_MOUNT_ROOTS and set(changed_fields) <= {'mtime_ns', 'ctime_ns', 'device'}:
            surface_changes.append(record)
        else:
            changed.append(record)
    originals = read_jsonl(HERE / 'before_hashes.jsonl')
    def check(row):
        try:
            actual = sha(ROOT / row['path'])
            return None if actual == row.get('sha256') else {'path': row['path'], 'before_sha256': row.get('sha256'), 'after_sha256': actual}
        except OSError as exc:
            return {'path': row['path'], 'error': repr(exc)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        hash_changes = [r for r in pool.map(check, originals) if r is not None]
    result = {'label': label, 'checked_unix': time.time(), 'before_files_hashed': len(originals),
              'metadata_missing': missing, 'metadata_added_outside_new': added,
              'metadata_changed': changed, 'hash_changes': hash_changes,
              'sandbox_managed_mount_surface_changes': surface_changes,
              'sandbox_surface_rule': 'Only mtime_ns, ctime_ns and device on the four sandbox-managed roots are separately reported; their content hashes, size, permissions, ownership, symlink targets and every ordinary path remain checked.',
              'passed': not (missing or added or changed or hash_changes)}
    (HERE / f'{label}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': result['passed'], 'missing': len(missing), 'added': len(added),
                      'metadata_changed': len(changed), 'hash_changes': len(hash_changes)}))
    return result['passed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['before', 'after'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--label', default='after')
    args = parser.parse_args()
    if args.mode == 'before':
        before(args.workers)
    elif not after(args.workers, args.label):
        sys.exit(1)
