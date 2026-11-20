#!/usr/bin/env python3
"""Read frozen R0/R1 results and render every frame; never run inference."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
SOURCE = WORKSPACE / 'candidate_background_ablation_20261116'
for key, sub in [('MPLCONFIGDIR', 'mpl'), ('XDG_CACHE_HOME', 'xdg'), ('TMPDIR', 'tmp')]:
    target = ROOT / 'cache' / sub
    target.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(target)
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import cv2
import numpy as np
import yaml

METHODS = ('R0', 'R1')
CAMERAS = ('camera1', 'camera2')
AXIS_M = 0.12


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    path = Path(path).resolve()
    assert path.is_relative_to(ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def load_rows(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def projection(T, xyz, camera):
    points = np.asarray(xyz, float) @ T[:3, :3].T + T[:3, 3]
    assert np.isfinite(points).all() and np.all(points[:, 2] > .001)
    return cv2.projectPoints(points, np.zeros(3), np.zeros(3),
                             np.asarray(camera['K'], float),
                             np.asarray(camera['D'], float))[0].reshape(-1, 2)


def prepare():
    config_path = SOURCE / 'configs/experiment.yaml'
    cfg = yaml.safe_load(config_path.read_text())
    paths = {'config': config_path, 'pairs': Path(cfg['inputs']['pairs']),
             'reference': Path(cfg['inputs']['reference']),
             'results': SOURCE / 'results/full/pair_results.jsonl',
             'metrics': SOURCE / 'reports/final/per_observation_metrics.csv',
             'fk_source': WORKSPACE / 'ur5e_capture/geometry.py',
             'solver_source': SOURCE / 'src/controlled_solver.py'}
    before = {k: {'path': str(p), 'sha256': sha(p), 'size': p.stat().st_size,
                  'mtime_ns': p.stat().st_mtime_ns} for k, p in paths.items()}
    pairs = {p['pair_id']: p for p in load_rows(paths['pairs'])}
    refs = {p['pair_id']: p for p in load_rows(paths['reference'])}
    ordered = sorted(pairs, key=lambda p: (np.mean([pairs[p]['cameras'][c]['timestamp_s']
                                                  for c in CAMERAS]), p))
    assert len(ordered) == 2596 and len(set(ordered)) == 2596
    ordinals = {p: i + 1 for i, p in enumerate(ordered)}
    with paths['metrics'].open() as f:
        metrics = {(r['pair_id'], r['method'], r['camera']): r
                   for r in csv.DictReader(f) if r['method'] in METHODS}
    compact = {}
    max_add = max_projection = max_origin = 0.
    axis_outside = 0
    source_images = []
    for result in load_rows(paths['results']):
        pid = result['pair_id']
        assert pid not in compact
        pair = pairs[pid]
        record = {'frame_index': ordinals[pid], 'pair_id': pid,
                  'timestamp_s': float(np.mean([pair['cameras'][c]['timestamp_s'] for c in CAMERAS])),
                  'cameras': {}}
        assert result['methods']['R0']['initialization']['pose_array_hash'] == result['methods']['R1']['initialization']['pose_array_hash']
        for camera in CAMERAS:
            cam = pair['cameras'][camera]
            image = Path(cam['image_path'])
            st = image.stat()
            source_images.append({'path': str(image), 'sha256': cam['image_sha256'],
                                  'size': st.st_size, 'mtime_ns': st.st_mtime_ns})
            xyz = np.asarray(cam['keypoints_base_m'], float)
            assert xyz.shape == (7, 3) and np.array_equal(xyz[0], [0, 0, 0])
            cr = {k: cam[k] for k in ('image_path', 'image_sha256', 'width', 'height', 'K', 'D')}
            cr['source_mtime_ns'] = st.st_mtime_ns
            cr['methods'] = {}
            raw0 = result['methods']['R0']['raw_inputs'][camera]
            assert raw0 == result['methods']['R1']['raw_inputs'][camera]
            for method in METHODS:
                m = result['methods'][method]
                T = np.asarray(m['poses'][camera], float)
                assert T.shape == (4, 4) and np.isfinite(T).all()
                assert np.allclose(T[3], [0, 0, 0, 1])
                assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6)
                assert np.isclose(np.linalg.det(T[:3, :3]), 1., atol=1e-6)
                assert m['final_valid'] and m['common_valid']
                assert np.array_equal(np.asarray(raw0['keypoints_base_m']), xyz)
                uv = projection(T, xyz, cam)
                saved = np.asarray(m['pose_projected'][camera], float)
                err = float(np.max(np.abs(saved - uv)))
                max_projection = max(max_projection, err)
                assert err < 1e-7, (pid, method, camera, err)
                axis_xyz = np.vstack((np.zeros(3), np.eye(3) * AXIS_M))
                axis_uv = projection(T, axis_xyz, cam)
                max_origin = max(max_origin, float(np.max(np.abs(axis_uv[0] - uv[0]))))
                axis_outside += int(np.any((axis_uv[:, 0] < 0) | (axis_uv[:, 0] >= cam['width']) |
                                          (axis_uv[:, 1] < 0) | (axis_uv[:, 1] >= cam['height'])))
                reference = np.asarray(refs[pid]['cameras'][camera]['T_camera_from_base'], float)
                estimate_pts = xyz @ T[:3, :3].T + T[:3, 3]
                reference_pts = xyz @ reference[:3, :3].T + reference[:3, 3]
                add = float(np.linalg.norm(estimate_pts - reference_pts, axis=1).mean() * 1000.)
                metric = metrics[(pid, method, camera)]
                assert metric['final_valid'] == 'True'
                max_add = max(max_add, abs(add - float(metric['final_ADD_mm'])))
                assert abs(add - float(metric['final_ADD_mm'])) < 1e-8
                cr['methods'][method] = {'pose_base_to_camera': T.tolist(), 'add_mm': float(metric['final_ADD_mm']),
                    'raw_uv': raw0['keypoints_uv'], 'raw_valid': raw0['valid_keypoints'],
                    'map_uv': m['candidate_selected_uv'][camera],
                    'map_valid': m['candidate_selected_valid'][camera],
                    'projected_fk_uv': uv.tolist(), 'axis_uv': axis_uv.tolist(), 'final_valid': True}
            record['cameras'][camera] = cr
        compact[pid] = record
    assert set(compact) == set(ordered) and len(metrics) == 10384
    with (ROOT / 'data/render_frames.jsonl').open('x') as f:
        for pid in ordered:
            f.write(json.dumps(compact[pid], separators=(',', ':'), allow_nan=False) + '\n')
    dump(ROOT / 'data/source_images.json', source_images)
    assert all(sha(paths[k]) == v['sha256'] for k, v in before.items())
    dump(ROOT / 'validation/preparation.json', {
        'passed': True, 'n_pairs': 2596, 'n_pose_observations': 10384,
        'source_files': before, 'source_files_unchanged_after_read': True,
        'maximum_ADD_recalculation_difference_mm': max_add,
        'maximum_saved_pose_projection_difference_px': max_projection,
        'maximum_base_axis_origin_vs_fk0_difference_px': max_origin,
        'axis_length_m': AXIS_M, 'axis_outside_image_count': axis_outside,
        'all_R0_R1_initial_pose_hashes_equal': True,
        'all_R0_R1_raw_observations_equal': True,
        'reference_used_only_for_ADD_crosscheck': True,
        'inference_or_optimization_rerun': False,
        'ordering': 'Mean of camera capture timestamps; pair_id tie-break; frame index 1..2596'})
    print('Prepared 2596 pairs; all 10384 poses, projections and ADD values verified.', flush=True)


def text(image, content, xy, color=(240, 240, 240), scale=.45, thickness=1):
    cv2.putText(image, content, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (12, 12, 12), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, content, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def point(uv):
    return tuple(np.rint(np.asarray(uv)).astype(int))


def inside(uv, w, h):
    return np.isfinite(uv).all() and 0 <= uv[0] < w and 0 <= uv[1] < h


def segment(image, a, b, color, width=1, arrow=False):
    if not np.isfinite([a, b]).all():
        return
    # Guard integer conversion and clip without shifting any geometric point.
    a, b = np.clip(a, -1e7, 1e7), np.clip(b, -1e7, 1e7)
    ok, q, r = cv2.clipLine((0, 0, image.shape[1], image.shape[0]), point(a), point(b))
    if ok:
        cv2.line(image, q, r, (10, 10, 10), width + 2, cv2.LINE_AA)
        if arrow:
            cv2.arrowedLine(image, q, r, color, width, cv2.LINE_AA, tipLength=.18)
        else:
            cv2.line(image, q, r, color, width, cv2.LINE_AA)


def draw_one(image, entry, method, camera, record):
    canvas = image.copy()
    h, w = canvas.shape[:2]
    cyan, orange, yellow = (255, 220, 25), (0, 155, 255), (20, 240, 255)
    uv = np.asarray(entry['projected_fk_uv'])
    for a, b in zip(uv[:-1], uv[1:]):
        segment(canvas, a, b, cyan, 1)
    for raw, valid in zip(entry['raw_uv'], entry['raw_valid']):
        if valid and inside(raw, w, h):
            cv2.drawMarker(canvas, point(raw), (10, 10, 10), cv2.MARKER_CROSS, 13, 3, cv2.LINE_AA)
            cv2.drawMarker(canvas, point(raw), orange, cv2.MARKER_CROSS, 13, 1, cv2.LINE_AA)
    for selected, valid in zip(entry['map_uv'], entry['map_valid']):
        if valid and inside(selected, w, h):
            cv2.circle(canvas, point(selected), 6, (10, 10, 10), 3, cv2.LINE_AA)
            cv2.circle(canvas, point(selected), 6, yellow, 1, cv2.LINE_AA)
    for i, p in enumerate(uv):
        if inside(p, w, h):
            cv2.circle(canvas, point(p), 3, (5, 5, 5), -1, cv2.LINE_AA)
            cv2.circle(canvas, point(p), 2, cyan, -1, cv2.LINE_AA)
            label = (min(w-16, max(2, int(p[0])+8)), min(h-5, max(13, int(p[1])-7)))
            text(canvas, str(i), label, cyan, .4, 1)
    axes = np.asarray(entry['axis_uv'])
    for i, (label, color) in enumerate(zip('XYZ', [(40, 40, 255), (60, 230, 50), (255, 95, 60)]), 1):
        segment(canvas, axes[0], axes[i], color, 2, True)
        if inside(axes[i], w, h):
            pos = (min(w-18, max(2, int(axes[i, 0])+5)), min(h-6, max(16, int(axes[i, 1])-5)))
            text(canvas, label, pos, color, .5, 1)
    output = np.full((h + 124, w, 3), (24, 27, 32), np.uint8)
    output[52:52+h] = canvas
    on = 'OFF' if method == 'R0' else 'ON'
    text(output, f'{method}  |  PnP init + Background {on}  |  {camera}', (12, 20), scale=.48)
    text(output, f'Frame {record["frame_index"]:04d}/2596  |  pair {record["pair_id"]}  |  ADD {entry["add_mm"]:.2f} mm', (12, 41), scale=.45)
    y = h + 72
    text(output, '+  Raw voting input', (12, y), orange, .43)
    text(output, 'o  Posterior MAP candidate', (265, y), yellow, .43)
    text(output, '0-6  Pose-projected FK skeleton', (12, y+21), cyan, .43)
    text(output, f'Robot-base axes: {AXIS_M*1000:.0f} mm', (12, y+42), scale=.43)
    for i, (label, color) in enumerate(zip('XYZ', [(40,40,255),(60,230,50),(255,95,60)])):
        text(output, label, (390+48*i, y+42), color, .5)
    return output


def render_task(task):
    record, output = task
    cv2.setNumThreads(1)
    rows = []
    for camera in CAMERAS:
        cr = record['cameras'][camera]
        source = Path(cr['image_path'])
        raw = source.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == cr['image_sha256']
        assert source.stat().st_mtime_ns == cr['source_mtime_ns']
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        assert image is not None and image.shape == (cr['height'], cr['width'], 3)
        for method in METHODS:
            drawn = draw_one(image, cr['methods'][method], method, camera, record)
            target = Path(output) / method / camera / (record['pair_id']+'.jpg')
            ok, encoded = cv2.imencode('.jpg', drawn, [cv2.IMWRITE_JPEG_QUALITY, 94])
            assert ok
            data = encoded.tobytes()
            with target.open('xb') as f:
                f.write(data)
            rows.append({'frame_index': record['frame_index'], 'pair_id': record['pair_id'],
                'method': method, 'camera': camera, 'path': str(target.relative_to(ROOT)),
                'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data),
                'width': drawn.shape[1], 'height': drawn.shape[0],
                'source_image_sha256': cr['image_sha256'],
                'add_mm': cr['methods'][method]['add_mm']})
    return rows


def render(args):
    out = (ROOT / args.output).resolve()
    assert out.is_relative_to(ROOT) and not any(out.rglob('*'))
    for method in METHODS:
        for camera in CAMERAS:
            (out / method / camera).mkdir(parents=True, exist_ok=False)
    records = list(load_rows(ROOT/'data/render_frames.jsonl'))
    if args.limit:
        # Prespecified distributed smoke frames, independent of ADD.
        indices = np.linspace(0, len(records)-1, args.limit, dtype=int)
        records = [records[i] for i in indices]
    started = time.perf_counter()
    count = 0
    with (out/'manifest.jsonl').open('x') as f, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rows in pool.map(render_task, ((r,str(out)) for r in records), chunksize=8):
            for row in rows:
                f.write(json.dumps(row)+'\n')
            count += 1
            if count % 200 == 0 or count == len(records):
                print(f'Rendered {count}/{len(records)} pairs ({count*4} images)', flush=True)
    dump(out/'completion.json', {'passed': True, 'pairs': count, 'images': count*4,
        'seconds': time.perf_counter()-started, 'workers': args.workers,
        'axis_length_m': AXIS_M, 'source_RGB_sha256_verified_for_every_pair': True})


def verify():
    meta = json.loads((ROOT/'validation/preparation.json').read_text())
    checks = {k: (sha(v['path']) == v['sha256'] and
                  Path(v['path']).stat().st_size == v['size'] and
                  Path(v['path']).stat().st_mtime_ns == v['mtime_ns'])
              for k,v in meta['source_files'].items()}
    images = json.loads((ROOT/'data/source_images.json').read_text())
    images_ok = True
    for image in images:
        p = Path(image['path']); st = p.stat()
        images_ok &= st.st_size == image['size'] and st.st_mtime_ns == image['mtime_ns']
        images_ok &= sha(p) == image['sha256']
    manifest = list(load_rows(ROOT/'overlays/manifest.jsonl'))
    keys = {(r['pair_id'],r['method'],r['camera']) for r in manifest}
    assert len(keys) == len(manifest) == 10384
    expected = {(r['pair_id'], m, c) for r in load_rows(ROOT/'data/render_frames.jsonl')
                for m in METHODS for c in CAMERAS}
    assert keys == expected
    for row in manifest:
        p = ROOT/row['path']
        assert sha(p) == row['sha256']
        image = cv2.imread(str(p))
        assert image is not None and image.shape == (604, 640, 3)
    assert all(checks.values()) and images_ok
    dump(ROOT/'validation/final_verification.json', {'passed':True,
        'source_metadata_and_sha256_unchanged': checks,
        'original_RGB_files_verified_unchanged':len(images),
        'output_images_decoded_and_sha256_verified':len(manifest),
        'missing_or_duplicate_pose_observations':0,
        'preservation_scope':'All 7 source files read by preparation and all 5192 original RGB files; unrelated datasets and weights were not rehashed.'})
    print('PASS: 10384 images decoded; 5192 source images and 7 source files unchanged.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare')
    p=sub.add_parser('render');p.add_argument('--output',default='overlays');p.add_argument('--workers',type=int,default=8);p.add_argument('--limit',type=int)
    sub.add_parser('verify')
    args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='render':render(args)
    else:verify()


if __name__=='__main__':
    main()
