"""Successful-pose-only evaluation of saved V2 per-camera measurements.

Preserves original prediction and evaluation files. Does not run inference.
Aggregation primitives used by the full-dataset evaluation CLI.
"""
import argparse
import csv
import json
from pathlib import Path

from pairwise.runtime import configure_runtime, output_path, sha256, write_json
configure_runtime()

import numpy as np
from pairwise.evaluation import (METHODS, DUAL_METHODS, MATCHING_MONO, _aggregate,
    _camera, _inside, _curve, _finite_summary, ADD_THRESHOLDS_MM, PCK_THRESHOLDS_PX)

ROOT = Path(__file__).resolve().parent
CAMERAS = ('camera1', 'camera2', 'pooled')
TRACKS = ('raw_voting', 'candidate_selected', 'pose_projected')


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def save_csv(path, rows, empty_columns=()):
    columns = list(dict.fromkeys(k for row in rows for k in row if not k.startswith('_'))) or list(empty_columns)
    with output_path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ('' if v is None else v) for k, v in row.items() if not k.startswith('_')})


def eligible(row):
    if not row['pose_success']:
        return False
    if row['method'] in DUAL_METHODS:
        return row['stereo_refined'] == 1 and row['status'] == 'stereo_refined'
    return True


def aggregate_successful(rows, method, camera):
    attempted = [r for r in rows if r['method'] == method and (camera == 'pooled' or r['camera'] == camera)]
    selected = [r for r in attempted if eligible(r)]
    baseline = [r for r in rows if r['method'] == MATCHING_MONO.get(method) and eligible(r)]
    summary, curves = _aggregate(selected, method, camera, baseline)
    # _aggregate counts only its input population. Preserve that meaning explicitly,
    # and report success coverage against the unfiltered attempts separately.
    summary['n_evaluated'] = summary.pop('n_total')
    summary['n_attempted'] = len(attempted)
    summary['n_excluded_failed_pose'] = len(attempted) - len(selected)
    summary['n_attempted_unique_pairs'] = len({r['pair_id'] for r in attempted})
    summary['n_successful_unique_pairs'] = len({r['pair_id'] for r in selected})
    summary['pose_success_rate'] = len(selected) / len(attempted) if attempted else None
    summary['stereo_success_rate'] = sum(r['stereo_refined'] for r in selected) / len(attempted) if attempted else None
    summary['evaluation_population'] = 'successful_pose_only'
    summary['ADD_denominator'] = 'n_evaluated'
    summary['PCK_denominator'] = 'valid_reference_keypoints_on_evaluated_poses'
    summary['timing_population'] = 'successful_poses_only'
    if 'candidates' not in method:
        for key in list(summary):
            if key.startswith('candidate_selected_'):
                summary[key] = None
        curves['candidate_selected'] = None
    return summary, curves


def paired_common(rows, method, camera):
    baseline_name = MATCHING_MONO.get(method)
    baseline = {(r['pair_id'], r['camera']): r for r in rows
                if r['method'] == baseline_name and eligible(r)}
    a, b = [], []
    for row in rows:
        if row['method'] != method or not eligible(row) or (camera != 'pooled' and row['camera'] != camera):
            continue
        other = baseline.get((row['pair_id'], row['camera']))
        if other is not None:
            a.append(row)
            b.append(other)
    d = [x['add_mm'] - y['add_mm'] for x, y in zip(a, b)]
    ds = _finite_summary(d)
    result = dict(method=method, baseline=baseline_name, camera=camera, n_common_success=len(a),
        method_ADD_mean_mm=_finite_summary([r['add_mm'] for r in a])['mean'],
        baseline_ADD_mean_mm=_finite_summary([r['add_mm'] for r in b])['mean'],
        paired_ADD_delta_mean_mm=ds['mean'], paired_ADD_delta_median_mm=ds['median'], paired_ADD_delta_p90_mm=ds['p90'],
        n_improved=sum(v < -1e-9 for v in d), n_worsened=sum(v > 1e-9 for v in d),
        population='identical_pair_id_and_camera_successful_in_both_methods')
    for label, values in (('method', a), ('baseline', b)):
        ac, auc = _curve([r['add_mm'] for r in values], ADD_THRESHOLDS_MM)
        result[label + '_ADD_at_50mm'] = float(ac[10]) if values else None
        result[label + '_ADD_AUC'] = auc
        for track in TRACKS:
            e = np.concatenate([r['_' + track + '_errors'] for r in values]) if values else np.array([])
            e = e[~np.isnan(e)]
            result[label + '_' + track + '_PCK_at_5px'] = float((e < 5).mean()) if len(e) else None
    return result


def load_rows(source, masks):
    raw = read_csv(source / 'per_pair_metrics.csv')
    assert len({(r['method'], r['pair_id'], r['camera']) for r in raw}) == len(raw)
    rows = []
    numeric = ('add_mm', 'translation_error_mm', 'rotation_error_deg', 'pair_runtime_ms',
               'relative_rotation_error_deg', 'translation_direction_error_deg', 'baseline_length_error_mm')
    for original in raw:
        r = dict(original)
        for key in numeric:
            r[key] = float(r[key]) if r.get(key) not in ('', None) else None
        for key in ('pose_success', 'stereo_refined'):
            r[key] = int(r[key])
        mask = masks[r['pair_id'], r['camera']]
        assert r['evaluation_track'] == 'main'
        for track in TRACKS:
            a = np.array([float(r[f'{track}_error_link{k}_px']) if r[f'{track}_error_link{k}_px'] != '' else np.nan for k in range(7)])
            a[mask & ~np.isfinite(a)] = np.inf
            a[~mask] = np.nan
            assert int(mask.sum()) == int(r[track + '_reference_points'])
            r['_' + track + '_errors'] = a
        assert r['pose_success'] == int(r['add_mm'] is not None)
        if r['method'] in DUAL_METHODS:
            assert bool(r['pose_success']) == eligible(r)
        if eligible(r):
            assert np.isfinite([r[k] for k in numeric[:3]]).all()
        rows.append(r)
    return rows


def plots(curves, summaries, out, backend, split):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    color = dict(zip(METHODS, ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple')))
    for camera in CAMERAS:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
        for method in METHODS:
            row = next(s for s in summaries if s['method'] == method and s['camera'] == camera)
            label = f"{method} ({row['n_evaluated']}/{row['n_attempted']})"
            for ax, track in zip(axes, ('ADD', 'candidate_selected', 'pose_projected')):
                c = curves[method][camera][track]
                if c is not None and row['n_evaluated']:
                    ax.plot(c['thresholds'], np.array(c['accuracy']) * 100, color=color[method], label=label)
        for ax, title, unit in zip(axes, ('ADD on successful poses', 'Selected-candidate PCK', 'Pose-projected PCK'), ('mm', 'px', 'px')):
            ax.set(title=title, xlabel=f'Threshold ({unit}), strict <', ylabel='Accuracy (%)', ylim=(0, 102))
            ax.grid(alpha=.25)
        axes[0].legend(fontsize=6.5)
        fig.suptitle(f'{backend} / {split} / {camera}: successful poses only; annotation-reference agreement', fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f'{camera}_curves.png', dpi=150)
        plt.close(fig)

