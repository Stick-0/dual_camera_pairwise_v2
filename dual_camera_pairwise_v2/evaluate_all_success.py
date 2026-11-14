"""Evaluate all attempted pairs using the existing successful-pose-only policy.

No temporal dev/final subset is generated. Each dataset supplies its own pair
manifest, so the original 889 reference masks are never reused for 2596.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
EXT = HERE
EXPERIMENT = EXT
sys.path.insert(0, str(EXT))
sys.dont_write_bytecode = True
from pairwise import runtime
runtime.configure_runtime()

from evaluate_success_pose_only import (CAMERAS, TRACKS, METHODS, MATCHING_MONO,
    aggregate_successful, paired_common, eligible, load_rows, save_csv, plots)
from pairwise.evaluation import _aggregate, _camera, _inside
import numpy as np


def summarize_all(rows, backend, dataset):
    summaries, curves, paired, checks = [], {}, [], []
    for method in METHODS:
        curves[method] = {}
        for camera in CAMERAS:
            summary, curve = aggregate_successful(rows, method, camera)
            summary.update(dataset=dataset, backend=backend, split='all')
            summaries.append(summary)
            curves[method][camera] = curve
            if method in MATCHING_MONO:
                entry = paired_common(rows, method, camera)
                entry.update(dataset=dataset, backend=backend, split='all')
                paired.append(entry)
            original, _ = _aggregate(rows, method, camera, [])
            assert summary['n_evaluated'] == original['n_pose_success']
            assert summary['n_attempted'] == original['n_total']
            if summary['n_evaluated']:
                factor = summary['n_attempted'] / summary['n_evaluated']
                assert abs(summary['conditional_ADD_mean'] - original['conditional_ADD_mean']) < 1e-9
                for metric in ('ADD_success_rate', 'ADD_AUC'):
                    assert abs(summary[metric] - original[metric] * factor) < 1e-10
            selected = [r for r in rows if r['method'] == method and eligible(r)
                        and (camera == 'pooled' or r['camera'] == camera)]
            pck_checks = 0
            for track in TRACKS:
                if track == 'candidate_selected' and 'candidates' not in method:
                    continue
                errors = np.concatenate([r['_' + track + '_errors'] for r in selected]) if selected else np.array([])
                errors = errors[~np.isnan(errors)]
                assert summary[track + '_reference_points'] == len(errors)
                for threshold in (2, 5, 10, 20):
                    expected = float((errors < threshold).mean()) if len(errors) else None
                    actual = summary[f'{track}_PCK_at_{threshold}px']
                    assert actual is None if expected is None else abs(actual - expected) < 1e-12
                    pck_checks += 1
            checks.append(dict(dataset=dataset, backend=backend, method=method,
                camera=camera, n_attempted=summary['n_attempted'],
                n_evaluated=summary['n_evaluated'], pck_checks=pck_checks, passed=True))
    return summaries, curves, paired, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--dataset', default='UR_dual_2596')
    parser.add_argument('--source', action='append', required=True, metavar='BACKEND=EVALUATION_DIR')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = runtime.output_path(args.output)
    assert out.is_relative_to(EXT), 'Output must remain in this extension.'
    if out.exists():
        raise FileExistsError(out)
    pairs_path = args.pairs.resolve()
    pair_rows = list(map(json.loads, pairs_path.open()))
    pairs = {str(r['pair_id']): r for r in pair_rows}
    assert len(pair_rows) == len(pairs), 'Duplicate pair ID in input manifest.'
    expected_pairs = {'UR_dual_2596': 2596, 'UR_dual_889': 889}.get(args.dataset)
    if expected_pairs is not None:
        assert len(pairs) == expected_pairs, 'Dataset manifest is incomplete for full-sequence evaluation.'
    policy_path = EXT / 'configs/evaluation_success_pose_only.json'
    policy = json.loads(policy_path.read_text())
    hashes = {str(p.resolve()): runtime.sha256(p) for p in [pairs_path, policy_path,
              Path(__file__), EXT / 'evaluate_success_pose_only.py', EXT / 'pairwise/evaluation.py']}
    all_summary, checks, failure_counts = [], [], {}
    out.mkdir(parents=True, exist_ok=False)
    seen = set()
    for item in args.source:
        backend, source = item.split('=', 1)
        assert backend in ('roma_v2', 'loftr_indoor_ds') and backend not in seen
        seen.add(backend)
        source = Path(source).resolve()
        meta_path = source / 'evaluation.json'
        meta = json.loads(meta_path.read_text())
        ref_path = Path(meta['reference_manifest_path'])
        assert runtime.sha256(ref_path) == meta['reference_manifest_sha256']
        for path in [meta_path, source / 'per_pair_metrics.csv', ref_path]:
            hashes[str(path)] = runtime.sha256(path)
        reference_rows = list(map(json.loads, ref_path.open()))
        refs = {str(r['pair_id']): r for r in reference_rows}
        assert len(reference_rows) == len(refs), 'Duplicate pair ID in reference manifest.'
        assert set(refs) == set(pairs), 'Full evaluation requires every pair reference.'
        masks = {(pid, camera): _inside(np.asarray(ref['cameras'][camera]['projected_keypoints_uv']),
                                    _camera(pairs[pid]['cameras'][camera]))
                 for pid, ref in refs.items() for camera in CAMERAS[:2]}
        rows = load_rows(source, masks)
        expected_keys = {(method, pid, camera) for method in METHODS for pid in pairs for camera in CAMERAS[:2]}
        assert {(r['method'], r['pair_id'], r['camera']) for r in rows} == expected_keys
        assert meta['n_pairs'] == len(pairs)
        # Keep original source split labels available only as provenance. All
        # current output rows have one full-sequence evaluation population.
        for row in rows:
            row['source_split'] = row.get('split', '')
            row['split'] = 'all'
            row['dataset'] = args.dataset
            row['backend'] = backend
        summaries, curves, paired, group_checks = summarize_all(rows, backend, args.dataset)
        checks.extend(group_checks)
        directory = out / f'{backend}_all'
        directory.mkdir()
        selected, excluded = [r for r in rows if eligible(r)], [r for r in rows if not eligible(r)]
        failure_counts[backend] = dict(Counter(f"{r['method']} / {r.get('reason', r['status'])}" for r in excluded if r['camera'] == 'camera1'))
        save_csv(directory / 'summary_metrics.csv', summaries)
        save_csv(directory / 'per_pair_metrics.csv', selected, ('pair_id', 'method', 'camera'))
        save_csv(directory / 'excluded_failed_poses.csv', excluded, ('pair_id', 'method', 'camera', 'reason'))
        save_csv(directory / 'paired_comparisons.csv', paired)
        runtime.write_json(directory / 'curves.json', curves)
        runtime.write_json(directory / 'evaluation.json', dict(policy=policy, source_evaluation=str(source),
            dataset=args.dataset, backend=backend, split='all', n_dataset_pairs=len(pairs),
            summary=summaries, paired_comparisons=paired))
        plots(curves, summaries, directory, backend, f'{args.dataset} / all {len(pairs)} pairs')
        all_summary.extend(summaries)
    assert all(runtime.sha256(path) == digest for path, digest in hashes.items())
    save_csv(out / 'summary_metrics.csv', all_summary)
    runtime.write_json(out / 'validation.json', dict(policy=policy, dataset=args.dataset,
        split='all', n_dataset_pairs=len(pairs), summary_groups=len(checks), checks=checks,
        pck_checks=sum(c['pck_checks'] for c in checks), input_hashes_unchanged=True,
        input_sha256=hashes, no_inference_or_training=True, existing_outputs_preserved=True,
        failure_counts_by_method_and_reason=failure_counts))
    print(json.dumps(dict(output=str(out), dataset=args.dataset, n_pairs=len(pairs),
                         summary_groups=len(checks), passed=True)))


if __name__ == '__main__':
    main()
