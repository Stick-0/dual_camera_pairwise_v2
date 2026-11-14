"""Evaluation-only annotation-reference agreement for independent current pairs.

No estimator imports this module.  Only ``prepare_reference`` opens recorder
labels; ``evaluate_pairs`` opens its separate, hashed reference manifest.  Failed
poses remain null, enter threshold denominators, and never enter error means.
"""
from collections import Counter
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .runtime import jsonable, output_path, sha256, write_json

KEYPOINT_NAMES = [f'ur5e_link{i}' for i in range(7)]
METHODS = ['mono_mean', 'mono_candidates', 'dual_mean', 'dual_candidates', 'dual_no_cross']
DUAL_METHODS = {'dual_mean', 'dual_candidates'}
MATCHING_MONO = {'dual_mean': 'mono_mean', 'dual_candidates': 'mono_candidates',
                 'dual_no_cross': 'mono_mean', 'mono_candidates': 'mono_mean'}
ADD_THRESHOLDS_MM = np.arange(0., 100.01, 5.)
PCK_THRESHOLDS_PX = np.arange(0., 20.001, .25)


def _array(value, shape):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return result if result.shape == shape else None


def _pose(value):
    result = _array(value, (4, 4))
    if result is None or not np.isfinite(result).all():
        return None
    R = result[:3, :3]
    if (not np.allclose(result[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(R), 1., atol=1e-4)):
        return None
    return result


def _camera(record):
    camera = record.get('calibration', record)
    return {'K': np.asarray(camera['K'], float),
            'D': np.asarray(camera.get('D', camera.get('distortion', [])), float),
            'width': int(camera.get('width', 640)), 'height': int(camera.get('height', 480))}


def _project(T, xyz, camera):
    result = np.full((7, 2), np.nan)
    if T is None or xyz is None:
        return result
    camera_xyz = np.asarray(xyz) @ T[:3, :3].T + T[:3, 3]
    finite = np.isfinite(camera_xyz).all(axis=1) & (camera_xyz[:, 2] > .001)
    if finite.any():
        result[finite] = cv2.projectPoints(camera_xyz[finite], np.zeros(3), np.zeros(3),
                                          camera['K'], camera['D'])[0].reshape(-1, 2)
    return result


def _inside(uv, camera):
    return (np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < camera['width'])
            & (uv[:, 1] >= 0) & (uv[:, 1] < camera['height']))


def _fk_from_pair(camera_record):
    for key in ('keypoints_base_m', 'FK', 'fk'):
        value = _array(camera_record.get(key), (7, 3))
        if value is not None:
            return value
    return None


def prepare_reference(pairs, data_root, output):
    """Write evaluation-only JSONL; never add references to pair/observation inputs."""
    root = Path(data_root).resolve()
    index = {}
    with (root / 'frames.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            key = str(row['sample_id'])
            if key in index:
                raise ValueError(f'Duplicate dataset pair: {key}')
            index[key] = row
    records = []
    max_fk_difference = 0.
    max_projection_difference = 0.
    for pair in pairs:
        pair_id = str(pair['pair_id'])
        source = index[pair_id]
        cameras = {}
        for cid, camera_record in pair['cameras'].items():
            annotation = (root / source['cameras'][cid]['gt']).resolve()
            if not annotation.is_relative_to(root):
                raise ValueError('Reference path escapes dataset')
            obj_list = [obj for obj in json.loads(annotation.read_text())['objects']
                        if obj.get('class') == 'ur5e']
            if len(obj_list) != 1:
                raise ValueError(f'Expected one ur5e object: {annotation}')
            obj = obj_list[0]
            points = {p['name']: p for p in obj['keypoints']}
            if len(points) != len(obj['keypoints']):
                raise ValueError(f'Duplicate reference keypoints: {annotation}')
            xyz = np.array([points[name]['location'] for name in KEYPOINT_NAMES], float)
            uv = np.array([points[name]['projected_location'] for name in KEYPOINT_NAMES], float)
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat(obj['xyzw_cameraframe']).as_matrix()
            T[:3, 3] = obj['location_cameraframe']
            if _pose(T) is None or not np.isfinite(xyz).all():
                raise ValueError(f'Malformed reference pose/3D points: {annotation}')
            fk = _fk_from_pair(camera_record)
            fk_error = None if fk is None else float(np.max(np.abs(fk - xyz)))
            if fk_error is not None:
                max_fk_difference = max(max_fk_difference, fk_error)
                if fk_error > 1e-7:
                    raise ValueError(f'Annotation location differs from pair FK: {pair_id}/{cid}')
            projection = _project(T, xyz, _camera(camera_record))
            finite = np.isfinite(projection).all(axis=1) & np.isfinite(uv).all(axis=1)
            projection_error = float(np.max(np.linalg.norm(projection[finite] - uv[finite], axis=1))) if finite.any() else None
            if projection_error is not None:
                max_projection_difference = max(max_projection_difference, projection_error)
            cameras[cid] = {'T_camera_from_base': T, 'keypoints_base_m': xyz,
                            'projected_keypoints_uv': uv, 'annotation_path': str(annotation),
                            'annotation_sha256': sha256(annotation),
                            'pair_fk_max_absolute_difference_m': fk_error,
                            'reference_projection_max_difference_px': projection_error}
        G = cameras['camera2']['T_camera_from_base'] @ np.linalg.inv(cameras['camera1']['T_camera_from_base'])
        records.append({'schema_version': 2, 'role': 'evaluation_only', 'pair_id': pair_id,
                        'split': pair['split'], 'reference_is_independent_gt': False,
                        'cameras': cameras, 'T_C2_C1_reference': G})
    output = output_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        for record in records:
            stream.write(json.dumps(jsonable(record), allow_nan=False) + '\n')
    return {'path': str(output), 'sha256': sha256(output), 'n_pairs': len(records),
            'role': 'evaluation_only', 'reference_is_independent_gt': False,
            'max_fk_absolute_difference_m': max_fk_difference,
            'max_reference_projection_difference_px': max_projection_difference}


def _write_csv(path, rows, columns=None):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ('' if v is None else v) for k, v in row.items()})
    return str(path)


def _finite_summary(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    return {'count': int(len(values)),
            'mean': float(np.mean(values)) if len(values) else None,
            'median': float(np.median(values)) if len(values) else None,
            'p90': float(np.percentile(values, 90)) if len(values) else None}


def _curve(values, thresholds):
    values = np.asarray([np.inf if v is None else v for v in values], float)
    curve = np.mean(values[:, None] < thresholds[None, :], axis=0) if len(values) else np.full(len(thresholds), np.nan)
    auc = float(np.trapezoid(curve, thresholds) / (thresholds[-1] - thresholds[0])) if len(values) else None
    return curve, auc


def _pck_errors(uv, reference, camera, valid=None):
    reference_valid = _inside(reference, camera)
    result = np.full(7, np.nan)
    result[reference_valid] = np.inf
    uv = _array(uv, (7, 2))
    if uv is not None:
        prediction_valid = np.isfinite(uv).all(axis=1)
        if valid is not None:
            prediction_valid &= np.asarray(valid, bool)
        selected = reference_valid & prediction_valid
        result[selected] = np.linalg.norm(uv[selected] - reference[selected], axis=1)
    return result


def _rotation_error(R, reference_R):
    return float(np.degrees(Rotation.from_matrix(R @ reference_R.T).magnitude()))


def _pose_metrics(T, reference, xyz):
    if T is None:
        return {'add_mm': None, 'translation_error_mm': None, 'rotation_error_deg': None}
    errors = (xyz @ T[:3, :3].T + T[:3, 3]) - (xyz @ reference[:3, :3].T + reference[:3, 3])
    return {'add_mm': float(np.mean(np.linalg.norm(errors, axis=1)) * 1000),
            'translation_error_mm': float(np.linalg.norm(T[:3, 3] - reference[:3, 3]) * 1000),
            'rotation_error_deg': _rotation_error(T[:3, :3], reference[:3, :3])}


def _relative_metrics(G, reference):
    result = {'relative_rotation_error_deg': None, 'translation_direction_error_deg': None,
              'baseline_length_error_mm': None, 'baseline_length_mm': None}
    if G is None:
        return result
    baseline, ref_baseline = np.linalg.norm(G[:3, 3]), np.linalg.norm(reference[:3, 3])
    result.update(relative_rotation_error_deg=_rotation_error(G[:3, :3], reference[:3, :3]),
                  baseline_length_error_mm=float(abs(baseline - ref_baseline) * 1000),
                  baseline_length_mm=float(baseline * 1000))
    if min(baseline, ref_baseline) > 1e-12:
        result['translation_direction_error_deg'] = float(np.degrees(np.arccos(np.clip(
            np.dot(G[:3, 3], reference[:3, 3]) / (baseline * ref_baseline), -1, 1))))
    return result


def _load_results(run_directory):
    directory = Path(run_directory).resolve()
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    results = {}
    rows = manifest.get('results', [])
    if isinstance(rows, dict):
        rows = [{'pair_id': key, 'result_path': value} for key, value in rows.items()]
    if rows:
        paths = []
        for row in rows:
            path = Path(row.get('result_path', row.get('path', '')))
            if not path.is_absolute():
                path = directory / path
            if row.get('sha256') and sha256(path) != row['sha256']:
                raise ValueError(f'Result hash mismatch: {path}')
            paths.append(path)
    else:
        paths = sorted((directory / 'results').glob('*.json'))
    for path in paths:
        result = json.loads(path.read_text())
        pair_id = str(result['pair_id'])
        if pair_id in results:
            raise ValueError(f'Duplicate result pair: {pair_id}')
        results[pair_id] = result
    return manifest, results


def _load_observations(manifest, results):
    """Read tiny exported raw fields, or use new bridge's validated immutable cache."""
    embedded = {(pid, cid): obs for pid, result in results.items()
                for cid, obs in result.get('observations', {}).items()}
    if embedded:
        return embedded
    path = manifest.get('observations_manifest_path', manifest.get('observations'))
    if isinstance(path, dict):
        path = path.get('path')
    if path:
        # The bridge has no reference input and performs its own cache hash checks.
        from .legacy_bridge import load_observations
        loaded = load_observations(path)
        if isinstance(loaded, tuple):
            loaded = loaded[-1]
        if isinstance(loaded, dict):
            return loaded
        return {(str(o['frame_id']), o['camera_id']): o for o in loaded}
    return {}


def _quality_fields(quality):
    fields = {}
    for key in ('actual_rgb_correspondences_used', 'rgb_matches_used', 'matches_count',
                'geometric_inliers', 'parallax_deg', 'median_parallax_deg',
                'rgb_reprojection_rmse_px', 'rgb_reprojection_median_px',
                'positive_depth_fraction', 'reduced_pose_condition_number',
                'effective_rgb_correspondences', 'rgb_median_error_px', 'rgb_p90_error_px',
                'parallax_median_deg', 'baseline_m', 'reduced_rank', 'reduced_condition'):
        value = quality.get(key)
        if isinstance(value, (int, float)):
            fields[key] = value
    return fields


def _method_runtime_ms(method, item, result):
    """Method-specific cached-input cost; shared matcher cost is charged to dual."""
    solver_ms = item.get('timing', {}).get('total_ms')
    if solver_ms is None:
        return None
    if method not in DUAL_METHODS:
        return solver_ms
    shared = result.get('timing', {})
    matcher_ms = shared.get('matcher_ms', shared.get('matching_runtime_ms'))
    return None if matcher_ms is None else solver_ms + matcher_ms


def _row_metrics(pair, cid, method, result, reference, observation, fallback=False):
    item = result.get('methods', {}).get(method, {})
    stereo = method in DUAL_METHODS and item.get('status') == 'stereo_refined' and item.get('stereo_pose') is not None
    if stereo:
        if any(_pose(item['stereo_pose'].get(camera)) is None for camera in ('camera1', 'camera2')):
            raise ValueError(f"Invalid stereo success pose: {pair['pair_id']}/{method}")
        actual_matches = item.get('quality', {}).get('actual_rgb_correspondences_used')
        if actual_matches is None or actual_matches <= 0:
            raise ValueError(f"Stereo success lacks actual RGB factor evidence: {pair['pair_id']}/{method}")
    if fallback:
        T = _pose((item.get('fallback_pose') or {}).get(cid))
    elif method in DUAL_METHODS:
        T = _pose((item.get('stereo_pose') or {}).get(cid)) if stereo else None
    else:
        T = _pose((item.get('poses') or {}).get(cid))
    camera = _camera(pair['cameras'][cid])
    ref = reference['cameras'][cid]
    ref_T = _pose(ref['T_camera_from_base'])
    xyz = _array(ref['keypoints_base_m'], (7, 3))
    ref_uv = np.asarray(ref['projected_keypoints_uv'], float)
    fk = _array(observation.get('keypoints_base_m'), (7, 3))
    if fk is None:
        fk = _fk_from_pair(pair['cameras'][cid])
    if fk is None:
        raise ValueError(f"Missing estimation FK for projection: {pair['pair_id']}/{cid}")
    if not np.allclose(fk, xyz, rtol=0, atol=1e-7):
        raise ValueError('Reference keypoint location is not the declared pair FK')
    raw = _pck_errors(observation.get('keypoints_uv', observation.get('raw_uv')),
                      ref_uv, camera, observation.get('valid_keypoints', observation.get('raw_valid')))
    selected = _pck_errors((item.get('candidate_selected_uv') or {}).get(cid), ref_uv, camera,
                           (item.get('candidate_selected_valid') or {}).get(cid))
    projection = _project(T, fk, camera)
    projected = _pck_errors(projection, ref_uv, camera)
    if method in DUAL_METHODS and not stereo and not fallback:
        # A failed joint result cannot gain candidate-selection credit from fallback.
        selected[~np.isnan(selected)] = np.inf
    q = (item.get('mono_quality') or {}).get(cid, {})
    # A joint rejection and its monocular initialization have different causes.
    # Never replace the joint quality failure with an earlier camera-only reason.
    reason = (q.get('reason') if T is None and q.get('reason')
              and (method not in DUAL_METHODS or fallback) else item.get('reason'))
    row = {'pair_id': str(pair['pair_id']), 'split': pair['split'], 'method': method,
           'camera': cid, 'evaluation_track': 'operational_fallback' if fallback else 'main',
           'status': item.get('status', 'not_executed'), 'reason': reason,
           'monocular_initialization_reason': q.get('reason'),
           'pose_success': int(T is not None), 'stereo_refined': int(stereo and not fallback and T is not None),
           'reference_role': 'annotation-reference agreement',
           **_pose_metrics(T, ref_T, xyz)}
    relative = item.get('stereo_pose') or {}
    G = _pose(relative.get('T_C2_C1')) if stereo and not fallback else None
    if G is None and stereo and not fallback:
        T1, T2 = _pose(relative.get('camera1')), _pose(relative.get('camera2'))
        if T1 is not None and T2 is not None:
            G = T2 @ np.linalg.inv(T1)
    row.update(_relative_metrics(G, np.asarray(reference['T_C2_C1_reference'], float)))
    row.update(_quality_fields(item.get('quality', {})))
    timing = item.get('timing', {})
    row['pair_runtime_ms'] = _method_runtime_ms(method, item, result)
    row['solver_runtime_ms'] = timing.get('total_ms')
    for name, values in (('raw_voting', raw), ('candidate_selected', selected), ('pose_projected', projected)):
        valid_reference = ~np.isnan(values)
        row[name + '_reference_points'] = int(valid_reference.sum())
        for threshold in (2, 5, 10, 20):
            row[f'{name}_PCK_at_{threshold}px'] = float(np.mean(values[valid_reference] < threshold)) if valid_reference.any() else None
        for k, error in enumerate(values):
            row[f'{name}_error_link{k}_px'] = float(error) if np.isfinite(error) else None
        # Private arrays retain +inf failures and NaN excluded references until aggregation.
        row['_' + name + '_errors'] = values
    return row


def _aggregate(rows, method, camera, paired_rows):
    selected = [r for r in rows if r['method'] == method and (camera == 'pooled' or r['camera'] == camera)]
    by_key = {(r['pair_id'], r['camera']): r for r in paired_rows}
    deltas = []
    for row in selected:
        other = by_key.get((row['pair_id'], row['camera']))
        if row['add_mm'] is not None and other is not None and other['add_mm'] is not None:
            deltas.append(row['add_mm'] - other['add_mm'])
    add = _finite_summary([r['add_mm'] for r in selected])
    delta = _finite_summary(deltas)
    add_curve, add_auc = _curve([r['add_mm'] for r in selected], ADD_THRESHOLDS_MM)
    summary = {'method': method, 'camera': camera, 'n_total': len(selected),
               'n_unique_pairs': len({r['pair_id'] for r in selected}),
               'n_pose_success': sum(r['pose_success'] for r in selected),
               'n_stereo_refined': sum(r['stereo_refined'] for r in selected),
               'n_unique_stereo_pairs': len({r['pair_id'] for r in selected if r['stereo_refined']}),
               'ADD_success_rate': float(add_curve[10]) if len(selected) else None,
               'ADD_success_threshold_mm': 50, 'ADD_AUC': add_auc, 'ADD_AUC_range_mm': 100,
               'conditional_ADD_mean': add['mean'], 'conditional_ADD_median': add['median'],
               'conditional_ADD_p90': add['p90'], 'conditional_ADD_unit': 'mm',
               'matching_mono': MATCHING_MONO.get(method),
               'common_success_count': delta['count'], 'paired_delta_vs_matching_mono': delta['mean'],
               'paired_delta_median_mm': delta['median'], 'paired_delta_p90_mm': delta['p90'],
               'paired_delta_definition': 'method_minus_matching_mono_mm_on_common_success_camera_results',
               'translation_error': _finite_summary([r['translation_error_mm'] for r in selected])['mean'],
               'translation_error_unit': 'mm_conditional_success',
               'rotation_error': _finite_summary([r['rotation_error_deg'] for r in selected])['mean'],
               'rotation_error_unit': 'deg_conditional_success',
               'pair_runtime_ms': _finite_summary([r['pair_runtime_ms'] for r in selected])['mean'],
               'reference_role': 'annotation-reference agreement',
               'reference_is_independent_gt': False}
    curves = {'ADD': {'thresholds': ADD_THRESHOLDS_MM.tolist(), 'accuracy': add_curve.tolist()}}
    for threshold in (20, 50, 100):
        summary[f'ADD_success_at_{threshold}mm'] = float(add_curve[int(threshold / 5)]) if len(selected) else None
    for name in ('raw_voting', 'candidate_selected', 'pose_projected'):
        errors = np.concatenate([r['_' + name + '_errors'] for r in selected]) if selected else np.array([])
        errors = errors[~np.isnan(errors)]
        curve, auc = _curve(errors, PCK_THRESHOLDS_PX)
        summary[name + '_reference_points'] = int(len(errors))
        summary[name + '_PCK_AUC'] = auc
        for threshold in (2, 5, 10, 20):
            summary[f'{name}_PCK_at_{threshold}px'] = float(curve[int(threshold * 4)]) if len(errors) else None
        curves[name] = {'thresholds': PCK_THRESHOLDS_PX.tolist(), 'accuracy': curve.tolist()}
    summary['candidate_PCK'] = summary['candidate_selected_PCK_at_5px'] if 'candidates' in method else None
    summary['candidate_PCK_reason'] = None if 'candidates' in method else 'mean_method_has_no_candidate_selection'
    summary['pose_projected_PCK'] = summary['pose_projected_PCK_at_5px']
    summary['PCK_threshold_px'] = 5
    summary['pose_success_rate'] = summary['n_pose_success'] / len(selected) if selected else None
    summary['stereo_success_rate'] = summary['n_stereo_refined'] / len(selected) if selected else None
    summary['conditional_statistics_unavailable_reason'] = None if add['count'] else 'no_successful_poses'
    summary['paired_statistics_unavailable_reason'] = None if delta['count'] else 'no_common_success_results'
    summary['execution_status_counts'] = json.dumps(dict(Counter(r['status'] for r in selected)), sort_keys=True)
    unique = {r['pair_id']: r for r in selected}
    for key in ('relative_rotation_error_deg', 'translation_direction_error_deg', 'baseline_length_error_mm'):
        stats = _finite_summary([r[key] for r in unique.values()])
        for suffix, value in stats.items():
            summary[key + '_' + suffix] = value
    return summary, curves


def _timing_rows(pairs, results):
    rows = []
    for pair in pairs:
        result = results.get(str(pair['pair_id']), {})
        shared = result.get('timing', {})
        for method, item in result.get('methods', {}).items():
            timing = item.get('timing', {})
            row = {'pair_id': str(pair['pair_id']), 'split': pair['split'], 'method': method,
                   'robot_forward_ms': shared.get('robot_forward_ms'), 'voting_ms': shared.get('voting_ms'),
                   'matcher_ms': shared.get('matcher_ms', shared.get('matching_runtime_ms')),
                   'initialization_ms': timing.get('initialization_ms'), 'ba_ms': timing.get('ba_ms'),
                   'solver_total_ms': timing.get('total_ms'), 'pair_all_methods_wall_ms': shared.get('total_ms'),
                   'pair_runtime_ms': _method_runtime_ms(method, item, result),
                   'robot_observations_reused': shared.get('robot_observations_reused'),
                   'source_robot_forward_ms': shared.get('source_robot_forward_ms'),
                   'source_voting_ms': shared.get('source_voting_ms'),
                   'timing_note': shared.get('note', 'Method runtime = solver plus matcher for dual, solver only for mono/no_cross. Robot forward/vote zero with robot_observations_reused=true means no execution in this cached run; source timings remain separate. Null means unmeasured. Matcher time may be reused from its recorded execution; all-method wall time is not a per-method runtime.')}
            rows.append(row)
    return rows


def _comparison_rows(rows, summaries):
    comparisons = [('dual_mean', 'mono_mean', 'stereo_added_mean'),
                   ('dual_candidates', 'mono_candidates', 'stereo_added_candidates'),
                   ('mono_candidates', 'mono_mean', 'candidate_processing_monocular'),
                   ('dual_candidates', 'dual_mean', 'candidate_processing_stereo'),
                   ('dual_no_cross', 'mono_mean', 'no_cross_control')]
    indexed = {(r['method'], r['pair_id'], r['camera']): r for r in rows}
    summary_index = {(r['method'], r['camera']): r for r in summaries}
    result = []
    for method, baseline, meaning in comparisons:
        for camera in ('camera1', 'camera2', 'pooled'):
            deltas = []
            for row in rows:
                if row['method'] != method or (camera != 'pooled' and row['camera'] != camera):
                    continue
                other = indexed.get((baseline, row['pair_id'], row['camera']))
                if other and row['add_mm'] is not None and other['add_mm'] is not None:
                    deltas.append(row['add_mm'] - other['add_mm'])
            stats = _finite_summary(deltas)
            a, b = summary_index.get((method, camera)), summary_index.get((baseline, camera))
            result.append({'method': method, 'baseline': baseline, 'camera': camera, 'meaning': meaning,
                           'common_success_count': stats['count'], 'conditional_paired_ADD_delta_mean_mm': stats['mean'],
                           'conditional_paired_ADD_delta_median_mm': stats['median'],
                           'conditional_paired_ADD_delta_p90_mm': stats['p90'],
                           'n_improved': sum(d < -1e-9 for d in deltas), 'n_worsened': sum(d > 1e-9 for d in deltas),
                           'ADD_50mm_success_delta_percentage_points': 100 * (a['ADD_success_rate'] - b['ADD_success_rate']) if a and b else None,
                           'ADD_AUC_delta': a['ADD_AUC'] - b['ADD_AUC'] if a and b else None,
                           'reason_if_unavailable': None if deltas else 'no_common_success_results',
                           'reference_role': 'annotation-reference agreement'})
    return result


def _plot_curves(curves, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paths = []
    colors = dict(zip(METHODS, ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple')))
    for camera in ('camera1', 'camera2', 'pooled'):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for method in METHODS:
            item = curves.get(method, {}).get(camera)
            if item is None:
                continue
            for axis, name in zip(axes, ('ADD', 'candidate_selected', 'pose_projected')):
                if name == 'candidate_selected' and 'candidates' not in method:
                    continue
                data = item[name]
                axis.plot(data['thresholds'], data['accuracy'], label=method, color=colors[method])
        for axis, title, unit in zip(axes, ('ADD (failed poses in denominator)', 'Selected candidate PCK', 'Pose-projected PCK'), ('mm', 'px', 'px')):
            axis.set(xlabel=f'Threshold ({unit}), strict <', ylabel='Success fraction', ylim=(0, 1.02), title=title)
            axis.grid(alpha=.25)
        axes[0].legend(fontsize=7, loc='lower right')
        fig.suptitle(f'{camera}: annotation-reference agreement; no independent ground truth', fontsize=10)
        fig.tight_layout()
        path = output_path(Path(output_dir) / f'{camera}_curves.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))
    return paths


def _representative_overlays(pairs, rows, results, observations, output_dir):
    """Select improved/worsened/failure examples by declared reference ADD only.

    Selection is evaluation-only. No overlay or chosen example is an estimator
    input. Candidates remain orange source locations; selected modes and pose
    projections have different symbols. Fallback projections are not drawn as
    a stereo result.
    """
    pair_index = {str(p['pair_id']): p for p in pairs}
    row_index = {(r['pair_id'], r['method'], r['camera']): r for r in rows}
    entries = []
    for method, baseline in (('dual_mean', 'mono_mean'), ('dual_candidates', 'mono_candidates')):
        differences = []
        failed = []
        for pair_id in pair_index:
            per_camera = []
            for cid in ('camera1', 'camera2'):
                a, b = row_index.get((pair_id, method, cid)), row_index.get((pair_id, baseline, cid))
                if a is not None and b is not None and a['add_mm'] is not None and b['add_mm'] is not None:
                    per_camera.append(a['add_mm'] - b['add_mm'])
            if len(per_camera) == 2:
                differences.append((float(np.mean(per_camera)), pair_id))
            if any(not (row_index.get((pair_id, method, cid)) or {}).get('pose_success') for cid in ('camera1', 'camera2')):
                failed.append(pair_id)
        selections = {'improved': min((v for v in differences if v[0] < -1e-9), default=None),
                      'worsened': max((v for v in differences if v[0] > 1e-9), default=None),
                      'failure': (None, sorted(failed)[0]) if failed else None}
        for category, selection in selections.items():
            if selection is None:
                entries.append({'method': method, 'category': category, 'path': None,
                                'reason': f'No {category} pair under the declared comparison'})
                continue
            difference, pair_id = selection
            pair, result = pair_index[pair_id], results.get(pair_id, {})
            item = result.get('methods', {}).get(method, {})
            panels = []
            for cid in ('camera1', 'camera2'):
                image_path = pair['cameras'][cid].get('image_path')
                image = cv2.imread(str(image_path)) if image_path else None
                if image is None:
                    break
                image = image.copy()
                obs = observations.get((pair_id, cid), observations.get((cid, pair_id), {}))
                raw = _array(obs.get('keypoints_uv', obs.get('raw_uv')), (7, 2))
                valid = np.asarray(obs.get('valid_keypoints', obs.get('raw_valid', np.ones(7))), bool)
                def point(uv):
                    return tuple(np.round(uv).astype(int))
                for kp, hypothesis in enumerate(obs.get('hypotheses', [])):
                    uv = np.asarray(hypothesis.get('uv', []), float)
                    if uv.ndim == 2 and uv.shape[1] == 2:
                        for value in uv[np.isfinite(uv).all(axis=1)]:
                            cv2.circle(image, point(value), 2, (0, 160, 255), -1)
                if raw is not None:
                    for kp, uv in enumerate(raw):
                        if valid[kp] and np.isfinite(uv).all():
                            cv2.circle(image, point(uv), 5, (0, 220, 0), 1)
                            cv2.putText(image, str(kp), tuple(np.array(point(uv)) + [5, -5]),
                                        cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 220, 0), 1, cv2.LINE_AA)
                selected = _array((item.get('candidate_selected_uv') or {}).get(cid), (7, 2))
                selected_valid = np.asarray((item.get('candidate_selected_valid') or {}).get(cid, np.zeros(7)), bool)
                if selected is not None:
                    for kp, uv in enumerate(selected):
                        if selected_valid[kp] and np.isfinite(uv).all():
                            cv2.drawMarker(image, point(uv), (255, 0, 255), cv2.MARKER_DIAMOND, 10, 1)
                stereo = item.get('status') == 'stereo_refined' and item.get('stereo_pose') is not None
                pose = _pose((item.get('stereo_pose') or {}).get(cid)) if stereo else None
                fk = _array(obs.get('keypoints_base_m'), (7, 3))
                if fk is None:
                    fk = _fk_from_pair(pair['cameras'][cid])
                projected = _project(pose, fk, _camera(pair['cameras'][cid]))
                for uv in projected[np.isfinite(projected).all(axis=1)]:
                    cv2.drawMarker(image, point(uv), (255, 255, 0), cv2.MARKER_TILTED_CROSS, 12, 2)
                panel = np.full((image.shape[0] + 180, image.shape[1], 3), 245, np.uint8)
                panel[:image.shape[0]] = image
                metric = row_index[(pair_id, method, cid)]
                other = row_index[(pair_id, baseline, cid)]
                lines = [f'{cid}: {method} / {item.get("status", "not_executed")}',
                         f'ADD mm: dual={metric["add_mm"]!s} mono={other["add_mm"]!s}',
                         'green circle=raw; orange dots=original hypotheses',
                         'magenta diamond=selected mode; cyan x=stereo projection']
                posterior = (item.get('posterior') or {}).get(cid) or []
                for kp, data in enumerate(posterior):
                    if isinstance(data, dict):
                        entropy = data.get('entropy')
                        outlier = data.get('outlier_responsibility')
                        if entropy is not None and outlier is not None:
                            lines.append(f'k{kp}: entropy={entropy:.3f}, outlier={outlier:.3f}')
                # Two columns for per-keypoint posterior diagnostics.
                for index, line in enumerate(lines[:11]):
                    x = 8 if index < 8 else 320
                    y = image.shape[0] + 19 + (index if index < 8 else index - 4) * 19
                    cv2.putText(panel, line[:95], (x, y), cv2.FONT_HERSHEY_SIMPLEX, .40, (25, 25, 25), 1, cv2.LINE_AA)
                panels.append(panel)
            if len(panels) != 2:
                entries.append({'method': method, 'category': category, 'pair_id': pair_id,
                                'path': None, 'reason': 'RGB image missing for overlay'})
                continue
            height = max(p.shape[0] for p in panels)
            padded = [cv2.copyMakeBorder(p, 0, height - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245)) for p in panels]
            canvas = np.concatenate(padded, axis=1)
            path = output_path(Path(output_dir) / f'{method}_{category}_{pair_id}.png')
            if not cv2.imwrite(str(path), canvas):
                raise IOError(f'Cannot write overlay: {path}')
            entries.append({'method': method, 'category': category, 'pair_id': pair_id, 'path': str(path),
                            'paired_mean_ADD_delta_mm': difference,
                            'selection': 'extreme signed mean of two camera ADD differences on common successful poses; first ID for failure',
                            'reference_role': 'evaluation-only representative selection'})
    write_json(Path(output_dir) / 'representative_overlays.json', entries)
    return entries


def evaluate_pairs(pairs, reference_manifest_path, run_directory, output_dir, split='final'):
    """Keep every selected pair, distinguish stereo from operational fallbacks."""
    chosen = [p for p in pairs if split == 'all' or p['split'] == split]
    if not chosen:
        raise ValueError(f'No pairs in evaluation split {split}')
    with Path(reference_manifest_path).open() as stream:
        reference_records = list(map(json.loads, stream))
    refs = {str(row['pair_id']): row for row in reference_records}
    if len(refs) != len(reference_records):
        raise ValueError('Duplicate evaluation reference pair')
    manifest, results = _load_results(run_directory)
    observations = _load_observations(manifest, results)
    methods = list(dict.fromkeys(METHODS + [method for result in results.values() for method in result.get('methods', {})]))
    rows, fallbacks = [], []
    for pair in chosen:
        pair_id = str(pair['pair_id'])
        ref = refs[pair_id]
        if ref.get('role') != 'evaluation_only' or ref.get('split') != pair['split']:
            raise ValueError(f'Reference role/split mismatch: {pair_id}')
        result = results.get(pair_id, {'pair_id': pair_id, 'methods': {}})
        for method in methods:
            for cid in pair['cameras']:
                obs = observations.get((pair_id, cid), observations.get((cid, pair_id), {}))
                rows.append(_row_metrics(pair, cid, method, result, ref, obs))
                if method in DUAL_METHODS:
                    fallbacks.append(_row_metrics(pair, cid, method, result, ref, obs, fallback=True))
    summaries, fallback_summaries, curves = [], [], {}
    for method in methods:
        curves[method] = {}
        for cid in ('camera1', 'camera2', 'pooled'):
            matching = [r for r in rows if r['method'] == MATCHING_MONO.get(method)]
            summary, curve = _aggregate(rows, method, cid, matching)
            summaries.append(summary)
            curves[method][cid] = curve
            if method in DUAL_METHODS:
                fsummary, _ = _aggregate(fallbacks, method, cid, matching)
                fsummary['evaluation_track'] = 'operational_fallback_only'
                fallback_summaries.append(fsummary)
    comparisons = _comparison_rows(rows, summaries)
    failures = Counter((r['method'], r['camera'], r['status'], str(r['reason'] or 'unspecified')) for r in rows if not r['pose_success'])
    failure_rows = [{'method': k[0], 'camera': k[1], 'status': k[2], 'reason': k[3], 'n_failed_camera_results': count}
                    for k, count in sorted(failures.items())]
    output = output_path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    public_rows = [{k: v for k, v in row.items() if not k.startswith('_')} for row in rows]
    public_fallbacks = [{k: v for k, v in row.items() if not k.startswith('_')} for row in fallbacks]
    artifacts = {
        'per_pair_metrics': _write_csv(output / 'per_pair_metrics.csv', public_rows),
        'summary_metrics': _write_csv(output / 'summary_metrics.csv', summaries),
        'failure_breakdown': _write_csv(output / 'failure_breakdown.csv', failure_rows,
                                       ['method', 'camera', 'status', 'reason', 'n_failed_camera_results']),
        'timing': _write_csv(output / 'timing.csv', _timing_rows(chosen, results),
                             ['pair_id', 'split', 'method', 'robot_forward_ms', 'voting_ms', 'matcher_ms',
                              'initialization_ms', 'ba_ms', 'solver_total_ms', 'pair_runtime_ms', 'pair_all_methods_wall_ms',
                              'robot_observations_reused', 'source_robot_forward_ms', 'source_voting_ms', 'timing_note']),
        'paired_comparisons': _write_csv(output / 'paired_comparisons.csv', comparisons),
        'fallback_per_pair': _write_csv(output / 'operational_fallback_per_pair.csv', public_fallbacks),
        'fallback_summary': _write_csv(output / 'operational_fallback_summary.csv', fallback_summaries),
    }
    write_json(output / 'curves.json', curves)
    artifacts['plots'] = _plot_curves(curves, output)
    artifacts['representative_overlays'] = _representative_overlays(chosen, rows, results, observations, output)
    report = {'schema_version': 2, 'reference_role': 'annotation-reference agreement',
              'reference_is_independent_gt': False, 'split': split, 'n_pairs': len(chosen),
              'n_result_pairs_present': sum(str(p['pair_id']) in results for p in chosen),
              'reference_manifest_path': str(Path(reference_manifest_path).resolve()),
              'reference_manifest_sha256': sha256(reference_manifest_path),
              'run_directory': str(Path(run_directory).resolve()), 'methods': methods,
              'summary': summaries, 'paired_comparisons': comparisons,
              'failure_breakdown': failure_rows, 'operational_fallback_summary': fallback_summaries,
              'artifacts': artifacts,
              'metric_policy': {'ADD': 'mean Euclidean distance of seven FK points, mm; not mesh ADD',
                                'ADD_curve': 'strict <; all expected camera results including failure; 0..100mm step5; normalized trapezoid AUC',
                                'PCK': 'strict <; finite in-image reference points; invalid/missing predictions fail; finite OOB predictions retain distance',
                                'PCK_curve': '0..20px step0.25; selected-candidate and pose projections are separate from raw voting',
                                'pooled_denominator': 'camera results: two per pair; n_unique_pairs and n_unique_stereo_pairs also reported',
                                'conditional_statistics': 'finite successful poses only; failures never receive arbitrary finite errors',
                                'paired_delta': 'method minus matching baseline on common successful camera results; lower is better',
                                'fallback': 'excluded from main dual metrics; separate operational fallback table',
                                'timing': 'pair_runtime_ms = solver_total_ms + recorded matcher_ms for dual_mean/dual_candidates; mono/no_cross use solver_total_ms. Current cached robot forward/vote cost is zero only when reuse is explicit; original measured source times are separate, and null means unmeasured. The all-method wall time is not substituted for per-method runtime.',
                                'sequence_scope': 'one correlated sequence; no independent-trial confidence interval claimed'}}
    write_json(output / 'evaluation.json', report)
    return report
