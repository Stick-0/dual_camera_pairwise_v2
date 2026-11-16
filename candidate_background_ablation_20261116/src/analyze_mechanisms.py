#!/usr/bin/env python3
"""Evaluation-only mechanism diagnostics; no optimization or inference rerun.

Read fixed stored initializations and main-run outputs after the complete run.
All initial/final ADD changes use the same pair/camera intersection. Diagnostics
describe associations and failure gates, not unmeasured matching ground truth.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
import run  # Experiment-local output guard only; no estimation is called.

import argparse
from collections import Counter
import json
import time

import cv2
import numpy as np
from scipy.stats import spearmanr
import yaml

from runtime import write_json, sha256
from evaluate import pose_array

METHODS = ('R0', 'R1', 'E0', 'E1')
CAMERAS = ('camera1', 'camera2')
CONTRASTS = (('R1-R0', 0, 1), ('E1-E0', 2, 3), ('E0-R0', 0, 2), ('E1-R1', 1, 3))


def stats(values):
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    return {'n': len(values), 'mean': float(values.mean()) if len(values) else None,
            'median': float(np.median(values)) if len(values) else None,
            'p10': float(np.percentile(values, 10)) if len(values) else None,
            'p90': float(np.percentile(values, 90)) if len(values) else None,
            'min': float(values.min()) if len(values) else None,
            'max': float(values.max()) if len(values) else None}


def finite(value):
    try: return float(value) if np.isfinite(float(value)) else np.nan
    except (TypeError, ValueError): return np.nan


def rotation_deg(T, U):
    return float(np.linalg.norm(cv2.Rodrigues(T[:3, :3]@U[:3, :3].T)[0])*180/np.pi)


def transform(T, X):
    return X@T[:3, :3].T + T[:3, 3]


def add(T, reference, X):
    return float(np.linalg.norm(transform(T, X)-transform(reference, X), axis=1).mean()*1000.)


def direction_deg(T, U):
    a, b = T[:3, 3], U[:3, 3]
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.rad2deg(np.arccos(np.clip(a@b/norm, -1, 1)))) if norm > 1e-15 else np.nan


def paired(a, b, mask):
    selected = mask & np.isfinite(a) & np.isfinite(b)
    delta = (b-a)[selected]
    return {'n_common_camera_observations': int(selected.sum()),
            'initial_or_before_ADD_mm': stats(a[selected]), 'final_or_after_ADD_mm': stats(b[selected]),
            'after_minus_before_ADD_mm': stats(delta),
            'improved_fraction': float(np.mean(delta < 0)) if len(delta) else None,
            'worsened_fraction': float(np.mean(delta > 0)) if len(delta) else None}


def group_label(method):
    if method.get('final_valid'): return 'final_accepted'
    if method.get('common_valid'): return 'background_gate_only_rejected'
    if method.get('initializations_tried', 0) != 1: return 'main_solve_unavailable'
    return 'common_gate_rejected'


def diagnostics(method):
    common = method.get('common_gate', {}).get('diagnostics', {})
    bg = method.get('background_gate', {}).get('diagnostics', {})
    robot = common.get('effective_robot_observations', {})
    robot_values = [finite(robot.get(c)) for c in CAMERAS]
    weights = np.asarray(bg.get('rgb_final_weights', []), float)
    values = {'rank': finite(common.get('reduced_rank')),
              'condition': finite(common.get('reduced_condition')),
              'effective_robot_min_camera': min(robot_values),
              'effective_robot_total': sum(robot_values),
              'effective_background_count': finite(bg.get('effective_rgb_correspondences')),
              'background_median_error_px': finite(bg.get('rgb_median_error_px')),
              'background_p90_error_px': finite(bg.get('rgb_p90_error_px')),
              'background_positive_depth_fraction': finite(bg.get('positive_depth_fraction')),
              'parallax_median_deg': finite(bg.get('parallax_median_deg')),
              'baseline_m': finite(bg.get('baseline_m')),
              'huber_downweighted_fraction': float(np.mean(weights < 1.-1e-12)) if len(weights) else np.nan,
              'below_effective_weight_fraction': float(np.mean(weights < .2)) if len(weights) else np.nan,
              'em_iterations': finite(method.get('em_iterations')),
              'optimizer_nfev_total': finite(method.get('optimizer_nfev_total')),
              'optimization_ms': finite(method.get('timing', {}).get('optimization_ms'))}
    values['log10_condition'] = np.log10(values['condition']) if values['condition'] > 0 else np.nan
    return values


def diagnostic_group(rows):
    if not rows: return {'n_pairs': 0, 'metrics': {}}
    keys = rows[0].keys()
    return {'n_pairs': len(rows), 'metrics': {k: stats([row[k] for row in rows]) for k in keys}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/experiment.yaml')
    parser.add_argument('--run-dir', type=Path, default=ROOT/'results/full')
    parser.add_argument('--output', type=Path, default=ROOT/'validation/mechanism_analysis.json')
    args = parser.parse_args()
    manifest_path = args.run_dir/'manifest.json'
    results_path = args.run_dir/'pair_results.jsonl'
    if not manifest_path.exists() or not results_path.exists():
        raise SystemExit('Complete run manifest/results not ready; no analysis written')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('completed_pairs') != 2596 or manifest.get('code_changed_during_run'):
        raise SystemExit('Full2596 complete immutable-source run required')
    assert sha256(results_path) == manifest['results_sha256']
    config = yaml.safe_load(args.config.read_text())
    frozen = ROOT/'src/controlled_solver.py'; frozen_hash = sha256(frozen)
    # Reference is intentionally opened only in this evaluation-only program.
    reference_path = Path(config['inputs']['reference'])
    refs = {r['pair_id']: r for r in (json.loads(line) for line in reference_path.open())}
    N = 2596
    init_add = np.full((N, 2, 2), np.nan); end_add = np.full((N, 4, 2), np.nan)
    common = np.zeros((N, 4), bool); final = common.copy(); optimized = common.copy()
    available = np.zeros((N, 2), bool); pre_eligible = np.zeros(N, bool)
    move_t = np.full((N, 4, 2), np.nan); move_r = move_t.copy()
    method_diags = {m: [] for m in METHODS}; groups = {m: {} for m in METHODS}
    failure_reasons = {m: {'primary': Counter(), 'common_all': Counter(), 'background_all': Counter(), 'stage': Counter()} for m in METHODS}
    initialization_reasons = {f: Counter() for f in ('R', 'E')}
    budget = {m: {'main_started_pairs': 0, 'EM10_count': 0, 'EM10_exhausted_count': 0,
                  'inner_solve_count': 0, 'nfev60_inner_count': 0, 'any_nfev60_pairs': 0,
                  'inner_status0_count': 0, 'any_rejected_update_pairs': 0,
                  'termination': Counter(), 'elapsed_ms': [], 'nfev_total': []} for m in METHODS}
    init_geometry = []; e_proposal_records = []; failed_examples = []; ids = []
    started = time.perf_counter()
    for n, line in enumerate(results_path.open()):
        row = json.loads(line); pid = row['pair_id']; ids.append(pid); ref = refs[pid]
        pre_eligible[n] = row.get('pre_eligible', False)
        initial_poses, initial_G = {}, {}
        for j, family in enumerate(('R', 'E')):
            init = row.get('initializations', {}).get(family, {})
            available[n, j] = bool(init.get('available', False))
            if not available[n, j]:
                initialization_reasons[family][init.get('reason', 'missing_initialization_record')] += 1
                continue
            A, G = pose_array(init['A0']), pose_array(init['G0'])
            assert A is not None and G is not None
            initial_poses[family] = {'camera1': A, 'camera2': G@A}; initial_G[family] = G
            for cidx, cam in enumerate(CAMERAS):
                Tref = pose_array(ref['cameras'][cam]['T_camera_from_base'])
                X = np.asarray(row['observations'][cam]['keypoints_base_m'], float)
                assert np.allclose(X, ref['cameras'][cam]['keypoints_base_m'], atol=1e-7, rtol=0)
                init_add[n, j, cidx] = add(initial_poses[family][cam], Tref, X)
            if family == 'E':
                chosen = next((d for d in init.get('diagnostics', []) if d.get('route') == 'rgb_essential_metric_scale' and d.get('proposal_id') == init.get('proposal_id')), {})
                e_info = next((d for d in init.get('diagnostics', []) if d.get('route') == 'rgb_essential'), {})
                e_proposal_records.append({'pair_id': pid, 'selected_proposal': init.get('proposal_id'),
                    'initial_scale_m': finite(chosen.get('initial_scale_m')),
                    'fitted_scale_m': finite(chosen.get('fitted_scale_m')),
                    'scale_ratio_fitted_to_start': finite(chosen.get('fitted_scale_m'))/finite(chosen.get('initial_scale_m')),
                    'unit_positive_fraction': finite(chosen.get('unit_positive_fraction')),
                    'essential_inlier_count': finite(e_info.get('inlier_count')),
                    'one_sided_monocular': bool(chosen.get('one_sided_monocular_initialization')),
                    'scale_solve_any_nfev60': any(t.get('nfev') == 60 for t in chosen.get('trace', [])),
                    'E0_common_valid': bool(row['methods']['E0'].get('common_valid')),
                    'E1_common_valid': bool(row['methods']['E1'].get('common_valid')),
                    'E1_final_valid': bool(row['methods']['E1'].get('final_valid'))})
        if len(initial_G) == 2:
            Rg, Eg = initial_G['R'], initial_G['E']
            init_geometry.append({'pair_id': pid, 'rotation_difference_deg': rotation_deg(Eg, Rg),
                'translation_vector_difference_mm': float(np.linalg.norm(Eg[:3, 3]-Rg[:3, 3])*1000),
                'translation_direction_difference_deg': direction_deg(Eg, Rg),
                'R_baseline_m': float(np.linalg.norm(Rg[:3, 3])), 'E_baseline_m': float(np.linalg.norm(Eg[:3, 3])),
                'E_minus_R_baseline_mm': float((np.linalg.norm(Eg[:3, 3])-np.linalg.norm(Rg[:3, 3]))*1000)})
        for j, method in enumerate(METHODS):
            v = row['methods'][method]; family = method[0]
            common[n, j], final[n, j] = bool(v.get('common_valid')), bool(v.get('final_valid'))
            optimized[n, j] = v.get('initializations_tried') == 1
            stage = group_label(v); failure_reasons[method]['stage'][stage] += 1
            if not final[n, j]: failure_reasons[method]['primary'][v.get('reason', 'unspecified')] += 1
            for reason in v.get('common_gate', {}).get('reasons', []): failure_reasons[method]['common_all'][reason] += 1
            for reason in v.get('background_gate', {}).get('reasons', []): failure_reasons[method]['background_all'][reason] += 1
            diag = diagnostics(v); method_diags[method].append(diag); groups[method].setdefault(stage, []).append(diag)
            for cidx, cam in enumerate(CAMERAS):
                T = pose_array(v.get('poses', {}).get(cam))
                if T is None: continue
                Tref = pose_array(ref['cameras'][cam]['T_camera_from_base'])
                X = np.asarray(row['observations'][cam]['keypoints_base_m'], float)
                end_add[n, j, cidx] = add(T, Tref, X)
                if family in initial_poses:
                    T0 = initial_poses[family][cam]
                    move_t[n, j, cidx] = np.linalg.norm(T[:3, 3]-T0[:3, 3])*1000
                    move_r[n, j, cidx] = rotation_deg(T, T0)
            b = budget[method]; trace = v.get('trace', [])
            b['main_started_pairs'] += int(optimized[n, j]); b['EM10_count'] += int(v.get('em_iterations') == 10)
            b['EM10_exhausted_count'] += int(v.get('termination_reason') == 'maximum_em_iterations')
            b['inner_solve_count'] += len(trace); b['nfev60_inner_count'] += sum(t.get('nfev') == 60 for t in trace)
            b['any_nfev60_pairs'] += int(any(t.get('nfev') == 60 for t in trace))
            b['inner_status0_count'] += sum(t.get('optimizer_status') == 0 for t in trace)
            b['any_rejected_update_pairs'] += int(any(not t.get('accepted', True) for t in trace))
            b['termination'][v.get('termination_reason', 'not_run')] += 1
            b['elapsed_ms'].append(diag['optimization_ms']); b['nfev_total'].append(diag['optimizer_nfev_total'])
            if not final[n, j]:
                failed_examples.append({'pair_id': pid, 'method': method, 'stage': stage, 'primary_reason': v.get('reason'),
                    'all_common_reasons': v.get('common_gate', {}).get('reasons', []),
                    'all_background_reasons': v.get('background_gate', {}).get('reasons', []),
                    'initial_ADD_mean_two_cameras_mm': float(np.mean(init_add[n, 0 if family == 'R' else 1])),
                    'pre_gate_ADD_mean_two_cameras_mm': float(np.mean(end_add[n, j])),
                    'diagnostics': diag, 'em_iterations': v.get('em_iterations'),
                    'termination': v.get('termination_reason'), 'any_inner_max60': any(t.get('nfev') == 60 for t in trace)})
    assert len(ids) == len(set(ids)) == 2596
    assert ids == manifest['pair_ids']
    assert sha256(frozen) == frozen_hash
    transitions, contrasts, gate_selection, moves = [], [], [], []
    for pop, popmask in (('all', np.ones(N, bool)), ('pre_eligible', pre_eligible)):
        for gate, gmask in (('finite_pre_gate', np.ones_like(common)), ('common', common), ('final', final)):
            for j, method in enumerate(METHODS):
                for camera, ci in (('pooled', [0, 1]), ('camera1', [0]), ('camera2', [1])):
                    mask = np.broadcast_to((popmask & gmask[:, j])[:, None], (N, len(ci)))
                    transitions.append({'population': pop, 'gate': gate, 'method': method, 'camera': camera,
                        **paired(init_add[:, j//2, ci], end_add[:, j, ci], mask)})
            for label, before, after in CONTRASTS:
                mask = np.broadcast_to((popmask & gmask[:, before] & gmask[:, after])[:, None], (N, 2))
                contrasts.append({'population': pop, 'gate': gate, 'contrast': label,
                                  **paired(end_add[:, before], end_add[:, after], mask)})
    for j, method in enumerate(METHODS):
        for label, mask in (('all_finite_pre_gate', np.ones(N, bool)), ('common_pass', common[:, j]),
                ('final_pass', final[:, j]), ('common_pass_background_rejected', common[:, j] & ~final[:, j]),
                ('common_rejected', ~common[:, j])):
            gate_selection.append({'method': method, 'group': label, 'pair_count': int(mask.sum()),
                'initial_ADD_mm': stats(init_add[mask, j//2]), 'finite_pre_gate_ADD_mm': stats(end_add[mask, j])})
            moves.append({'method': method, 'group': label,
                'initial_to_output_translation_mm': stats(move_t[mask, j]),
                'initial_to_output_rotation_deg': stats(move_r[mask, j])})
    for b in budget.values():
        den = b['main_started_pairs']; inner = b['inner_solve_count']
        b['EM10_fraction_of_started'] = b['EM10_count']/den if den else None
        b['EM10_exhausted_fraction_of_started'] = b['EM10_exhausted_count']/den if den else None
        b['any_nfev60_fraction_of_started'] = b['any_nfev60_pairs']/den if den else None
        b['nfev60_fraction_of_inner_solves'] = b['nfev60_inner_count']/inner if inner else None
        b['optimization_ms'] = stats(b.pop('elapsed_ms')); b['nfev_total'] = stats(b['nfev_total'])
    # Prespecified descriptive threshold, not a setting selected to improve pose.
    effect_groups = {}; associations = []
    pairdelta = np.mean(end_add[:, 1]-end_add[:, 0], axis=1)
    eligible = common[:, 0] & common[:, 1] & np.isfinite(pairdelta)
    for label, mask in (('improved_more_than_1mm', pairdelta < -1),
                        ('small_effect_abs_at_most_1mm', np.abs(pairdelta) <= 1),
                        ('worsened_more_than_1mm', pairdelta > 1)):
        mask &= eligible; indices = np.flatnonzero(mask)
        effect_groups[label] = {'pair_count': int(mask.sum()), 'pair_mean_ADD_delta_mm': stats(pairdelta[mask]),
            'R_initial_ADD_mm': stats(init_add[mask, 0]), 'R0_ADD_mm': stats(end_add[mask, 0]),
            'R1_ADD_mm': stats(end_add[mask, 1]),
            'R1_diagnostics': diagnostic_group([method_diags['R1'][i] for i in indices])}
    for name in ('log10_condition', 'effective_robot_min_camera', 'background_median_error_px',
                 'background_p90_error_px', 'parallax_median_deg', 'huber_downweighted_fraction', 'em_iterations'):
        x = np.asarray([d[name] for d in method_diags['R1']]); mask = eligible & np.isfinite(x)
        rho = spearmanr(x[mask], pairdelta[mask]).statistic if mask.sum() > 2 and np.std(x[mask]) > 0 else None
        associations.append({'diagnostic': name, 'n_pairs': int(mask.sum()), 'spearman_rho_with_R1_minus_R0_pairmean_ADD': rho,
                             'interpretation': 'descriptive association only, not a causal attribution; sequential pairs are not independent'})
    e_groups = {}
    for label, select in (('all', lambda d: True), ('E0_common_rejected', lambda d: not d['E0_common_valid']),
                          ('E1_common_rejected', lambda d: not d['E1_common_valid']),
                          ('E1_final_accepted', lambda d: d['E1_final_valid']),
                          ('E1_background_only_rejected', lambda d: d['E1_common_valid'] and not d['E1_final_valid'])):
        rows = [r for r in e_proposal_records if select(r)]
        e_groups[label] = {'n_pairs': len(rows), 'selected_proposals': dict(Counter(r['selected_proposal'] for r in rows)),
            'one_sided_monocular_count': sum(r['one_sided_monocular'] for r in rows),
            'scale_solve_any_nfev60_count': sum(r['scale_solve_any_nfev60'] for r in rows),
            'metrics': {key: stats([r[key] for r in rows]) for key in ('initial_scale_m', 'fitted_scale_m',
                'scale_ratio_fitted_to_start', 'unit_positive_fraction', 'essential_inlier_count')}}
    report = {'status': 'COMPLETE', 'analysis_role': 'evaluation_only_no_estimation_rerun',
        'run_pair_count': N, 'camera_observations': 2*N, 'pre_eligible_pairs': int(pre_eligible.sum()),
        'method_order': METHODS, 'initial_available_pairs': {f: int(available[:, j].sum()) for j, f in enumerate(('R', 'E'))},
        'initial_unavailable_reasons': initialization_reasons,
        'main_optimized_pairs': {m: int(optimized[:, j].sum()) for j, m in enumerate(METHODS)},
        'common_valid_pairs': {m: int(common[:, j].sum()) for j, m in enumerate(METHODS)},
        'final_valid_pairs': {m: int(final[:, j].sum()) for j, m in enumerate(METHODS)},
        'units': {'ADD': 'mm, same seven FK points and camera-specific capture q',
                  'pose_movement_translation': 'mm in each camera coordinates', 'rotation': 'degrees'},
        'definitions': {'stage_transition': 'Initial and output ADD compared on exactly the same pair/camera intersection; no subtraction of separately selected means',
            'finite_pre_gate': 'All saved finite transforms, including initial transforms if a main solve was unavailable; main_optimized_pairs is provided separately',
            'common': 'Common robot/finite/information gate only', 'final': 'Common gate plus applicable background gates',
            'gate_failure_counts': 'pair-level; all-reasons counters may overlap',
            'budget': 'main nonlinear solves only; nfev is SciPy nfev, not residual calls including numeric Jacobians',
            'R1_R0_small_effect': 'Absolute change of two-camera pair-mean ADD <=1mm, common-valid pair population; descriptive only'},
        'initial_to_stage_paired_changes': transitions, 'stage_paired_contrasts': contrasts,
        'gate_selection_characterization': gate_selection, 'pose_movement': moves,
        'failure_reasons': failure_reasons,
        'method_diagnostics_by_failure_group': {m: {g: diagnostic_group(rows) for g, rows in grouped.items()} for m, grouped in groups.items()},
        'optimizer_budget_and_termination': budget,
        'E_selected_proposal_characterization': e_groups,
        'E_minus_R_initial_G': {'n_pairs': len(init_geometry), 'metrics': {k: stats([r[k] for r in init_geometry]) for k in init_geometry[0] if k != 'pair_id'},
            'interpretation': 'Differences between two estimated initializations, not reference errors. A larger change does not itself identify a correct initialization.'},
        'R1_R0_effect_groups': effect_groups, 'descriptive_associations': associations,
        'failure_pair_records': failed_examples,
        'facts_vs_hypotheses': {'facts': ['Recorded gate reasons, actual iteration budgets, initialization arrays and observed paired ADD changes are measured facts.',
                'All methods share fixed candidate observations and R1/E1 share the same selected background coordinates.',
                'E and R are separately selected initial states; after initialization all A/G pose parameters are free.',
                'No accuracy metric, hypothesis or diagnostic here changes inference settings or stored results.'],
            'hypotheses_requiring_more_evidence': ['Associations of weak effective robot weights or rank with E failure are compatible with a poor local basin or candidate-outlier saturation, but alone do not prove a unique cause.',
                'A solve reaching EM10 or inner nfev60 is budget limited; it does not prove a longer solve would improve reference ADD.',
                'High background residual or Huber downweighting is a mismatch proxy, not verified incorrect correspondence. No ground-truth match labels are available.',
                'Small ADD improvement can reflect already accurate initialization, remaining depth ambiguity, local optimization, or observation/calibration bias; these summaries do not isolate them causally.']},
        'frozen_solver_unchanged': sha256(frozen) == frozen_hash,
        'source_sha256': {str(args.config): sha256(args.config), str(results_path): sha256(results_path),
            str(manifest_path): sha256(manifest_path), str(reference_path): sha256(reference_path),
            str(frozen): frozen_hash, str(Path(__file__)): sha256(__file__)},
        'analysis_wall_seconds': time.perf_counter()-started}
    write_json(args.output, report)
    print(json.dumps({'status': 'COMPLETE', 'output': str(args.output),
        'initial_available_pairs': report['initial_available_pairs'], 'common_valid_pairs': report['common_valid_pairs'],
        'final_valid_pairs': report['final_valid_pairs'],
        'R1_R0_effect_counts': {k: v['pair_count'] for k, v in effect_groups.items()}}, ensure_ascii=False))


if __name__ == '__main__': main()
