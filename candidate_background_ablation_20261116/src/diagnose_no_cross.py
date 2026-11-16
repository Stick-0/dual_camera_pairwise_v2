#!/usr/bin/env python3
"""Numerical separability diagnostic from the four prespecified stored starts.

This is a diagnostic, not a fifth ablation condition.  It compares the saved
12-variable A/G OFF solve with two six-variable camera solves from exactly
A0 and B0=G0@A0.  Finite-budget differences do not imply image coupling.
No reference, images, model, matcher, ADD or pose-accuracy metric is accessed.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
import run  # Reuses experiment-local write guard and single-thread settings.

import argparse
import json
import time

import cv2
import numpy as np
import yaml

from controlled_solver import _context, _shared_mono, _counted_solve, array_hash
from dual_camera_pairwise_v2.pairwise import solver as original
from dual_camera_pairwise_v2.pairwise.geometry import pose, pose_vector
from pairwise.bridge import ObservationStore, read_pairs
from runtime import write_json, sha256, jsonable


def robot_objective(T, data, cfg):
    state = original._robot_state(T, data, cfg)
    penalty = .5*np.sum(original._depth_residual(state['depth'][data['valid']], cfg)**2)
    return float(state['cost']+penalty), state, float(penalty)


def independent_camera(T0, data, cfg):
    """One fixed start, same candidate/depth objective and main EM thresholds."""
    x = pose_vector(np.asarray(T0, float)); old = np.inf; trace = []
    termination = 'maximum_em_iterations'
    started = time.perf_counter()
    for iteration in range(int(cfg['candidate_options']['max_em_iterations'])):
        cost, state, _ = robot_objective(pose(x), data, cfg)
        def residual(value):
            rows, depth = original._robot_residual(pose(value), data, state)
            return np.r_[rows, original._depth_residual(depth, cfg)]
        # Six columns, dense direct trust-region solve. This deliberate numerical
        # distinction from joint sparse LSMR is recorded, never a new factor.
        opt, work = _counted_solve(residual, x, max_nfev=cfg['max_nfev'],
            x_scale='jac', ftol=1e-8, xtol=1e-8, gtol=1e-8)
        newcost, _, _ = robot_objective(pose(opt.x), data, cfg)
        rejected = not np.isfinite(opt.x).all() or not np.isfinite(newcost) or newcost > cost+1e-5
        trace.append(dict(work, iteration=iteration+1, objective_before=cost,
                          objective_proposed=newcost, accepted=not rejected))
        if rejected:
            termination = 'true_objective_increase_rejected'; break
        x = opt.x
        if abs(old-newcost) < 1e-7*(1+abs(newcost)):
            termination = 'objective_relative_tolerance'; break
        old = newcost
    T = pose(x); cost, state, penalty = robot_objective(T, data, cfg)
    return {'pose': T, 'objective': cost, 'candidate_nll': state['cost'],
            'robot_depth': penalty, 'initial_pose': np.asarray(T0, float),
            'initial_pose_hash': array_hash(T0), 'trace': trace,
            'em_iterations': len(trace), 'nfev_total': sum(t['nfev'] for t in trace),
            'termination_reason': termination, 'elapsed_ms': (time.perf_counter()-started)*1000,
            'parameter_count': 6, 'starts': 1,
            'positive_robot_depth': bool(np.all(state['depth'][data['valid']] > cfg['depth_min_m'])),
            'effective_robot_observations': float(state['effective'].sum())}


def pose_difference(T, U):
    """Numerical difference between estimates, not an error against reference."""
    T, U = np.asarray(T, float), np.asarray(U, float)
    return {'matrix_max_abs': float(np.max(np.abs(T-U))),
            'translation_m': float(np.linalg.norm(T[:3, 3]-U[:3, 3])),
            'rotation_deg': float(np.linalg.norm(cv2.Rodrigues(T[:3, :3]@U[:3, :3].T)[0])*180/np.pi)}


def equal_optional_array(a, b):
    return a is None and b is None if a is None or b is None else np.array_equal(a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/experiment.yaml')
    parser.add_argument('--smoke-dir', type=Path, default=ROOT/'results/smoke')
    parser.add_argument('--output', type=Path, default=ROOT/'validation/no_cross_diagnostic.json')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    run.prohibit_reference(config['inputs']['reference'])
    safe_config = run.solver_config(config)
    store = ObservationStore(config['inputs']['observations'])
    pairs = {p['pair_id']: p for p in read_pairs(config['inputs']['pairs'])}
    frozen_solver = ROOT/'src/controlled_solver.py'
    frozen_hash = sha256(frozen_solver)
    comparisons, mono_checks = [], []
    sources = {str(args.config): sha256(args.config), str(frozen_solver): frozen_hash,
               str(Path(original.__file__)): sha256(original.__file__), str(Path(__file__)): sha256(__file__)}
    for pid in config['control']['smoke_pair_ids']:
        pair = pairs[pid]; obs = store.pair(pair); context = _context(pair, obs, safe_config)
        saved_path = args.smoke_dir/'pairs'/f'{pid}.json'
        saved = json.loads(saved_path.read_text()); sources[str(saved_path)] = sha256(saved_path)
        for i, cam in enumerate(('camera1', 'camera2')):
            seed = (context['seed']+i) % (2**31-1)
            old = original._fit_mono(context['data'][cam], seed, context['cfg'])
            recorded = _shared_mono(context['data'][cam], seed, context['cfg'])
            old_alts, new_alts = old['alternatives'], recorded['alternatives']
            exact_pose = equal_optional_array(old['pose'], recorded['pose'])
            exact_alts = len(old_alts) == len(new_alts) and all(np.array_equal(a, b) for a, b in zip(old_alts, new_alts))
            exact_saved = equal_optional_array(recorded['pose'], saved['shared_monocular'][cam]['pose'])
            exact_quality = old['quality'] == recorded['quality']
            assert exact_pose and exact_alts and exact_saved and exact_quality
            mono_checks.append({'pair_id': pid, 'camera': cam, 'passed': True,
                'original_pose_exact': exact_pose, 'original_alternatives_exact': exact_alts,
                'original_quality_exact': exact_quality, 'stored_shared_pose_exact': exact_saved,
                'alternative_count': len(old_alts), 'pose_hash': array_hash(recorded['pose']) if recorded['pose'] is not None else None})
        for family in ('R', 'E'):
            init = saved['initializations'][family]; off = saved['methods'][family+'0']
            if not init['available']:
                comparisons.append({'pair_id': pid, 'family': family, 'available': False, 'reason': init['reason']})
                continue
            A0, G0 = np.asarray(init['A0'], float), np.asarray(init['G0'], float)
            poses0 = {'camera1': A0, 'camera2': G0@A0}
            independent = {cam: independent_camera(T0, context['data'][cam], context['cfg']) for cam, T0 in poses0.items()}
            differences = {cam: pose_difference(independent[cam]['pose'], off['poses'][cam]) for cam in independent}
            independent_cost = sum(v['objective'] for v in independent.values())
            off_recomputed = sum(robot_objective(np.asarray(off['poses'][cam]), context['data'][cam], context['cfg'])[0] for cam in independent)
            assert np.isclose(off_recomputed, off['objective'], rtol=1e-12, atol=1e-10)
            objective_delta = independent_cost-off['objective']
            # Flag only for investigation, never alter main settings/results.
            flags = []
            if abs(objective_delta) > 1e-4*(1+abs(off['objective'])): flags.append('objective_difference_above_diagnostic_tolerance')
            if max(v['translation_m'] for v in differences.values()) > .001: flags.append('translation_difference_above_1mm')
            if max(v['rotation_deg'] for v in differences.values()) > .1: flags.append('rotation_difference_above_0_1deg')
            comparisons.append({'pair_id': pid, 'family': family, 'available': True,
                'initial_A_G_hash': init['pose_array_hash'], 'same_A0_and_B0_rule': True,
                'joint_OFF_objective': off['objective'], 'joint_OFF_objective_recomputed_from_saved_poses': off_recomputed,
                'independent_total_objective': independent_cost,
                'independent_minus_joint_objective': objective_delta,
                'pose_difference_independent_vs_joint': differences,
                'joint_OFF_common_valid': off['common_valid'], 'joint_OFF_final_valid': off['final_valid'],
                'joint_OFF_termination': off['termination_reason'],
                'joint_OFF_em_iterations': off['em_iterations'], 'joint_OFF_nfev': off['optimizer_nfev_total'],
                'independent_cameras': independent, 'investigation_flags': flags})
    assert sha256(frozen_solver) == frozen_hash
    try:
        Path(config['inputs']['reference']).read_text()
    except PermissionError:
        reference_denied = True
    else:
        raise AssertionError('Reference reads must be denied')
    valid = [c for c in comparisons if c['available']]
    report = {'status': 'PASS', 'prespecified_pairs': config['control']['smoke_pair_ids'],
        'purpose': 'Numerical separability diagnostic only, not accuracy evaluation or an additional ablation condition',
        'equivalence': 'F(A,G)=F1(A)+F2(GA); setting B=GA gives two independent pose objectives. E initialization may still contain two-view information.',
        'algorithm_comparison': {'common': 'same fixed candidate prior/coordinates, robot depth, EM10, max_nfev60 per inner solve, ftol/xtol/gtol1e-8, true-objective reject above1e-5, relative stop1e-7*(1+abs(cost))',
            'joint_OFF': 'one start; free A/G12 parameters; sparse Jacobian LSMR',
            'independent': 'one start each at A0 and G0@A0; two free6-parameter camera solves; dense direct trust-region solver',
            'interpretation': 'Equal mathematical objectives do not imply identical finite-iteration estimates; any difference is a numerical/parameterization/convergence effect, not added cross-view factors.'},
        'comparisons': comparisons, 'shared_monocular_exactness_checks': mono_checks,
        'summary': {'comparisons': len(valid), 'all_shared_monocular_exact': all(x['passed'] for x in mono_checks),
            'max_absolute_objective_difference': max(abs(x['independent_minus_joint_objective']) for x in valid),
            'max_translation_difference_m': max(d['translation_m'] for x in valid for d in x['pose_difference_independent_vs_joint'].values()),
            'max_rotation_difference_deg': max(d['rotation_deg'] for x in valid for d in x['pose_difference_independent_vs_joint'].values()),
            'flagged_comparisons': [{'pair_id': x['pair_id'], 'family': x['family'], 'flags': x['investigation_flags']} for x in valid if x['investigation_flags']]},
        'reference_access_denied': reference_denied, 'reference_used': False,
        'controlled_solver_unchanged': True, 'source_sha256': sources}
    write_json(args.output, report)
    print(json.dumps(jsonable({'output': args.output, **report['summary']}), ensure_ascii=False))


if __name__ == '__main__':
    main()
