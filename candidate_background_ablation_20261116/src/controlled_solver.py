"""Controlled Candidates x background experiment, using read-only V2 numerics.

Sources: dual_camera_pairwise_v2/pairwise/{solver,candidates,geometry}.py.
The original candidate likelihood, PnP seeds, geometry, residual construction,
LSMR parameterization, tolerances and information reduction are reused.  This
adapter changes selection/control flow: shared mono initialization once, one
robot-cost-selected R/E proposal, fixed observations, and separately saved gates.
No evaluation/reference module or file is read; no function writes files.

Public APIs return JSON-compatible values (nonfinite diagnostics become null):
  generate_initializations(pair, observations_by_camera, rgb_matches, config)
  solve_from_fixed_initial(pair, observations_by_camera, initial,
                           background_on, rgb_matches=None, config=None, P0=None)
  solve_pair_controlled(pair, observations_by_camera, rgb_matches=None, config=None)
"""
import sys
sys.dont_write_bytecode = True

import copy
import hashlib
import json
import time

import cv2
import numpy as np
from scipy.optimize import least_squares

from dual_camera_pairwise_v2.pairwise import solver as original
from dual_camera_pairwise_v2.pairwise.geometry import (
    pose, pose_vector, inverse, transform, project, triangulate,
    essential_candidates, parallax_degrees, hull_fraction, huber_cost,
)
from dual_camera_pairwise_v2.pairwise.runtime import pair_seed, jsonable

CAMERAS = ('camera1', 'camera2')
METHODS = ('R0', 'R1', 'E0', 'E1')
SELECTION_RULE = ('minimum current candidate negative-log-likelihood plus robot '
                  'depth penalty, then lexical proposal_id; no background cost '
                  'or final gate or evaluation reference used')


def _digest(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array_hash(value):
    """Hash shape and canonical little-endian float64 C-order array bytes."""
    value = np.ascontiguousarray(value, dtype='<f8')
    return hashlib.sha256(json.dumps(list(value.shape)).encode() + b'\0' +
                          value.tobytes()).hexdigest()


def _pose_hash(A, G):
    return array_hash(np.stack((np.asarray(A, float), np.asarray(G, float))))


def _context(pair, observations, config):
    config = copy.deepcopy(config or {})
    config.setdefault('solver', {}).setdefault('max_nfev', 60)
    config['solver'].setdefault('max_joint_starts', 1)
    cfg = original._settings(config)
    if cfg['max_joint_starts'] != 1:
        raise ValueError('Controlled experiment requires max_joint_starts=1')
    cameras, raw, metadata = {}, {}, {}
    for cid in CAMERAS:
        record = pair['cameras'][cid]
        cal = record.get('calibration', record)
        cameras[cid] = {'K': np.array(cal['K'], float, copy=True),
            'D': np.array(cal.get('D', cal.get('distortion', [])), float, copy=True),
            'width': int(cal.get('width', 640)), 'height': int(cal.get('height', 480)),
            'pixel_space': cal.get('pixel_space', 'native_distorted')}
        # Explicit whitelist: arbitrary annotation/reference/legacy-pose keys
        # cannot become estimation inputs or affect hashes.
        obs = observations[cid]
        raw[cid] = {key: copy.deepcopy(obs[key]) for key in
                    ('keypoints_base_m', 'keypoints_uv', 'valid_keypoints', 'covariance_uv')}
        raw[cid]['hypotheses'] = copy.deepcopy(obs.get('hypotheses'))
        metadata[cid] = {key: copy.deepcopy(record[key]) for key in
                         ('q_rad', 'timestamp_s', 'frame_id') if key in record}
    data = {c: original._data(raw[c], cameras[c], 'candidates', cfg) for c in CAMERAS}
    identity = _digest({'observations': raw, 'cameras': cameras,
                        'capture_metadata': metadata, 'candidate_options': cfg['candidate_options']})
    return {'pair_id': str(pair['pair_id']), 'cfg': cfg, 'cameras': cameras,
            'raw': raw, 'data': data, 'capture_metadata': metadata,
            'observation_hash': identity,
            'seed': pair_seed(str(pair['pair_id']), cfg['seed'])}


def _counted_solve(fun, x, **kwargs):
    calls = 0
    def counted(value):
        nonlocal calls
        calls += 1
        return fun(value)
    started = time.perf_counter()
    result = least_squares(counted, x, **kwargs)
    return result, {'nfev': int(result.nfev),
                    'njev': None if result.njev is None else int(result.njev),
                    'residual_function_calls': calls,
                    'elapsed_ms': (time.perf_counter()-started)*1000,
                    'optimizer_success': bool(result.success),
                    'optimizer_status': int(result.status),
                    'optimizer_message': str(result.message)}


def _shared_mono(data, seed, cfg):
    """Instrumented Candidates-only copy of original._fit_mono, same numerics.

    This runs once per camera, not once per ablation condition.  Its original
    multi-start PnP/EM procedure is retained; the main joint solve has one start.
    """
    started = time.perf_counter()
    if data['prepared'] is None or data['valid'].sum() < 4:
        reason = ('missing_original_hypotheses' if data['prepared'] is None
                  else 'fewer_than_four_valid_keypoints')
        return {'pose': None, 'alternatives': [], 'quality': {'status': 'monocular_failed',
                'reason': reason}, 'elapsed_ms': (time.perf_counter()-started)*1000,
                'traces': [], 'initializations_tried': 0}
    try:
        seeds = original._pnp_seeds(data, seed, cfg)
    except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
        return {'pose': None, 'alternatives': [], 'quality': {'status': 'monocular_failed',
                'reason': 'pnp_exception', 'exception': str(error)},
                'elapsed_ms': (time.perf_counter()-started)*1000, 'traces': [],
                'initializations_tried': 0}
    initialized = time.perf_counter()
    fitted, traces = [], []
    for ordinal, initial in enumerate(seeds[:cfg['max_pose_starts']]):
        x = pose_vector(initial); previous_cost = np.inf; trace = []
        termination = 'maximum_em_iterations'
        try:
            for iteration in range(int(cfg['candidate_options']['max_em_iterations'])):
                state = original._robot_state(pose(x), data, cfg)
                def residual(value):
                    rows, depth = original._robot_residual(pose(value), data, state)
                    return np.r_[rows, original._depth_residual(depth, cfg)]
                opt, measured = _counted_solve(residual, x, max_nfev=cfg['max_nfev'],
                    x_scale='jac', ftol=1e-8, xtol=1e-8, gtol=1e-8)
                candidate = original._robot_state(pose(opt.x), data, cfg)
                rejected = (candidate['cost'] > state['cost'] + 1e-6 and
                            np.all(state['depth'][data['valid']] > cfg['depth_min_m']))
                trace.append(dict(measured, iteration=iteration+1,
                    objective_before=state['cost'], objective_proposed=candidate['cost'],
                    accepted=not rejected))
                if rejected:
                    termination = 'original_monocular_increasing_update_rejected'; break
                x = opt.x
                if abs(previous_cost-candidate['cost']) < 1e-7*(1+abs(candidate['cost'])):
                    termination = 'objective_relative_tolerance'; break
                previous_cost = candidate['cost']
            T = pose(x); state = original._robot_state(T, data, cfg)
            def final_residual(value):
                rows, depth = original._robot_residual(pose(value), data, state)
                return np.r_[rows, original._depth_residual(depth, cfg)]
            final, final_work = _counted_solve(final_residual, x, max_nfev=1)
            quality = original._mono_quality(T, data, state, final.jac, cfg)
            quality.update(initializations_tried=min(len(seeds), cfg['max_pose_starts']),
                objective_type='gaussian_mixture_nll', exact_block_huber_final_polish=False,
                iterations=len(trace))
            fitted.append({'pose': T, 'state': state, 'quality': quality, 'ordinal': ordinal})
            traces.append({'proposal_id': f'mono_{ordinal:02d}', 'trace': trace,
                           'termination_reason': termination, 'final_jacobian_work': final_work,
                           'quality': quality})
        except (ValueError, np.linalg.LinAlgError, cv2.error, FloatingPointError) as error:
            traces.append({'proposal_id': f'mono_{ordinal:02d}', 'trace': trace,
                           'termination_reason': 'numerical_exception', 'exception': str(error)})
    fitted.sort(key=lambda item: (item['quality']['status'] != 'monocular_refined',
                                  item['state']['cost']))
    alternatives = []
    for item in fitted:
        if (item['quality']['status'] == 'monocular_refined' and
                not any(np.linalg.norm(item['pose']-old) < 1e-4 for old in alternatives)):
            alternatives.append(item['pose'])
    best = fitted[0] if fitted else None
    return {'pose': best['pose'] if best and best['quality']['status'] == 'monocular_refined' else None,
            'alternatives': alternatives,
            'quality': best['quality'] if best else {'status': 'monocular_failed',
                         'reason': 'current_raw_pnp_initialization_failed'},
            'elapsed_ms': (time.perf_counter()-started)*1000,
            'initialization_ms': (initialized-started)*1000,
            'optimization_ms': (time.perf_counter()-initialized)*1000,
            'initializations_tried': min(len(seeds), cfg['max_pose_starts']), 'traces': traces}


def _robot_initial_cost(A, G, context):
    data, cfg = context['data'], context['cfg']
    states = {c: original._robot_state(T, data[c], cfg)
              for c, T in zip(CAMERAS, (A, G@A))}
    depths = np.concatenate([states[c]['depth'][data[c]['valid']] for c in CAMERAS])
    cost = sum(s['cost'] for s in states.values()) + .5*np.sum(original._depth_residual(depths, cfg)**2)
    return float(cost), states


def _fixed_matches(matches, cfg):
    p1, p2, info = original._match_arrays(matches, cfg)
    info = dict(info)
    info['coordinate_hash'] = array_hash(np.stack((p1, p2)))
    info['selection_hash'] = _digest({'original_indices': info.get('optimizer_selected_indices', []),
                                    'coordinates': np.stack((p1, p2))})
    info['selection_policy'] = 'original matcher valid mask plus deterministic original maximum-80 cap; no initialization-dependent removal'
    return p1, p2, info


def _prepare_landmarks(A, G, p1, p2, context):
    cfg = context['cfg']
    if len(p1) < cfg['min_rgb_matches']:
        return {'available': False, 'reason': 'too_few_actual_rgb_matches', 'P0': None}
    try:
        P = triangulate(p1, p2, G, context['cameras'])
        if P.shape != (len(p1), 3) or not np.isfinite(P).all():
            return {'available': False, 'reason': 'nonfinite_background_triangulation', 'P0': None,
                    'finite_landmarks': int(np.isfinite(P).all(1).sum())}
        positive = (P[:, 2] > cfg['depth_min_m']) & (transform(G, P)[:, 2] > cfg['depth_min_m'])
        return {'available': True, 'reason': None, 'P0': P, 'P0_hash': array_hash(P),
                'landmark_count': len(P), 'positive_depth_fraction': float(positive.mean()),
                'filter_policy': 'all fixed correspondences retained; finite negative depths are penalized, never removed'}
    except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
        return {'available': False, 'reason': 'background_triangulation_exception',
                'exception': str(error), 'P0': None}


def _select_initial(family, proposals, context, p1, p2, match_info, diagnostics, elapsed_ms):
    ranked = []
    for p in proposals:
        if not np.isfinite(p['A0']).all() or not np.isfinite(p['G0']).all():
            continue
        try:
            cost, states = _robot_initial_cost(p['A0'], p['G0'], context)
        except (ValueError, np.linalg.LinAlgError, cv2.error):
            continue
        if np.isfinite(cost):
            ranked.append((cost, p['proposal_id'], p, states))
    ranked.sort(key=lambda item: (item[0], item[1]))
    summary = [{'proposal_id': p['proposal_id'], 'robot_selection_cost': cost,
                'A0': p['A0'], 'G0': p['G0'], 'pose_array_hash': _pose_hash(p['A0'], p['G0'])}
               for cost, _, p, _ in ranked]
    base = {'pair_id': context['pair_id'], 'init_family': family, 'seed': context['seed'],
            'selection_rule': SELECTION_RULE, 'observation_hash': context['observation_hash'],
            'fixed_match_hash': match_info['selection_hash'], 'proposals': summary,
            'diagnostics': diagnostics, 'generation_ms': elapsed_ms}
    if not ranked:
        return dict(base, available=False,
            reason='robot_init_unavailable' if family == 'R' else 'essential_init_unavailable',
            A0=None, G0=None, on_initialization={'available': False, 'reason': 'pose_initialization_unavailable', 'P0': None})
    cost, _, chosen, states = ranked[0]
    A, G = chosen['A0'].copy(), chosen['G0'].copy()
    post = {c: states[c]['posterior'] for c in CAMERAS}
    return dict(base, available=True, reason=None, A0=A, G0=G,
        A0_hash=array_hash(A), G0_hash=array_hash(G), pose_array_hash=_pose_hash(A, G),
        proposal_id=chosen['proposal_id'], robot_selection_cost=cost,
        initial_posterior=post, initial_responsibility_hash=_digest(post),
        on_initialization=_prepare_landmarks(A, G, p1, p2, context))


def _initialize(context, rgb_matches):
    started = time.perf_counter(); data, cfg = context['data'], context['cfg']
    p1, p2, match_info = _fixed_matches(rgb_matches, cfg)
    mono_started = time.perf_counter()
    mono = {c: _shared_mono(data[c], (context['seed']+i) % (2**31-1), cfg)
            for i, c in enumerate(CAMERAS)}
    mono_ms = (time.perf_counter()-mono_started)*1000
    r_started = time.perf_counter(); proposals = []
    A1, A2 = (mono[c]['pose'] for c in CAMERAS)
    if A1 is not None and A2 is not None:
        for i, T1 in enumerate(mono['camera1']['alternatives'][:2]):
            for j, T2 in enumerate(mono['camera2']['alternatives'][:2]):
                proposals.append({'A0': T1, 'G0': T2@inverse(T1), 'proposal_id': f'R_{i:02d}_{j:02d}'})
    R = _select_initial('R', proposals, context, p1, p2, match_info,
        [{'route': 'shared_current_pair_monocular', 'camera_status': {c: mono[c]['quality'] for c in CAMERAS}}],
        (time.perf_counter()-r_started)*1000)
    e_started = time.perf_counter(); proposals = []; diagnostics = []
    if any(data[c]['prepared'] is None for c in CAMERAS):
        diagnostics.append({'reason': 'missing_original_hypotheses'})
    elif A1 is None and A2 is None:
        diagnostics.append({'reason': 'neither_camera_has_valid_monocular_anchor'})
    else:
        try:
            candidates, e_info = essential_candidates(p1, p2, context['cameras'],
                context['seed'], cfg['essential_ransac_threshold_px'])
            diagnostics.append({'route': 'rgb_essential', **e_info})
        except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
            candidates = []; diagnostics.append({'reason': 'essential_exception', 'exception': str(error)})
        scale0 = np.linalg.norm((A2@inverse(A1))[:3, 3]) if A1 is not None and A2 is not None else .5
        scale0 = max(.01, min(scale0, cfg['max_baseline_m']))
        # Same two A/log(s) IRLS solves as original._metric_initializations.
        # E RANSAC masks only generate a pose proposal and never filter p1/p2.
        for ordinal, item in enumerate(candidates):
            Rmat, direction = item['R'], item['direction']
            G0 = np.eye(4); G0[:3, :3] = Rmat; G0[:3, 3] = scale0*direction
            anchor = A1 if A1 is not None else inverse(G0)@A2
            x = np.r_[pose_vector(anchor), np.log(scale0)]
            states = {c: original._robot_state(T, data[c], cfg) for c, T in zip(CAMERAS, (anchor, G0@anchor))}
            lo = np.full(7, -np.inf); hi = np.full(7, np.inf)
            lo[-1], hi[-1] = np.log(1e-5), np.log(cfg['max_baseline_m'])
            trace = []
            try:
                for iteration in range(2):
                    def residual(value):
                        A = pose(value[:6]); G = np.eye(4)
                        G[:3, :3] = Rmat; G[:3, 3] = np.exp(value[6])*direction
                        rows, depths = [], []
                        for c, T in zip(CAMERAS, (A, G@A)):
                            rr, dd = original._robot_residual(T, data[c], states[c])
                            rows.extend(rr); depths.extend(dd)
                        return np.r_[rows, original._depth_residual(depths, cfg)]
                    opt, measured = _counted_solve(residual, x, bounds=(lo, hi),
                        max_nfev=cfg['max_nfev'], x_scale='jac')
                    x = opt.x; A = pose(x[:6]); G = np.eye(4)
                    G[:3, :3] = Rmat; G[:3, 3] = np.exp(x[6])*direction
                    cost, states = _robot_initial_cost(A, G, context)
                    trace.append(dict(measured, iteration=iteration+1, robot_objective=cost))
                proposal_id = f'E_{ordinal:02d}'
                proposals.append({'A0': A, 'G0': G, 'proposal_id': proposal_id})
                diagnostics.append({'route': 'rgb_essential_metric_scale', 'proposal_id': proposal_id,
                    'initial_scale_m': scale0, 'fitted_scale_m': float(np.exp(x[6])),
                    'unit_positive_fraction': item['unit_positive_fraction'],
                    'one_sided_monocular_initialization': A1 is None or A2 is None,
                    'fixed_rotation': Rmat, 'fixed_translation_direction': direction,
                    'trace': trace, 'original_scale_fit_iterations': 2})
            except (ValueError, np.linalg.LinAlgError, cv2.error, FloatingPointError) as error:
                diagnostics.append({'route': 'rgb_essential_metric_scale', 'proposal_id': f'E_{ordinal:02d}',
                                    'reason': 'metric_scale_fit_exception', 'exception': str(error), 'trace': trace})
    E = _select_initial('E', proposals, context, p1, p2, match_info, diagnostics,
                        (time.perf_counter()-e_started)*1000)
    exclusions = []
    for family, init in (('R', R), ('E', E)):
        if not init['available']: exclusions.append(f'{family}:{init["reason"]}')
        elif not init['on_initialization']['available']:
            exclusions.append(f'{family}1:{init["on_initialization"]["reason"]}')
    return {'pair_id': context['pair_id'], 'initializations': {'R': R, 'E': E},
            'shared_monocular': mono, 'fixed_matches': match_info,
            'pre_eligible': not exclusions, 'pre_eligible_reasons': exclusions,
            'pre_eligible_policy': 'R/E finite pose initialization and both fixed-match ON landmark initializations available, before any main optimization or final gate',
            'observation_hash': context['observation_hash'],
            'capture_metadata': context['capture_metadata'],
            'timing': {'shared_monocular_ms': mono_ms,
                       'all_initialization_ms': (time.perf_counter()-started)*1000}}


def generate_initializations(pair, observations_by_camera, rgb_matches=None, config=None):
    """Generate both shared families once; never runs the four main solves."""
    return jsonable(_initialize(_context(pair, observations_by_camera, config), rgb_matches))


def _gate(reasons, diagnostics, applied=True):
    return {'applied': applied, 'passed': not reasons,
            'reason': reasons[0] if reasons else None, 'reasons': reasons,
            'diagnostics': diagnostics}


def _gates(x, jac, states, context, p1, p2, weights, background_on, numerical_error=None):
    cfg, data, cameras = context['cfg'], context['data'], context['cameras']
    A, G = pose(x[:6]), pose(x[6:12]); n = len(p1) if background_on else 0
    effective = {c: float(states[c]['effective'].sum()) for c in CAMERAS}
    positive = {c: bool(np.all(states[c]['depth'][data[c]['valid']] > cfg['depth_min_m'])) for c in CAMERAS}
    info = original._reduced_information(jac, n, cfg)
    if not background_on: info['nuisance_policy'] = 'none_no_landmark_variables_robot_only_Jacobian'
    common_diag = {'pose_finite': bool(np.isfinite(A).all() and np.isfinite(G).all()),
        'robot_positive_depth': positive, 'robot_depth_m': {c: states[c]['depth'] for c in CAMERAS},
        'effective_robot_observations': effective,
        'robot_final_effective_weights': {c: states[c]['effective'] for c in CAMERAS},
        'valid_robot_points': {c: int(data[c]['valid'].sum()) for c in CAMERAS}, **info}
    reasons = []
    if numerical_error: reasons.append('numerical_solve_failure')
    if not np.isfinite(x).all() or not all(positive.values()): reasons.append('nonfinite_or_negative_robot_depth')
    if min(effective.values()) < cfg['min_effective_robot_per_camera'] or sum(effective.values()) < cfg['min_effective_robot_total']:
        reasons.append('insufficient_final_robot_information')
    if info['reduced_rank'] < 12 or info['reduced_condition'] > cfg['max_reduced_condition']:
        reasons.append('reduced_pose_system_rank_or_condition')
    common = _gate(reasons, common_diag)
    if not background_on:
        return common, _gate([], {'actual_rgb_correspondences_used': 0,
            'background_factor_rows': 0, 'background_depth_rows': 0,
            'landmark_parameter_count': 0, 'background_gate_count': 0}, applied=False)
    P = x[12:].reshape(n, 3); Q = transform(G, P)
    errors = np.maximum(np.linalg.norm(p1-project(P, cameras['camera1']), axis=1),
                        np.linalg.norm(p2-project(Q, cameras['camera2']), axis=1))
    final_weight = np.minimum(weights[:n], weights[n:])
    bg_positive = np.isfinite(P).all(1) & np.isfinite(Q).all(1) & (P[:, 2] > cfg['depth_min_m']) & (Q[:, 2] > cfg['depth_min_m'])
    bg_effective = (final_weight >= cfg['min_effective_rgb_weight']) & bg_positive
    parallax = parallax_degrees(P, G)
    diag = {'actual_rgb_correspondences_used': n, 'effective_rgb_correspondences': int(bg_effective.sum()),
        'effective_rgb_weight_sum': float(final_weight.sum()), 'rgb_final_weights': final_weight,
        'rgb_median_error_px': float(np.median(errors)), 'rgb_p90_error_px': float(np.percentile(errors, 90)),
        'positive_depth_fraction': float(bg_positive.mean()),
        'parallax_median_deg': float(np.median(parallax[bg_effective])) if bg_effective.any() else 0.,
        'baseline_m': float(np.linalg.norm(G[:3, 3])),
        'rgb_hull_fraction': {c: hull_fraction(p[bg_effective], cameras[c]) for c, p in zip(CAMERAS, (p1, p2))},
        'background_depths_m': {'camera1': P[:, 2], 'camera2': Q[:, 2]},
        'background_factor_rows': 4*n, 'background_depth_rows': 2*n, 'landmark_parameter_count': 3*n}
    reasons = []
    if diag['baseline_m'] < cfg['min_baseline_m'] or diag['parallax_median_deg'] < cfg['min_parallax_deg']:
        reasons.append('small_baseline_or_parallax')
    if diag['baseline_m'] > cfg['max_baseline_m']: reasons.append('baseline_outside_configured_metric_range')
    if diag['positive_depth_fraction'] < cfg['min_positive_depth_fraction']: reasons.append('insufficient_positive_depth')
    if diag['effective_rgb_correspondences'] < cfg['min_rgb_matches']: reasons.append('too_few_effective_rgb_matches')
    if min(diag['rgb_hull_fraction'].values()) < cfg['min_rgb_hull_fraction']: reasons.append('insufficient_rgb_spatial_distribution')
    if diag['rgb_median_error_px'] > cfg['max_rgb_median_error_px'] or diag['rgb_p90_error_px'] > cfg['max_rgb_p90_error_px']:
        reasons.append('rgb_reprojection_inconsistent')
    return common, _gate(reasons, diag)


def _components(x, states, context, p1, p2, background_on):
    data, cfg = context['data'], context['cfg']; G = pose(x[6:12])
    depths = np.concatenate([states[c]['depth'][data[c]['valid']] for c in CAMERAS])
    values = {'candidate_nll': float(sum(states[c]['cost'] for c in CAMERAS)),
              'robot_depth': float(.5*np.sum(original._depth_residual(depths, cfg)**2)),
              'background_reprojection': 0., 'background_depth': 0.}
    if background_on:
        P = x[12:].reshape(-1, 3); Q = transform(G, P)
        blocks = np.vstack(((p1-project(P, context['cameras']['camera1']))/cfg['rgb_noise_std_px'],
                            (p2-project(Q, context['cameras']['camera2']))/cfg['rgb_noise_std_px']))
        values['background_reprojection'] = cfg['lambda_rgb']*huber_cost(blocks, cfg['huber_delta'])
        values['background_depth'] = float(.5*np.sum(original._depth_residual(np.r_[P[:, 2], Q[:, 2]], cfg)**2))
    values['total'] = float(sum(values.values()))
    return values


def _attach(result, context):
    original._attach_outputs(result, context['data'], {})
    result['raw_inputs'] = {c: {k: context['raw'][c][k] for k in
                              ('keypoints_uv', 'valid_keypoints', 'keypoints_base_m')} for c in CAMERAS}
    result['candidate_selected_original_id'] = {}
    for c in CAMERAS:
        chosen = []
        for k, post in enumerate(result['posterior'].get(c) or []):
            mode = context['data'][c]['prepared']['modes'][k]
            index = post['selected_index']
            post['original_candidate_ids'] = mode['original_indices']
            chosen.append(int(mode['original_indices'][index]) if index is not None and post['outlier_responsibility'] < .5 else None)
        result['candidate_selected_original_id'][c] = chosen if chosen else [None]*7
    return result


def _solve(context, initial, background_on, p1=None, p2=None, match_info=None, P0=None):
    started = time.perf_counter(); cfg = context['cfg']; family = initial.get('init_family', 'fixed')
    method_id = family + ('1' if background_on else '0')
    result = {'method': method_id, 'init_family': family, 'background_on': bool(background_on),
        'observation_hash': context['observation_hash'], 'seed': context['seed'],
        'poses': {c: None for c in CAMERAS}, 'T_C2_C1': None,
        'common_valid': False, 'final_valid': False, 'pre_gate_pose_available': False,
        'trace': [], 'fallback_used': False, 'initializations_tried': 0,
        'timing': {'optimization_ms': 0., 'total_ms': 0.},
        'common_gate': _gate(['initialization_unavailable'], {}),
        'background_gate': _gate(['initialization_unavailable'], {}) if background_on else _gate([], {}, False)}
    if not initial.get('available', True) or initial.get('A0') is None or initial.get('G0') is None:
        result.update(status=initial.get('reason', 'initialization_unavailable'), reason=initial.get('reason', 'initialization_unavailable'))
        return _attach(result, context)
    A0 = np.array(initial['A0'], float, copy=True); G0 = np.array(initial['G0'], float, copy=True)
    if A0.shape != (4, 4) or G0.shape != (4, 4) or not np.isfinite(A0).all() or not np.isfinite(G0).all():
        result.update(status='initialization_failed', reason='nonfinite_or_invalid_initial_pose'); return _attach(result, context)
    if initial.get('observation_hash', context['observation_hash']) != context['observation_hash']:
        raise ValueError('Fixed initialization belongs to different robot observations')
    result['initialization'] = {'A0': A0.copy(), 'G0': G0.copy(),
        'A0_hash': array_hash(A0), 'G0_hash': array_hash(G0), 'pose_array_hash': _pose_hash(A0, G0),
        'proposal_id': initial.get('proposal_id'), 'selection_rule': initial.get('selection_rule', 'caller_fixed')}
    _, initial_states = _robot_initial_cost(A0, G0, context)
    result['initialization']['responsibility_hash'] = _digest({c: initial_states[c]['posterior'] for c in CAMERAS})
    x = np.r_[pose_vector(A0), pose_vector(G0)]
    # Save finite initial pose even when an ON landmark initialization cannot run.
    result.update(poses={'camera1': A0.copy(), 'camera2': G0@A0}, T_C2_C1=G0.copy(), pre_gate_pose_available=True)
    if background_on:
        result['fixed_matches'] = copy.deepcopy(match_info)
        prepared = (_prepare_landmarks(A0, G0, p1, p2, context) if P0 is None
                    else {'available': True, 'P0': np.array(P0, float, copy=True)})
        P = prepared.get('P0')
        if not prepared['available'] or P is None or np.shape(P) != (len(p1), 3) or not np.isfinite(P).all():
            reason = prepared.get('reason') or 'invalid_fixed_background_landmarks'
            result.update(status='on_initialization_failed', reason=reason,
                          background_gate=_gate([reason], prepared))
            result['common_gate'] = _gate(['main_optimization_not_run'], {})
            return _attach(result, context)
        if len(p1) < cfg['min_rgb_matches']:
            result.update(status='on_initialization_failed', reason='too_few_actual_rgb_matches',
                          background_gate=_gate(['too_few_actual_rgb_matches'], {}))
            return _attach(result, context)
        x = np.r_[x, np.asarray(P).ravel()]
        result['initialization']['P0_hash'] = array_hash(P)
    else:
        # Deliberately never consult matches, P0, cheirality, baseline or E priors.
        p1 = np.empty((0, 2)); p2 = np.empty((0, 2)); P = None
    result['joint_parameter_count'] = len(x)
    result['landmark_parameter_count'] = len(x)-12
    result['initializations_tried'] = 1
    initial_x = x.copy(); old = np.inf; trace = []; numerical_error = None
    termination = 'maximum_em_iterations'; optimize_started = time.perf_counter()
    try:
        for iteration in range(int(cfg['candidate_options']['max_em_iterations'])):
            fun, pattern, states, weights, cost = original._joint_functions(
                context['data'], p1, p2, context['cameras'], cfg, x)
            opt, measured = _counted_solve(fun, x, jac_sparsity=pattern,
                tr_solver='lsmr', x_scale='jac', max_nfev=cfg['max_nfev'],
                ftol=1e-8, xtol=1e-8, gtol=1e-8)
            newcost = original._joint_functions(context['data'], p1, p2, context['cameras'], cfg, opt.x)[-1]
            rejected = not np.isfinite(newcost) or not np.isfinite(opt.x).all() or newcost > cost+1e-5
            trace.append(dict(measured, iteration=iteration+1, objective_before=float(cost),
                objective_proposed=float(newcost), objective=float(cost if rejected else newcost),
                accepted=not rejected, rejected_increasing_update=bool(newcost > cost+1e-5)))
            if rejected:
                termination = 'true_objective_increase_rejected' if np.isfinite(newcost) else 'nonfinite_update_rejected'
                break
            x = opt.x
            if abs(old-newcost) < 1e-7*(1+abs(newcost)):
                termination = 'objective_relative_tolerance'; break
            old = newcost
    except (ValueError, np.linalg.LinAlgError, cv2.error, FloatingPointError) as error:
        numerical_error = str(error); termination = 'numerical_exception'
    optimization_ms = (time.perf_counter()-optimize_started)*1000
    final_started = time.perf_counter()
    try:
        fun, pattern, states, weights, cost = original._joint_functions(
            context['data'], p1, p2, context['cameras'], cfg, x)
        final, final_work = _counted_solve(fun, x, jac_sparsity=pattern, max_nfev=1)
        common, background = _gates(x, final.jac, states, context, p1, p2, weights,
                                   background_on, numerical_error)
        components = _components(x, states, context, p1, p2, background_on)
        result.update(common_gate=common, background_gate=background,
                      common_valid=common['passed'], final_valid=common['passed'] and background['passed'],
                      objective=float(cost), objective_components=components,
                      residual_rows=len(fun(x)), final_jacobian_work=final_work)
    except (ValueError, np.linalg.LinAlgError, cv2.error, FloatingPointError) as error:
        numerical_error = str(error)
        result.update(common_gate=_gate(['final_diagnostics_numerical_failure'], {'exception': str(error)}),
                      background_gate=_gate(['final_diagnostics_unavailable'], {}) if background_on else _gate([], {}, False))
    A, G = pose(x[:6]), pose(x[6:12])
    if np.isfinite(A).all() and np.isfinite(G).all():
        result.update(poses={'camera1': A, 'camera2': G@A}, T_C2_C1=G,
                      landmarks_c1_m=x[12:].reshape(-1, 3) if background_on else None,
                      pre_gate_pose_available=True)
    reason = result['common_gate']['reason'] or result['background_gate']['reason']
    result.update(status='accepted' if result['final_valid'] else 'common_gate_failed' if not result['common_valid'] else 'background_gate_failed',
        reason=reason, trace=trace, em_iterations=len(trace), termination_reason=termination,
        numerical_error=numerical_error, optimizer_nfev_total=sum(t['nfev'] for t in trace),
        optimizer_residual_calls_total=sum(t['residual_function_calls'] for t in trace),
        initialization_pose_change_norm=float(np.linalg.norm(x[:12]-initial_x[:12])),
        timing={'optimization_ms': optimization_ms, 'final_diagnostics_ms': (time.perf_counter()-final_started)*1000,
                'total_ms': (time.perf_counter()-started)*1000})
    return _attach(result, context)


def solve_from_fixed_initial(pair, observations_by_camera, initial, background_on,
                             rgb_matches=None, config=None, P0=None):
    """One main solve, without any PnP/essential generation or fallback.

    OFF ignores rgb_matches and P0 entirely, including malformed values. This
    entry point supports the required fixed-initialization invariance check.
    """
    context = _context(pair, observations_by_camera, config)
    if background_on:
        p1, p2, info = _fixed_matches(rgb_matches, context['cfg'])
        if initial.get('fixed_match_hash', info['selection_hash']) != info['selection_hash']:
            raise ValueError('ON fixed initialization belongs to a different match selection')
        if P0 is None and initial.get('on_initialization', {}).get('available'):
            P0 = initial['on_initialization']['P0']
    else:
        p1 = p2 = info = None
    return jsonable(_solve(context, initial, bool(background_on), p1, p2, info, P0))


def solve_pair_controlled(pair, observations_by_camera, rgb_matches=None, config=None):
    """R0/R1/E0/E1, shared observations and single stored initialization per family."""
    started = time.perf_counter()
    context = _context(pair, observations_by_camera, config)
    output = _initialize(context, rgb_matches)
    p1, p2, info = _fixed_matches(rgb_matches, context['cfg'])
    output['methods'] = {}
    for family in ('R', 'E'):
        initial = output['initializations'][family]
        for background_on in (False, True):
            P0 = initial['on_initialization'].get('P0') if background_on else None
            result = _solve(context, initial, background_on, p1, p2, info, P0)
            output['methods'][family+str(int(background_on))] = result
        off, on = (output['methods'][family+n] for n in ('0', '1'))
        if initial['available']:
            assert off['initialization']['pose_array_hash'] == on['initialization']['pose_array_hash'] == initial['pose_array_hash']
            assert off['initialization']['responsibility_hash'] == on['initialization']['responsibility_hash'] == initial['initial_responsibility_hash']
    output.update(schema_version=1, experiment='candidate_background_ablation_20261116',
                  seed=context['seed'], observations=context['raw'], cameras=context['cameras'],
                  effective_settings=context['cfg'], inference_or_matching_rerun=False,
                  evaluation_reference_used=False)
    output['timing']['all_methods_optimization_ms'] = sum(m['timing']['optimization_ms'] for m in output['methods'].values())
    output['timing']['solver_wall_ms'] = (time.perf_counter()-started)*1000
    return jsonable(output)
