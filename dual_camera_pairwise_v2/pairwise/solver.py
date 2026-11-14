"""Independent-current-pair mono and true A/G/P stereo estimation.

No input except the whitelisted current observation/matcher arrays is read. The
five method IDs share their robot likelihood, kernels, and convergence budgets.
Monocular poses are initial values only; no pose or E prior occurs in joint BA.
"""
import time
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix, issparse
from .runtime import pair_seed
from .geometry import (camera_arrays, pose, pose_vector, inverse, transform, project,
    triangulate, parallax_degrees, covariance_whiteners, hull_fraction,
    huber_weights, huber_cost, essential_candidates)
from .candidates import (prepare_candidates, responsibilities, candidate_residuals,
                         selected_candidates, candidate_seed_pixels)

CAMERAS = ('camera1', 'camera2')
DEFAULTS = dict(covariance_floor_px=1., huber_delta=3., max_nfev=50,
    irls_iterations=3, max_pose_starts=3, max_joint_starts=3,
    min_rgb_matches=8, max_rgb_matches=80, rgb_noise_std_px=1.5,
    lambda_rgb=1., min_parallax_deg=.15, min_rgb_hull_fraction=.005,
    max_rgb_median_error_px=3., max_rgb_p90_error_px=8.,
    min_baseline_m=.002, max_baseline_m=10., max_reduced_condition=1e12,
    min_effective_robot_per_camera=2., min_effective_robot_total=7.,
    min_effective_rgb_weight=.2, min_positive_depth_fraction=.95,
    max_robot_median_error_px=30., pnp_ransac_threshold_px=12.,
    essential_ransac_threshold_px=2., depth_min_m=.001,
    depth_penalty_scale=1000., pose_translation_scale_m=.5,
    final_rank_relative_tolerance=1e-10, initialization_ambiguity_cost_tolerance=1.,
    initialization_ambiguity_rotation_deg=20., initialization_ambiguity_translation_m=.25, seed=0)


def _settings(config):
    result = dict(DEFAULTS)
    result['seed'] = config.get('seed', DEFAULTS['seed'])
    result.update(config.get('solver', {}))
    matching = config.get('matching', {})
    if 'background_noise_std_px' in matching:
        result['rgb_noise_std_px'] = matching['background_noise_std_px']
    for key in DEFAULTS:
        if key in config.get('geometry', {}):
            result[key] = config['geometry'][key]
    # YAML implementations sometimes parse e.g. 1.0e12 as a string.
    for key, default in DEFAULTS.items():
        result[key] = type(default)(result[key])
    result['candidate_options'] = dict(config.get('candidates', {}))
    result['candidate_options'].setdefault('max_em_iterations', 10)
    return result


def _data(obs, camera, mode, cfg):
    # Deliberately do not copy/read legacy_pose, reference, or any arbitrary keys.
    X = np.array(obs['keypoints_base_m'], float, copy=True)
    uv = np.array(obs['keypoints_uv'], float, copy=True)
    valid = np.array(obs['valid_keypoints'], bool, copy=True)
    covariance = np.array(obs['covariance_uv'], float, copy=True)
    if X.shape != (7, 3) or uv.shape != (7, 2) or covariance.shape != (7, 2, 2) or valid.shape != (7,):
        raise ValueError('Expected original seven FK points, pixels, covariances and flags')
    valid &= np.isfinite(X).all(1) & np.isfinite(uv).all(1) & np.isfinite(covariance).all((1, 2))
    covariance[~valid] = np.eye(2)
    W, regular, eigenvalues = covariance_whiteners(covariance, cfg['covariance_floor_px'])
    prepared = prepare_candidates(obs, camera, cfg['candidate_options']) if mode == 'candidates' else None
    if mode == 'candidates' and prepared is not None:
        valid &= np.array([m['valid'] for m in prepared['modes']])
        for k in np.flatnonzero(~valid):
            prepared['modes'][k]['valid'] = False
    return dict(X=X, uv=uv, valid=valid, W=W, regularized_covariance=regular,
                original_eigenvalues=eigenvalues, prepared=prepared, mode=mode, camera=camera)


def _project_robot(T, data):
    points = transform(T, data['X'])
    return project(points, data['camera']), points[:, 2]


def _robot_state(T, data, cfg):
    uv, depth = _project_robot(T, data)
    if data['mode'] == 'candidates':
        posterior, cost = responsibilities(uv, data['prepared'])
        effective = np.array([1-p['outlier_responsibility'] for p in posterior])
        state = {'posterior':posterior, 'cost':cost, 'effective':effective}
    else:
        errors = np.einsum('kij,kj->ki', data['W'], data['uv']-uv)[data['valid']]
        weights = huber_weights(errors, cfg['huber_delta'])
        effective = np.zeros(7); effective[data['valid']] = weights
        state = {'weights':weights, 'cost':huber_cost(errors, cfg['huber_delta']), 'effective':effective}
    state.update(uv=uv, depth=depth)
    return state


def _robot_residual(T, data, state):
    uv, depth = _project_robot(T, data)
    if data['mode'] == 'candidates':
        blocks = candidate_residuals(uv, data['prepared'], state['posterior'])
    else:
        blocks = np.einsum('kij,kj->ki', data['W'], data['uv']-uv)[data['valid']]
        blocks *= np.sqrt(state['weights'])[:, None]
    return blocks.ravel(), depth[data['valid']]


def _depth_residual(depth, cfg):
    return np.maximum(cfg['depth_min_m']-np.asarray(depth), 0.) * cfg['depth_penalty_scale']


def _exact_huber_rows(blocks, delta):
    """Radial residual with exactly 0.5||r||² = 2D block Huber cost.

    Final polish removes truncated-IRLS drift in the no-cross control. This is
    still one block loss, never component-wise robustification.
    """
    norms=np.linalg.norm(blocks,axis=1)
    twice_cost=np.where(norms<=delta,norms**2,2*delta*(norms-.5*delta))
    return blocks*np.sqrt(twice_cost/np.maximum(norms**2,1e-30))[:,None]


def _mean_exact_residual(T,data,cfg):
    uv,depth=_project_robot(T,data)
    blocks=np.einsum('kij,kj->ki',data['W'],data['uv']-uv)[data['valid']]
    return _exact_huber_rows(blocks,cfg['huber_delta']).ravel(),depth[data['valid']]


def _pnp_seeds(data, seed, cfg):
    arrays = [data['uv']]
    if data['mode'] == 'candidates' and data['prepared']:
        arrays += candidate_seed_pixels(data['prepared'])
    K, D = camera_arrays(data['camera'])
    output = []
    for j, uv in enumerate(arrays):
        good = data['valid'] & np.isfinite(uv).all(1)
        if good.sum() < 4:
            continue
        X = np.ascontiguousarray(data['X'][good]); u = np.ascontiguousarray(uv[good])
        cv2.setRNGSeed(int((seed+j) % (2**31-1)))
        attempts = []
        try:
            ok, r, t, inliers = cv2.solvePnPRansac(X, u, K, D, flags=cv2.SOLVEPNP_EPNP,
                iterationsCount=100, reprojectionError=cfg['pnp_ransac_threshold_px'], confidence=.999)
            if ok:
                attempts.append((r, t))
            ok, r, t = cv2.solvePnP(X, u, K, D, flags=cv2.SOLVEPNP_SQPNP)
            if ok:
                attempts.append((r, t))
        except cv2.error:
            continue
        for r, t in attempts:
            T = pose(np.r_[r.ravel(), t.ravel()])
            if not np.isfinite(T).all() or np.any(transform(T, data['X'])[good, 2] <= cfg['depth_min_m']):
                continue
            if any(np.linalg.norm(T-old) < 1e-5 for old in output):
                continue
            output.append(T)
    output.sort(key=lambda T:_robot_state(T, data, cfg)['cost'])
    return output


def _mono_quality(T, data, state, jac, cfg):
    good = data['valid']
    depth_good = bool(np.all(state['depth'][good] > cfg['depth_min_m']))
    norms = np.linalg.norm(data['uv']-state['uv'], axis=1)
    effective = state['effective']
    scaled = np.asarray(jac) * np.array([1, 1, 1, *([cfg['pose_translation_scale_m']]*3)])
    sv = np.linalg.svd(scaled, compute_uv=False)
    rank = int((sv > (sv[0]*1e-7 if len(sv) else np.inf)).sum())
    quality = {'valid_count':int(good.sum()), 'effective_robot_observations':float(effective.sum()),
               'positive_depth':depth_good, 'rank':rank,
               'raw_mean_median_error_px':float(np.median(norms[good])) if good.any() else None,
               'objective':state['cost']}
    if not np.isfinite(T).all() or not depth_good:
        quality.update(status='monocular_failed', reason='nonfinite_or_negative_depth')
    elif rank < 6 or effective.sum() < 3.5:
        quality.update(status='monocular_failed', reason='insufficient_effective_pose_information')
    elif data['mode'] == 'mean' and quality['raw_mean_median_error_px'] > cfg['max_robot_median_error_px']:
        quality.update(status='monocular_failed', reason='robot_residual_too_large')
    else:
        quality.update(status='monocular_refined', reason=None)
    return quality


def _fit_mono(data, seed, cfg):
    started = time.perf_counter()
    if data['mode'] == 'candidates' and data['prepared'] is None:
        return {'pose':None, 'alternatives':[], 'quality':{'status':'candidates_unavailable','reason':'missing_original_hypotheses'}, 'elapsed_ms':0.}
    if data['valid'].sum() < 4:
        return {'pose':None, 'alternatives':[], 'quality':{'status':'monocular_failed','reason':'fewer_than_four_valid_keypoints','valid_count':int(data['valid'].sum())}, 'elapsed_ms':0.}
    seeds = _pnp_seeds(data, seed, cfg)
    initialized = time.perf_counter()
    fitted = []
    iterations = cfg['candidate_options']['max_em_iterations'] if data['mode'] == 'candidates' else cfg['irls_iterations']
    for initial in seeds[:cfg['max_pose_starts']]:
        x = pose_vector(initial)
        previous_cost = np.inf
        for iteration in range(iterations):
            state = _robot_state(pose(x), data, cfg)
            def residual(value):
                rows, depth = _robot_residual(pose(value), data, state)
                return np.r_[rows, _depth_residual(depth, cfg)]
            opt = least_squares(residual, x, max_nfev=cfg['max_nfev'], x_scale='jac',
                                ftol=1e-8, xtol=1e-8, gtol=1e-8)
            candidate = _robot_state(pose(opt.x), data, cfg)
            if candidate['cost'] > state['cost'] + 1e-6 and np.all(state['depth'][data['valid']] > cfg['depth_min_m']):
                break
            x = opt.x
            if abs(previous_cost-candidate['cost']) < 1e-7*(1+abs(candidate['cost'])):
                break
            previous_cost = candidate['cost']
        if data['mode']=='mean':
            def exact_residual(value):
                rows,depth=_mean_exact_residual(pose(value),data,cfg)
                return np.r_[rows,_depth_residual(depth,cfg)]
            polish=least_squares(exact_residual,x,max_nfev=cfg['max_nfev'],x_scale='jac',
                                 ftol=1e-11,xtol=1e-11,gtol=1e-11)
            x=polish.x
        T = pose(x); state = _robot_state(T, data, cfg)
        # Recompute the Jacobian with FINAL EM/IRLS weights, not stale weights.
        def residual(value):
            rows, depth = _robot_residual(pose(value), data, state)
            return np.r_[rows, _depth_residual(depth, cfg)]
        final = least_squares(residual, x, max_nfev=1)
        quality = _mono_quality(T, data, state, final.jac, cfg)
        quality.update(initializations_tried=min(len(seeds), cfg['max_pose_starts']),
                       objective_type='gaussian_mixture_nll' if data['mode']=='candidates' else '2d_block_huber',
                       exact_block_huber_final_polish=data['mode']=='mean',
                       iterations=iteration+1)
        fitted.append({'pose':T, 'state':state, 'quality':quality})
    fitted.sort(key=lambda v:(v['quality']['status'] != 'monocular_refined', v['state']['cost']))
    if not fitted:
        return {'pose':None, 'alternatives':[], 'quality':{'status':'monocular_failed','reason':'current_raw_pnp_initialization_failed'}, 'elapsed_ms':(time.perf_counter()-started)*1000}
    best = fitted[0]
    alternatives=[]
    for value in fitted:
        if value['quality']['status']=='monocular_refined' and not any(np.linalg.norm(value['pose']-old)<1e-4 for old in alternatives):
            alternatives.append(value['pose'])
    return {'pose':best['pose'] if best['quality']['status']=='monocular_refined' else None,
            'alternatives':alternatives,
            'state':best['state'], 'quality':best['quality'], 'elapsed_ms':(time.perf_counter()-started)*1000,
            'initialization_ms':(initialized-started)*1000,'optimization_ms':(time.perf_counter()-initialized)*1000}


def _match_arrays(matches, cfg):
    if matches is None:
        return np.zeros((0, 2)), np.zeros((0, 2)), {'status':'geometry_failed', 'reason':'missing_actual_rgb_matches'}
    status = matches.get('status')
    hashes=matches.get('provenance',{}).get('image_sha256',matches.get('image_sha256'))
    if isinstance(hashes,(list,tuple)) and len(hashes)==2 and hashes[0]==hashes[1]:
        return np.zeros((0,2)),np.zeros((0,2)),{'status':'geometry_failed','reason':'identical_image_pair'}
    if status in ('blocked_matcher_dependency', 'blocked_missing_pretrained_weights'):
        return np.zeros((0, 2)), np.zeros((0, 2)), {'status':status, 'reason':matches.get('reason', status)}
    p1 = np.array(matches.get('points_c1_native', []), float, copy=True).reshape(-1, 2)
    p2 = np.array(matches.get('points_c2_native', []), float, copy=True).reshape(-1, 2)
    if len(p1) != len(p2):
        raise ValueError('RGB match coordinate lengths differ')
    valid = np.array(matches.get('valid_mask', np.ones(len(p1), bool)), bool, copy=True)
    valid &= np.isfinite(p1).all(1) & np.isfinite(p2).all(1)
    indices = np.flatnonzero(valid)
    # Spatial selection is primarily the matcher adapter's responsibility. A
    # deterministic evenly-spaced cap prevents correlated dense points exploding BA.
    if len(indices) > cfg['max_rgb_matches']:
        indices = indices[np.linspace(0, len(indices)-1, cfg['max_rgb_matches'], dtype=int)]
    return p1[indices], p2[indices], {'input_count':len(p1), 'filter_valid_count':int(valid.sum()),
             'optimizer_selected_indices':indices.tolist(), 'actual_rgb_correspondences_used':len(indices)}


def _metric_initializations(data, mono, p1, p2, cameras, seed, cfg):
    output = []; diagnostics = []
    A1, A2 = mono['camera1']['pose'], mono['camera2']['pose']
    if A1 is not None and A2 is not None:
        for initial1 in mono['camera1']['alternatives'][:2]:
            for initial2 in mono['camera2']['alternatives'][:2]:
                output.append((initial1, initial2 @ inverse(initial1), 'current_pair_monocular'))
    E_candidates, E_info = essential_candidates(p1, p2, cameras, seed, cfg['essential_ransac_threshold_px'])
    diagnostics.append({'route':'rgb_essential', **E_info})
    if A1 is None and A2 is None:
        return output, diagnostics
    scale0 = np.linalg.norm((A2 @ inverse(A1))[:3, 3]) if A1 is not None and A2 is not None else .5
    scale0 = max(.01, min(scale0, cfg['max_baseline_m']))
    for item in E_candidates:
        R, d = item['R'], item['direction']
        G0 = np.eye(4); G0[:3,:3] = R; G0[:3,3] = scale0*d
        anchor = A1 if A1 is not None else inverse(G0) @ A2
        # A valid opposite pose supplies an anchor only at initialization. The
        # following fit uses current raw robot residuals and optimizes A/log(s).
        x = np.r_[pose_vector(anchor), np.log(scale0)]
        states = {cid:_robot_state(anchor if cid=='camera1' else G0@anchor, data[cid], cfg) for cid in CAMERAS}
        def residual(value):
            A = pose(value[:6]); G = np.eye(4); G[:3,:3] = R; G[:3,3] = np.exp(value[6])*d
            rows = []; depths = []
            for cid, T in zip(CAMERAS, (A, G@A)):
                r, dep = _robot_residual(T, data[cid], states[cid]); rows.extend(r); depths.extend(dep)
            return np.r_[rows, _depth_residual(depths, cfg)]
        lo = np.full(7, -np.inf); hi = np.full(7, np.inf)
        lo[-1] = np.log(1e-5); hi[-1] = np.log(cfg['max_baseline_m'])
        try:
            for _ in range(2):
                opt = least_squares(residual, x, bounds=(lo, hi), max_nfev=cfg['max_nfev'], x_scale='jac')
                x = opt.x
                A = pose(x[:6]); G = np.eye(4); G[:3,:3] = R; G[:3,3] = np.exp(x[6])*d
                states = {cid:_robot_state(T, data[cid], cfg) for cid,T in zip(CAMERAS,(A,G@A))}
            output.append((A, G, 'rgb_essential_metric_scale'))
            diagnostics.append({'route':'rgb_essential_metric_scale', 'initial_scale_m':scale0,
                                'fitted_scale_m':float(np.exp(x[6])),
                                'one_sided_monocular_initialization':A1 is None or A2 is None,
                                'unit_positive_fraction':item['unit_positive_fraction']})
        except (ValueError, np.linalg.LinAlgError) as error:
            diagnostics.append({'route':'rgb_essential_metric_scale', 'reason':str(error)})
    return output, diagnostics


def _joint_functions(data, p1, p2, cameras, cfg, x, exact_huber=False):
    n = len(p1)
    def decode(value):
        return pose(value[:6]), pose(value[6:12]), value[12:].reshape(n, 3)
    A, G, P = decode(x)
    states = {cid:_robot_state(T, data[cid], cfg) for cid,T in zip(CAMERAS,(A,G@A))}
    rgb_blocks = np.vstack(((p1-project(P, cameras['camera1']))/cfg['rgb_noise_std_px'],
                           (p2-project(transform(G,P), cameras['camera2']))/cfg['rgb_noise_std_px']))
    rgb_weights = huber_weights(rgb_blocks, cfg['huber_delta'])
    r1, d1 = _robot_residual(A, data['camera1'], states['camera1'])
    r2, d2 = _robot_residual(G@A, data['camera2'], states['camera2'])
    nr1, nr2 = len(r1), len(r2)
    dr1, dr2 = len(d1), len(d2)
    def residual(value):
        A, G, P = decode(value)
        if exact_huber:
            rows1,depth1=_mean_exact_residual(A,data['camera1'],cfg)
            rows2,depth2=_mean_exact_residual(G@A,data['camera2'],cfg)
        else:
            rows1, depth1 = _robot_residual(A, data['camera1'], states['camera1'])
            rows2, depth2 = _robot_residual(G@A, data['camera2'], states['camera2'])
        Q = transform(G, P)
        rgb = np.vstack(((p1-project(P,cameras['camera1']))/cfg['rgb_noise_std_px'],
                         (p2-project(Q,cameras['camera2']))/cfg['rgb_noise_std_px']))
        if exact_huber:
            rgb=_exact_huber_rows(rgb,cfg['huber_delta'])*np.sqrt(cfg['lambda_rgb'])
        else:
            rgb *= np.sqrt(cfg['lambda_rgb']*rgb_weights)[:,None]
        return np.r_[rows1, rows2, rgb.ravel(),
                     _depth_residual(np.r_[depth1,depth2,P[:,2],Q[:,2]], cfg)]
    pattern = lil_matrix((nr1+nr2+6*n+dr1+dr2, 12+3*n), dtype=int)
    pattern[:nr1,:6] = 1; pattern[nr1:nr1+nr2,:12] = 1
    rgb_offset = nr1+nr2
    depth_offset = rgb_offset+4*n
    pattern[depth_offset:depth_offset+dr1,:6] = 1
    pattern[depth_offset+dr1:depth_offset+dr1+dr2,:12] = 1
    for j in range(n):
        cols = slice(12+3*j,15+3*j)
        pattern[rgb_offset+2*j:rgb_offset+2*j+2,cols] = 1
        pattern[rgb_offset+2*n+2*j:rgb_offset+2*n+2*j+2,cols] = 1
        pattern[rgb_offset+2*n+2*j:rgb_offset+2*n+2*j+2,6:12] = 1
        pattern[depth_offset+dr1+dr2+j,cols] = 1
        pattern[depth_offset+dr1+dr2+n+j,cols] = 1
        pattern[depth_offset+dr1+dr2+n+j,6:12] = 1
    cost = sum(v['cost'] for v in states.values()) + cfg['lambda_rgb']*huber_cost(rgb_blocks,cfg['huber_delta'])
    cost += .5*np.sum(_depth_residual(np.r_[d1,d2,P[:,2],transform(G,P)[:,2]],cfg)**2)
    return residual, pattern.tocsr(), states, rgb_weights, float(cost)


def _reduced_information(jac, n, cfg):
    J = jac.toarray() if issparse(jac) else np.asarray(jac)
    scale = np.array([1,1,1,*([cfg['pose_translation_scale_m']]*3)]*2)
    Jp = J[:,:12]*scale
    # Eliminate EACH landmark's 3D nuisance directions in observation space.
    # This is a rank-revealing Schur projection, not pose-only conditional J.
    reduced = Jp.copy()
    for j in range(n):
        L = J[:,12+3*j:15+3*j]
        rows = np.flatnonzero(np.any(L != 0,axis=1))
        if not len(rows):
            continue
        U,s,_ = np.linalg.svd(L[rows],full_matrices=False)
        U = U[:,s>max(s[0]*1e-9,1e-12)]
        reduced[rows] -= U @ (U.T @ Jp[rows])
    eig = np.linalg.eigvalsh(reduced.T@reduced)
    eig = np.maximum(eig,0)
    threshold = max(eig[-1]*cfg['final_rank_relative_tolerance'],1e-12)
    rank = int((eig>threshold).sum())
    condition = float(eig[-1]/max(eig[0],1e-30))
    return {'reduced_rank':rank, 'reduced_condition':condition,
            'reduced_information_eigenvalues':eig.tolist(),
            'nuisance_policy':'per_landmark_rank_revealing_elimination',
            'pose_parameter_scaling':scale.tolist()}


def _joint_quality(x, jac, data, p1, p2, cameras, cfg, states, rgb_weights):
    A,G,P = pose(x[:6]),pose(x[6:12]),x[12:].reshape(-1,3)
    Q = transform(G,P); n = len(P)
    pixels1 = p1-project(P,cameras['camera1']); pixels2 = p2-project(Q,cameras['camera2'])
    errors = np.maximum(np.linalg.norm(pixels1,axis=1),np.linalg.norm(pixels2,axis=1))
    final_weight = np.minimum(rgb_weights[:n],rgb_weights[n:])
    effective = final_weight >= cfg['min_effective_rgb_weight']
    positive = np.isfinite(P).all(1) & np.isfinite(Q).all(1) & (P[:,2]>cfg['depth_min_m']) & (Q[:,2]>cfg['depth_min_m'])
    effective &= positive
    effective_robot = {c:float(states[c]['effective'].sum()) for c in CAMERAS}
    parallax = parallax_degrees(P,G)
    quality = {'actual_rgb_correspondences_used':n, 'effective_rgb_correspondences':int(effective.sum()),
        'effective_rgb_weight_sum':float(final_weight.sum()),
        'effective_robot_observations':effective_robot,
        'rgb_median_error_px':float(np.median(errors)), 'rgb_p90_error_px':float(np.percentile(errors,90)),
        'positive_depth_fraction':float(positive.mean()),
        'parallax_median_deg':float(np.median(parallax[effective])) if effective.any() else 0.,
        'baseline_m':float(np.linalg.norm(G[:3,3])),
        'rgb_hull_fraction':{c:hull_fraction(p[effective],cameras[c]) for c,p in zip(CAMERAS,(p1,p2))},
        'robot_positive_depth':{c:bool(np.all(states[c]['depth'][data[c]['valid']]>cfg['depth_min_m'])) for c in CAMERAS},
        'rgb_final_weights':final_weight.tolist(),
        'robot_final_effective_weights':{c:states[c]['effective'].tolist() for c in CAMERAS}}
    quality.update(_reduced_information(jac,n,cfg))
    reason = None; status = 'stereo_refined'
    if not np.isfinite(x).all() or not all(quality['robot_positive_depth'].values()):
        status,reason = 'joint_pose_failed','nonfinite_or_negative_robot_depth'
    elif quality['baseline_m'] < cfg['min_baseline_m'] or quality['parallax_median_deg'] < cfg['min_parallax_deg']:
        status,reason = 'geometry_failed','small_baseline_or_parallax'
    elif quality['baseline_m'] > cfg['max_baseline_m']:
        status,reason = 'metric_scale_unobservable','baseline_outside_configured_metric_range'
    elif quality['positive_depth_fraction'] < cfg['min_positive_depth_fraction']:
        status,reason = 'geometry_failed','insufficient_positive_depth'
    elif quality['effective_rgb_correspondences'] < cfg['min_rgb_matches']:
        status,reason = 'geometry_failed','too_few_effective_rgb_matches'
    elif min(quality['rgb_hull_fraction'].values()) < cfg['min_rgb_hull_fraction']:
        status,reason = 'geometry_failed','insufficient_rgb_spatial_distribution'
    elif quality['rgb_median_error_px'] > cfg['max_rgb_median_error_px'] or quality['rgb_p90_error_px'] > cfg['max_rgb_p90_error_px']:
        status,reason = 'geometry_failed','rgb_reprojection_inconsistent'
    elif min(effective_robot.values()) < cfg['min_effective_robot_per_camera'] or sum(effective_robot.values()) < cfg['min_effective_robot_total']:
        status,reason = 'metric_scale_unobservable','insufficient_final_robot_information'
    elif quality['reduced_rank'] < 12 or quality['reduced_condition'] > cfg['max_reduced_condition']:
        status,reason = 'metric_scale_unobservable','reduced_pose_system_rank_or_condition'
    elif data['camera1']['mode']=='mean' and any(np.median(np.linalg.norm(data[c]['uv'][data[c]['valid']]-states[c]['uv'][data[c]['valid']],axis=1)) > cfg['max_robot_median_error_px'] for c in CAMERAS if data[c]['valid'].any()):
        status,reason = 'joint_pose_failed','robot_residual_too_large'
    quality.update(status=status,reason=reason)
    return quality


def _fit_joint(data, mono, matches, cameras, seed, cfg):
    started = time.perf_counter()
    p1,p2,match_info = _match_arrays(matches,cfg)
    if len(p1)<cfg['min_rgb_matches']:
        return {'status':match_info.get('status','geometry_failed'), 'reason':match_info.get('reason','too_few_actual_rgb_matches'),
                'quality':match_info, 'timing':{'total_ms':(time.perf_counter()-started)*1000,'initialization_ms':0.,'ba_ms':0.}}
    if any(data[c]['mode']=='candidates' and data[c]['prepared'] is None for c in CAMERAS):
        return {'status':'candidates_unavailable','reason':'missing_original_hypotheses','quality':match_info,'timing':{'total_ms':0.,'initialization_ms':0.,'ba_ms':0.}}
    if any(data[c]['valid'].sum()==0 for c in CAMERAS):
        return {'status':'metric_scale_unobservable','reason':'one_camera_has_no_robot_observations','quality':match_info,'timing':{'total_ms':0.,'initialization_ms':0.,'ba_ms':0.}}
    initializations, init_diagnostics = _metric_initializations(data,mono,p1,p2,cameras,seed,cfg)
    initialized = time.perf_counter()
    proposals = []
    for A,G,route in initializations:
        P = triangulate(p1,p2,G,cameras)
        positive = np.isfinite(P).all(1) & (P[:,2]>.001) & (transform(G,P)[:,2]>.001)
        if positive.mean()<.5:
            continue
        # A fixed subset is chosen only using current geometric initialization.
        # All selected matches remain in the objective; bad depths carry penalties.
        P[~np.isfinite(P).all(1)] = [0,0,2.]
        x = np.r_[pose_vector(A),pose_vector(G),P.ravel()]
        cost = _joint_functions(data,p1,p2,cameras,cfg,x)[-1]
        proposals.append((cost,x,route))
    proposals.sort(key=lambda v:v[0])
    fitted=[]
    iterations = cfg['candidate_options']['max_em_iterations'] if data['camera1']['mode']=='candidates' else cfg['irls_iterations']
    for _,initial,route in proposals[:cfg['max_joint_starts']]:
        x=initial.copy(); old=np.inf; trace=[]
        try:
            for iteration in range(iterations):
                fun,pattern,states,weights,cost = _joint_functions(data,p1,p2,cameras,cfg,x)
                opt=least_squares(fun,x,jac_sparsity=pattern,tr_solver='lsmr',x_scale='jac',
                                  max_nfev=cfg['max_nfev'],ftol=1e-8,xtol=1e-8,gtol=1e-8)
                newcost=_joint_functions(data,p1,p2,cameras,cfg,opt.x)[-1]
                if newcost>cost+1e-5:
                    trace.append({'objective':cost,'rejected_increasing_update':newcost}); break
                x=opt.x; trace.append({'objective':newcost,'nfev':opt.nfev,'optimizer_success':bool(opt.success)})
                if abs(old-newcost)<1e-7*(1+abs(newcost)):
                    break
                old=newcost
            if data['camera1']['mode']=='mean':
                fun,pattern,_,_,before=_joint_functions(data,p1,p2,cameras,cfg,x,exact_huber=True)
                polish=least_squares(fun,x,jac_sparsity=pattern,x_scale='jac',max_nfev=cfg['max_nfev'],
                                     ftol=1e-11,xtol=1e-11,gtol=1e-11)
                after=_joint_functions(data,p1,p2,cameras,cfg,polish.x)[-1]
                if after<=before+1e-6:
                    x=polish.x
                trace.append({'objective':min(before,after),'nfev':polish.nfev,'exact_block_huber_final_polish':True})
            fun,pattern,states,weights,cost = _joint_functions(data,p1,p2,cameras,cfg,x)
            final=least_squares(fun,x,jac_sparsity=pattern,max_nfev=1)
            quality=_joint_quality(x,final.jac,data,p1,p2,cameras,cfg,states,weights)
            quality.update(objective=cost,objective_trace=trace,initialization_route=route,
                optimizer_nfev_total=sum(v.get('nfev',0) for v in trace),
                initialization_pose_change_norm=float(np.linalg.norm(x[:12]-initial[:12])))
            fitted.append({'x':x,'states':states,'quality':quality,'cost':cost})
        except (ValueError,np.linalg.LinAlgError,cv2.error) as error:
            init_diagnostics.append({'route':route,'optimization_error':str(error)})
    timing={'total_ms':(time.perf_counter()-started)*1000,'initialization_ms':(initialized-started)*1000,
            'ba_ms':(time.perf_counter()-initialized)*1000}
    if not fitted:
        status = 'metric_scale_unobservable' if not initializations else 'geometry_failed'
        return {'status':status,'reason':'no_valid_current_pair_metric_initialization',
                'quality':{**match_info,'initialization_diagnostics':init_diagnostics},'timing':timing}
    fitted.sort(key=lambda v:(v['quality']['status']!='stereo_refined',v['cost']))
    best=fitted[0]
    disagreement=[]
    for other in fitted[1:]:
        if other['quality']['status']!='stereo_refined':
            continue
        detail={'objective_delta':float(other['cost']-best['cost']),'poses':{}}
        for label,offset in (('A',0),('G',6)):
            reference=pose(best['x'][offset:offset+6]); alternative=pose(other['x'][offset:offset+6])
            detail['poses'][label]={'rotation_deg':float(np.linalg.norm(cv2.Rodrigues(alternative[:3,:3]@reference[:3,:3].T)[0])*180/np.pi),
                                    'translation_m':float(np.linalg.norm(alternative[:3,3]-reference[:3,3]))}
        detail['near_equal_cost']=abs(detail['objective_delta'])<=cfg['initialization_ambiguity_cost_tolerance']
        detail['large_disagreement']=any(d['rotation_deg']>cfg['initialization_ambiguity_rotation_deg'] or d['translation_m']>cfg['initialization_ambiguity_translation_m'] for d in detail['poses'].values())
        disagreement.append(detail)
    ambiguity=any(d['near_equal_cost'] and d['large_disagreement'] for d in disagreement)
    if ambiguity:
        best['quality'].update(status='joint_pose_failed',reason='competing_near_equal_cost_pose_initializations')
    best['quality'].update(match_input=match_info,initialization_diagnostics=init_diagnostics,
        initializations_tested=len(fitted),valid_initialization_pose_disagreements=disagreement,
        competing_initializations_ambiguous=ambiguity)
    if best['quality']['status']!='stereo_refined':
        return {'status':best['quality']['status'],'reason':best['quality']['reason'],
                'quality':best['quality'],'timing':timing}
    x=best['x']; A,G=pose(x[:6]),pose(x[6:12])
    return {'status':'stereo_refined','reason':None,'poses':{'camera1':A,'camera2':G@A},
            'stereo_pose':{'camera1':A,'camera2':G@A,'T_C2_C1':G},'landmarks_c1_m':x[12:].reshape(-1,3),
            'states':best['states'],'quality':best['quality'],'timing':timing}


def _attach_outputs(result, data, states):
    result['posterior']={}; result['candidate_selected_uv']={}; result['candidate_selected_valid']={}; result['pose_projected']={}
    result['candidate_model']={}
    result['regularized_covariance_uv']={c:data[c]['regularized_covariance'] for c in CAMERAS}
    result['raw_covariance_eigenvalues']={c:data[c]['original_eigenvalues'] for c in CAMERAS}
    for cid in CAMERAS:
        T=result['poses'].get(cid)
        if T is None:
            result['pose_projected'][cid]=None
            result['candidate_selected_uv'][cid]=None; result['candidate_selected_valid'][cid]=[False]*7
            result['posterior'][cid]=None
            continue
        projected=project(transform(T,data[cid]['X']),data[cid]['camera'])
        result['pose_projected'][cid]=projected
        prepared=data[cid]['prepared']
        if prepared is None:
            result['posterior'][cid]=None; result['candidate_selected_uv'][cid]=None
            result['candidate_selected_valid'][cid]=[False]*7
        else:
            post,_=responsibilities(projected,prepared)
            uv,valid=selected_candidates(prepared,post)
            result['posterior'][cid]=post; result['candidate_selected_uv'][cid]=uv
            result['candidate_selected_valid'][cid]=valid
            result['candidate_model'][cid]=prepared


def _fit_no_cross(data, mono, cameras, cfg):
    """Actually optimize free A/G with zero RGB rows and independent mono starts."""
    started=time.perf_counter()
    poses={c:mono[c]['pose'] for c in CAMERAS}
    quality={'actual_rgb_correspondences_used':0,'temporal_information_used':False,
             'separable_objective':True,'relative_pose_free':True,
             'implementation':'joint_free_A_G_optimization_with_no_RGB_or_pose_prior',
             'max_pose_difference_from_mono_mean':None}
    if not all(v is not None for v in poses.values()):
        return dict(poses=poses,status='independent_no_cross_partial',quality=quality,
                    stereo_pose=None,fallback_pose=None,T_C2_C1=None,
                    timing={'total_ms':(time.perf_counter()-started)*1000,'initialization_ms':0.,'ba_ms':0.})
    A=poses['camera1']; G=poses['camera2']@inverse(A)
    x=np.r_[pose_vector(A),pose_vector(G)]
    empty=np.empty((0,2)); trace=[]
    fun,pattern,_,_,cost=_joint_functions(data,empty,empty,cameras,cfg,x,exact_huber=True)
    # Dense 12-column numeric Jacobian avoids LSMR stopping differences in this
    # tiny diagnostic. A/G remain jointly free and all robot factors are present.
    opt=least_squares(fun,x,x_scale='jac',max_nfev=cfg['max_nfev'],
                      ftol=1e-11,xtol=1e-11,gtol=1e-11)
    x=opt.x; trace.append({'objective':cost,'nfev':opt.nfev,'exact_block_huber':True})
    A,G=pose(x[:6]),pose(x[6:12]); fitted={'camera1':A,'camera2':G@A}
    differences={c:{'matrix_max_abs':float(np.max(np.abs(fitted[c]-poses[c]))),
        'translation_m':float(np.linalg.norm(fitted[c][:3,3]-poses[c][:3,3])),
        'rotation_deg':float(np.linalg.norm(cv2.Rodrigues(fitted[c][:3,:3]@poses[c][:3,:3].T)[0])*180/np.pi)} for c in CAMERAS}
    quality.update(max_pose_difference_from_mono_mean=max(d['matrix_max_abs'] for d in differences.values()),
                   pose_difference_from_mono_mean=differences,objective_trace=trace,
                   factor_rows=len(fun(x)),joint_parameter_count=12)
    return dict(poses=fitted,status='independent_no_cross',quality=quality,stereo_pose=None,
                fallback_pose=None,T_C2_C1=G,reason=None,
                timing={'total_ms':(time.perf_counter()-started)*1000,'initialization_ms':0.,'ba_ms':(time.perf_counter()-started)*1000})


def solve_pair(pair_id, observations, cameras, matches=None, config=None, methods=None):
    """Estimate requested methods from one pair; no state persists between calls.

    Failed dual poses remain null. Operational fallback is a separate field and
    cannot be counted as stereo by inspecting poses. dual_no_cross optimizes free
    A/G with zero RGB factors, initialized at the independent monocular optima.
    """
    config=config or {}; cfg=_settings(config)
    methods=list(methods or ['mono_mean','mono_candidates','dual_mean','dual_candidates','dual_no_cross'])
    allowed={'mono_mean','mono_candidates','dual_mean','dual_candidates','dual_no_cross'}
    if not set(methods)<=allowed:
        raise ValueError('Unknown method ID')
    seed=pair_seed(str(pair_id),cfg['seed']); output={}
    for mode in ('mean','candidates'):
        requested=[m for m in methods if m.endswith(mode)]
        if mode=='mean' and 'dual_no_cross' in methods: requested+=['dual_no_cross']
        if not requested: continue
        started=time.perf_counter()
        data={c:_data(observations[c],cameras[c],mode,cfg) for c in CAMERAS}
        mono={c:_fit_mono(data[c],(seed+i)% (2**31-1),cfg) for i,c in enumerate(CAMERAS)}
        mono_poses={c:mono[c]['pose'] for c in CAMERAS}
        mono_result={'poses':mono_poses,'status':'monocular_refined' if all(v is not None for v in mono_poses.values()) else 'monocular_partial' if any(v is not None for v in mono_poses.values()) else 'candidates_unavailable' if mode=='candidates' and any(data[c]['prepared'] is None for c in CAMERAS) else 'monocular_failed',
            'stereo_pose':None,'fallback_pose':None,'reason':None,
            'mono_quality':{c:mono[c]['quality'] for c in CAMERAS},
            'quality':{'actual_rgb_correspondences_used':0,'temporal_information_used':False},
            'timing':{'total_ms':(time.perf_counter()-started)*1000,
                      'initialization_ms':sum(mono[c].get('initialization_ms',0.) for c in CAMERAS),
                      'ba_ms':sum(mono[c].get('optimization_ms',0.) for c in CAMERAS),
                      'monocular_ms_by_camera':{c:mono[c]['elapsed_ms'] for c in CAMERAS}},
            'seed':seed,'epipolar_gating_skipped':True}
        _attach_outputs(mono_result,data,{c:mono[c].get('state') for c in CAMERAS})
        if 'mono_'+mode in methods: output['mono_'+mode]=mono_result
        if mode=='mean' and 'dual_no_cross' in methods:
            no_cross=_fit_no_cross(data,mono,cameras,cfg)
            no_cross['mono_quality']=mono_result['mono_quality']
            no_cross['seed']=seed; no_cross['epipolar_gating_skipped']=True
            no_cross['timing']['total_ms']+=mono_result['timing']['total_ms']
            no_cross['timing']['monocular_initialization_ms']=mono_result['timing']['total_ms']
            _attach_outputs(no_cross,data,{})
            output['dual_no_cross']=no_cross
        if 'dual_'+mode in methods:
            result=_fit_joint(data,mono,matches,cameras,seed,cfg)
            result.setdefault('poses',{c:None for c in CAMERAS}); result.setdefault('stereo_pose',None)
            result['fallback_pose']=mono_poses if result['stereo_pose'] is None else None
            result['fallback_status']='single_view_fallback' if result['stereo_pose'] is None and any(v is not None for v in mono_poses.values()) else None
            result['mono_quality']={c:mono[c]['quality'] for c in CAMERAS}
            result['seed']=seed; result['epipolar_gating_skipped']=True
            result['timing']['monocular_initialization_ms']=mono_result['timing']['total_ms']
            result['timing']['total_ms']+=mono_result['timing']['total_ms']
            _attach_outputs(result,data,result.pop('states',{}))
            output['dual_'+mode]=result
    return output
