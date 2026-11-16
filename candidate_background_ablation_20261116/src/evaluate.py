"""Evaluation-only full-denominator controlled-ablation metrics.

Reference poses are opened only here, never by the inference adapter. ADD and
PCK conventions follow read-only pairwise/evaluation.py (strict < threshold).
The bootstrap resamples original time blocks before applying any validity mask.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
for name, suffix in {'MPLCONFIGDIR': 'mpl', 'XDG_CACHE_HOME': 'xdg', 'TMPDIR': 'tmp'}.items():
    directory = ROOT / 'cache' / suffix
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(directory)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import cv2
import numpy as np
import yaml

METHODS = ('R0', 'R1', 'E0', 'E1')
CAMERAS = ('camera1', 'camera2')
CONTRASTS = (('R1-R0', 'R0', 'R1'), ('E1-E0', 'E0', 'E1'),
             ('E0-R0', 'R0', 'E0'), ('E1-R1', 'R1', 'E1'))
THRESHOLDS = np.arange(0., 100.01, 5.)
PCK_THRESHOLDS = np.arange(0., 20.001, .25)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, Path): return str(value)
    return value


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(clean(value), f, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: clean(v) for k, v in row.items()} for row in rows])


def jsonl(path):
    rows = [json.loads(line) for line in Path(path).open() if line.strip()]
    ids = [str(r['pair_id']) for r in rows]
    if len(ids) != len(set(ids)): raise ValueError(f'Duplicate pair IDs: {path}')
    return {str(r['pair_id']): r for r in rows}


def config_paths(config_path):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text())
    def resolve(value):
        p = Path(value).expanduser()
        return p.resolve() if p.is_absolute() else (config_path.parent / p).resolve()
    inputs = config.get('inputs', config.get('paths', {}))
    paths = {}
    for name, aliases in {'pairs': ('pairs', 'pairs_manifest'),
                          'reference': ('reference', 'reference_manifest'),
                          'observations': ('observations', 'observations_manifest'),
                          'candidates': ('candidates', 'candidates_manifest'),
                          'matches': ('matches', 'matches_manifest')}.items():
        value = next((inputs[k] for k in aliases if k in inputs), None)
        if value is None: value = next((config[k] for k in aliases if isinstance(config.get(k),(str,os.PathLike))), None)
        if value is not None: paths[name] = resolve(value)
    reference = config.get('reference', config.get('production_reference', {}))
    if isinstance(reference, dict):
        for name in ('production_run', 'production_evaluation'):
            if reference.get(name): paths[name] = resolve(reference[name])
    return config, paths


def pose_array(value):
    try: T = np.asarray(value, dtype=float)
    except (ValueError, TypeError): return None
    if T.shape != (4, 4) or not np.isfinite(T).all(): return None
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-7): return None
    if not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-4): return None
    if not np.isclose(np.linalg.det(T[:3, :3]), 1., atol=1e-4): return None
    return T


def project(T, xyz, camera):
    uv = np.full((7, 2), np.nan)
    if T is None: return uv
    pts = xyz @ T[:3, :3].T + T[:3, 3]
    valid = np.isfinite(pts).all(1) & (pts[:, 2] > .001)
    if valid.any():
        uv[valid] = cv2.projectPoints(pts[valid], np.zeros(3), np.zeros(3),
            np.asarray(camera['K'], float), np.asarray(camera.get('D', []), float))[0].reshape(-1, 2)
    return uv


def pck_errors(prediction, reference, camera, valid=None):
    ref = np.asarray(reference, float)
    mask = np.isfinite(ref).all(1) & (ref[:, 0] >= 0) & (ref[:, 0] < camera.get('width', 640))
    mask &= (ref[:, 1] >= 0) & (ref[:, 1] < camera.get('height', 480))
    errors = np.full(7, np.nan); errors[mask] = np.inf
    try: uv = np.asarray(prediction, float)
    except (ValueError, TypeError): return errors
    if uv.shape != (7, 2): return errors
    ok = mask & np.isfinite(uv).all(1)
    if valid is not None: ok &= np.asarray(valid, bool)
    errors[ok] = np.linalg.norm(uv[ok] - ref[ok], axis=1)
    return errors


def auc_scores(errors):
    """Per-observation exact 0:5:100 trapezoidal contribution, failures zero."""
    return np.trapezoid((np.asarray(errors)[..., None] < THRESHOLDS).astype(float),
                        THRESHOLDS, axis=-1) / 100.


def distribution(errors):
    values = np.asarray(errors, float).ravel()
    finite = values[np.isfinite(values)]
    curve = np.mean(values[:, None] < THRESHOLDS, axis=0) if len(values) else np.full(21, np.nan)
    return dict(n_total=len(values), n_valid=len(finite), n_failed=len(values)-len(finite),
        ADD50=float(np.mean(values < 50.)) if len(values) else None,
        ADD_AUC=float(np.trapezoid(curve, THRESHOLDS) / 100.) if len(values) else None,
        ADD_mean_mm=float(finite.mean()) if len(finite) else None,
        ADD_median_mm=float(np.median(finite)) if len(finite) else None,
        ADD_p90_mm=float(np.percentile(finite, 90)) if len(finite) else None), curve


def make_time_blocks(timestamps, seconds):
    """Nonoverlapping fixed-origin time bins; no validity-based compression."""
    ts = np.asarray(timestamps, float)
    if len(ts) == 0 or not np.isfinite(ts).all(): raise ValueError('Raw timestamps unavailable')
    if np.any(np.diff(ts) < 0): raise ValueError('Pairs must be sorted by original time')
    raw = np.floor((ts-ts[0])/float(seconds)).astype(np.int64)
    labels, assignment = np.unique(raw, return_inverse=True)
    return assignment, dict(seconds=float(seconds), origin_s=float(ts[0]),
        original_bin_labels=labels.tolist(), n_blocks=len(labels), n_pairs=len(ts),
        pair_counts=np.bincount(assignment).tolist(), empty_time_bins=int(raw[-1]+1-len(labels)))


def bootstrap_block_counts(timestamps, seconds, n_resamples=2000, seed=0):
    assignment, meta = make_time_blocks(timestamps, seconds)
    rng = np.random.default_rng(seed)
    # Every replicate draws the original number of occupied time blocks.
    draws = rng.integers(0, meta['n_blocks'], size=(n_resamples, meta['n_blocks']))
    counts = np.zeros((n_resamples, meta['n_blocks']), dtype=np.int32)
    np.add.at(counts, (np.arange(n_resamples)[:, None], draws), 1)
    meta.update(n_resamples=n_resamples, seed=seed,
        scheme='nonoverlapping_fixed_origin_time_blocks_with_replacement',
        two_cameras_stay_together=True, all_methods_same_indices=True,
        masks_applied_after_resampling=True,
        sample_pair_count_min=int((counts @ np.bincount(assignment)).min()),
        sample_pair_count_max=int((counts @ np.bincount(assignment)).max()))
    return assignment, counts, meta


def bootstrap_ratios(numerators, denominators, assignment, counts):
    """Block sum then resample = resample raw indices then apply masks."""
    numerator = np.asarray(numerators, float); denominator = np.asarray(denominators, float)
    if numerator.ndim == 1: numerator = numerator[:, None]
    if denominator.ndim == 1: denominator = denominator[:, None]
    bnum = np.zeros((counts.shape[1], numerator.shape[1]))
    bden = np.zeros((counts.shape[1], denominator.shape[1]))
    np.add.at(bnum, assignment, numerator); np.add.at(bden, assignment, denominator)
    ns = counts @ bnum; ds = counts @ bden
    return np.divide(ns, ds, out=np.full_like(ns, np.nan), where=ds>0)


def ci(values):
    values = np.asarray(values); finite = values[np.isfinite(values)]
    return dict(low=float(np.percentile(finite, 2.5)) if len(finite) else None,
                high=float(np.percentile(finite, 97.5)) if len(finite) else None,
                valid_replicates=len(finite), invalid_replicates=len(values)-len(finite))


def comparison_statistics(errors, eligibility, camera_indices):
    """Build all paired contrasts and interaction on unchanged original pairs."""
    e = np.asarray(errors)[:, :, camera_indices]
    included = np.asarray(eligibility, bool)[:, None]
    rows, nums, dens = [], [], []
    for label, before, after in CONTRASTS:
        a, b = e[:, METHODS.index(before)], e[:, METHODS.index(after)]
        common = np.isfinite(a) & np.isfinite(b) & included
        delta = np.where(common, b, 0)-np.where(common, a, 0)
        selected = delta[common]
        whole = np.broadcast_to(included, a.shape)
        row = dict(contrast=label, before=before, after=after,
            n_denominator=int(whole.sum()), n_common_valid=int(common.sum()),
            n_common_pairs=int(common.any(1).sum()),
            paired_ADD_delta_mm=float(selected.mean()) if len(selected) else None,
            improved_fraction=float(np.mean(selected < 0)) if len(selected) else None,
            worsened_fraction=float(np.mean(selected > 0)) if len(selected) else None,
            tied_fraction=float(np.mean(selected == 0)) if len(selected) else None,
            ADD50_delta=float((((b < 50).astype(float)-(a < 50))*whole).sum()/whole.sum()) if whole.sum() else None)
        # Keep numerator/denominator separately; masks are NOT applied to time axis.
        auc_delta = (auc_scores(b)-auc_scores(a))*whole
        row['ADD_AUC_delta'] = float(auc_delta.sum()/whole.sum()) if whole.sum() else None
        rows.append(row)
        nums.extend([delta.sum(1), (((b < 50).astype(float)-(a < 50))*whole).sum(1), auc_delta.sum(1)])
        dens.extend([common.sum(1), whole.sum(1), whole.sum(1)])
    common = np.isfinite(e).all(1) & included
    safe = np.where(common[:, None, :], e, 0)
    interaction = (safe[:, 3]-safe[:, 2])-(safe[:, 1]-safe[:, 0])
    selected = interaction[common]
    rows.append(dict(contrast='(E1-E0)-(R1-R0)', before='R1-R0', after='E1-E0',
        n_denominator=int(np.broadcast_to(included, common.shape).sum()),
        n_common_valid=int(common.sum()), n_common_pairs=int(common.any(1).sum()),
        paired_ADD_delta_mm=float(selected.mean()) if len(selected) else None,
        improved_fraction=None, worsened_fraction=None, tied_fraction=None,
        ADD50_delta=None, ADD_AUC_delta=None))
    nums.extend([interaction.sum(1), np.zeros(len(e)), np.zeros(len(e))])
    dens.extend([common.sum(1), np.zeros(len(e)), np.zeros(len(e))])
    return rows, np.stack(nums, axis=1), np.stack(dens, axis=1)


def _camera_value(value, camera, default=None):
    return value.get(camera, default) if isinstance(value, dict) else default


def load_arrays(results, pairs, refs):
    ids = sorted(results, key=lambda pid: (np.mean([pairs[pid]['cameras'][c]['timestamp_s'] for c in CAMERAS]), pid))
    n = len(ids); add = np.full((n, 4, 2), np.inf)
    flags = {gate: np.zeros((n, 4, 2), bool) for gate in ('common', 'final')}
    pcks = {track: np.full((n, 4, 2, 7), np.nan) for track in ('raw', 'map', 'projected')}
    metadata, failures, timings = [], [], []
    eligible = np.array([bool(results[p]['pre_eligible']) for p in ids])
    timestamps = np.array([np.mean([pairs[p]['cameras'][c]['timestamp_s'] for c in CAMERAS]) for p in ids])
    for i, pid in enumerate(ids):
        result = results[pid]
        if set(result['methods']) != set(METHODS): raise ValueError(f'Missing method: {pid}')
        for j, method in enumerate(METHODS):
            m = result['methods'][method]
            for k, camera in enumerate(CAMERAS):
                cam = pairs[pid]['cameras'][camera]; ref = refs[pid]['cameras'][camera]
                xyz = np.asarray(cam['keypoints_base_m'], float)
                if not np.allclose(xyz, ref['keypoints_base_m'], atol=1e-7, rtol=0): raise ValueError(f'FK mismatch {pid}/{camera}')
                T = pose_array(_camera_value(m.get('poses'), camera))
                Tr = pose_array(ref['T_camera_from_base'])
                if Tr is None: raise ValueError('Malformed reference pose')
                if T is not None:
                    d = xyz @ (T[:3, :3]-Tr[:3, :3]).T + T[:3, 3]-Tr[:3, 3]
                    add[i, j, k] = np.linalg.norm(d, axis=1).mean()*1000
                for gate in flags:
                    flag = m.get(gate+'_valid', False)
                    flag = flag.get(camera, False) if isinstance(flag, dict) else flag
                    if flag and T is None: raise ValueError(f'Valid flag without finite proper pose: {pid}/{method}')
                    flags[gate][i,j,k] = bool(flag) and T is not None
                if flags['final'][i,j,k] and not flags['common'][i,j,k]: raise ValueError('Final gate cannot bypass common gate')
                raw = _camera_value(m.get('raw_inputs'), camera, {})
                uv = raw.get('keypoints_uv', raw.get('mean', raw.get('means', raw.get('uv')))) if isinstance(raw, dict) else raw
                valid = raw.get('valid_keypoints', raw.get('valid', raw.get('valid_mask'))) if isinstance(raw, dict) else None
                pcks['raw'][i,j,k] = pck_errors(uv, ref['projected_keypoints_uv'], cam, valid)
                pcks['map'][i,j,k] = pck_errors(_camera_value(m.get('candidate_selected_uv'), camera), ref['projected_keypoints_uv'], cam,
                    _camera_value(m.get('candidate_selected_valid'), camera))
                pcks['projected'][i,j,k] = pck_errors(project(T, xyz, cam), ref['projected_keypoints_uv'], cam)
                metadata.append(dict(pair_id=pid, timestamp_s=float(timestamps[i]), method=method, camera=camera,
                    pre_eligible=bool(eligible[i]), common_valid=bool(flags['common'][i,j,k]), final_valid=bool(flags['final'][i,j,k]),
                    add_before_gate_mm=float(add[i,j,k]) if np.isfinite(add[i,j,k]) else None,
                    common_ADD_mm=float(add[i,j,k]) if flags['common'][i,j,k] else 'inf',
                    final_ADD_mm=float(add[i,j,k]) if flags['final'][i,j,k] else 'inf',
                    status=m.get('status'), reason=m.get('reason')))
            timings.append(dict(pair_id=pid, method=method, **m.get('timing', {}),
                trace_json=json.dumps(clean(m.get('trace')), ensure_ascii=False)))
            for gate in ('common', 'final'):
                if not flags[gate][i,j].all():
                    reasons = m.get('common_gate', {}).get('reasons', []) if gate=='common' else m.get('reason', '')
                    failures.append(dict(pair_id=pid, method=method, gate=gate,
                        status=m.get('status'), reason=json.dumps(reasons, ensure_ascii=False) if not isinstance(reasons,str) else reasons,
                        pre_eligible=bool(eligible[i])))
    # Raw observations must be identical across all four conditions, missing included.
    for j in range(1,4):
        if not np.array_equal(pcks['raw'][:,0], pcks['raw'][:,j], equal_nan=True): raise ValueError('Raw observations differ between conditions')
    return dict(ids=ids, timestamps=timestamps, add=add, valid=flags, pcks=pcks,
                eligible=eligible, rows=metadata, failures=failures, timings=timings)


def evaluate_arrays(arrays, bootstrap_settings=None):
    settings = bootstrap_settings or {}; repetitions = int(settings.get('n_resamples', settings.get('repetitions', 2000)))
    block_lengths = settings.get('block_seconds', [1., .5, 2.]); seed=int(settings.get('seed', 0))
    if np.isscalar(block_lengths): block_lengths=[float(block_lengths)]
    summaries, curves, comparisons, nums, dens, group_sizes = [], {}, [], [], [], []
    method_nums,method_dens=[],[]
    for gate in ('final', 'common'):
        errors = np.where(arrays['valid'][gate], arrays['add'], np.inf)
        for population in ('full', 'pre_eligible'):
            selected = np.ones(len(errors), bool) if population=='full' else arrays['eligible']
            for camera, indexes in [('pooled', [0,1]), ('camera1', [0]), ('camera2', [1])]:
                key = f'{gate}/{population}/{camera}'; curves[key]={}
                for j, method in enumerate(METHODS):
                    values=errors[selected,j][:,indexes]
                    row, curve=distribution(values)
                    row.update(gate=gate,population=population,camera=camera,method=method,n_pairs=int(selected.sum()))
                    fullvalues=errors[:,j][:,indexes]
                    whole=np.broadcast_to(selected[:,None],fullvalues.shape)
                    finite=np.isfinite(fullvalues)&whole
                    method_nums.extend([((fullvalues<50)*whole).sum(1),(auc_scores(fullvalues)*whole).sum(1),np.where(finite,fullvalues,0).sum(1)])
                    method_dens.extend([whole.sum(1),whole.sum(1),finite.sum(1)])
                    curves[key][method]={'thresholds_mm':THRESHOLDS.tolist(),'ADD':curve.tolist()}
                    for track, allpoints in arrays['pcks'].items():
                        points=allpoints[selected,j][:,indexes].copy()
                        # raw PCK is input-only; MAP/projected report both pre-gate
                        # diagnostics and adoption-gated full-population scores.
                        diagnostics=points[~np.isnan(points)]
                        row[track+'_PCK5_before_gate']=float(np.mean(diagnostics<5)) if len(diagnostics) else None
                        if track!='raw':
                            failed=~arrays['valid'][gate][selected,j][:,indexes]
                            points[np.broadcast_to(failed[...,None],points.shape)&~np.isnan(points)] = np.inf
                        ev=points[~np.isnan(points)]
                        row[track+'_PCK5']=float(np.mean(ev<5)) if len(ev) else None
                        row[track+'_reference_points']=len(ev)
                    summaries.append(row)
                prows,pnum,pden=comparison_statistics(errors, selected, indexes)
                for row in prows: row.update(gate=gate,population=population,camera=camera)
                comparisons.extend(prows); nums.append(pnum);dens.append(pden);group_sizes.append(len(prows))
    numerator=np.concatenate(nums,axis=1);denominator=np.concatenate(dens,axis=1)
    method_numerator=np.stack(method_nums,axis=1);method_denominator=np.stack(method_dens,axis=1)
    sensitivity, method_sensitivity, bootmeta, bootstrap_data = [], [], [], {}
    for seconds in block_lengths:
        assignment, counts, meta=bootstrap_block_counts(arrays['timestamps'],seconds,repetitions,seed)
        samples=bootstrap_ratios(numerator,denominator,assignment,counts)
        method_samples=bootstrap_ratios(method_numerator,method_denominator,assignment,counts)
        bootmeta.append(meta);bootstrap_data[str(seconds)]=dict(assignment=assignment,counts=counts)
        for i,row in enumerate(summaries):
            record={k:row[k] for k in ('gate','population','camera','method')};record['block_seconds']=seconds
            for offset,metric in enumerate(('ADD50','ADD_AUC','ADD_mean_mm')):
                interval=ci(method_samples[:,i*3+offset]);record[metric]=row[metric]
                record[metric+'_ci_low']=interval['low'];record[metric+'_ci_high']=interval['high']
                record[metric+'_bootstrap_valid']=interval['valid_replicates'];record[metric+'_bootstrap_empty']=interval['invalid_replicates']
                if seconds==1.:
                    for suffix in ('ci_low','ci_high','bootstrap_valid','bootstrap_empty'):row[metric+'_'+suffix]=record[metric+'_'+suffix]
            method_sensitivity.append(record)
        for i,row in enumerate(comparisons):
            record={k:row[k] for k in ('gate','population','camera','contrast')};record['block_seconds']=seconds
            for offset,metric in enumerate(('paired_ADD_delta_mm','ADD50_delta','ADD_AUC_delta')):
                interval=ci(samples[:,i*3+offset])
                record[metric]=row[metric]
                record[metric+'_ci_low']=interval['low'];record[metric+'_ci_high']=interval['high']
                record[metric+'_bootstrap_valid']=interval['valid_replicates']
                record[metric+'_bootstrap_empty']=interval['invalid_replicates']
                if seconds==1.:
                    for suffix in ('ci_low','ci_high','bootstrap_valid','bootstrap_empty'):
                        row[metric+'_'+suffix]=record[metric+'_'+suffix]
            sensitivity.append(record)
    return dict(summary=summaries,curves=curves,paired=comparisons,sensitivity=sensitivity,method_sensitivity=method_sensitivity,
                bootstrap=bootmeta,bootstrap_data=bootstrap_data)


def production_reference(paths, pair_ids, hashes):
    """Separate historical multi-start reference; failures retain full denominator."""
    directory=paths.get('production_evaluation');run=paths.get('production_run')
    if directory is None: return [],{'available':False,'reason':'production_evaluation path not configured'}
    path=directory/'per_pair_metrics.csv'; hashes[str(path)]=sha(path)
    rows=list(csv.DictReader(path.open())); out=[]
    for method in ('mono_candidates','dual_candidates'):
        source=[r for r in rows if r['method']==method and r['pair_id'] in pair_ids]
        keys=[(r['pair_id'],r['camera']) for r in source]
        expected={(pid,c) for pid in pair_ids for c in CAMERAS}
        if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('Incomplete production reference')
        for camera in ('pooled',*CAMERAS):
            subset=[r for r in source if camera=='pooled' or r['camera']==camera]
            errors=[]
            for row in subset:
                ok=row['pose_success']=='1'
                if method=='dual_candidates':ok &= row['stereo_refined']=='1' and row['status']=='stereo_refined'
                errors.append(float(row['add_mm']) if ok and row['add_mm'] else np.inf)
            summary,_=distribution(errors);summary.update(method=method,camera=camera,
                role='historical_production_multistart_reference',not_controlled_E1=True)
            out.append(summary)
    identity={'available':True,'evaluation':str(directory),'rows_checked':len(pair_ids)*4}
    if run is not None:
        mp=run/'manifest.json';hashes[str(mp)]=sha(mp);manifest=json.loads(mp.read_text())
        identity['production_manifest']={k:v for k,v in manifest.items() if k!='results'}
        identity['pairs_sha_matches_current']=manifest.get('pairs_sha256')==sha(paths['pairs'])
        identity['observations_manifest_same_path']=Path(manifest.get('observations_manifest','')).resolve()==paths.get('observations')
        original_root=ROOT.parent/'dual_camera_pairwise_v2'
        identity['source_code_sha_matches_current']={name:sha(original_root/name)==digest
            for name,digest in manifest.get('source_code_sha256',{}).items()}
        obs=paths.get('observations')
        if obs:
            hashes[str(obs)]=sha(obs)
            metadata=json.loads(obs.read_text())
            canonical=hashlib.sha256(json.dumps({k:v for k,v in metadata.items() if k!='identity'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            identity['observations_identity_matches_current']=manifest.get('observations_identity')==metadata.get('identity')==canonical
            identity['observation_identity_definition']='SHA256 of canonical JSON excluding the identity field itself; same as original ObservationStore.'
        match=Path(manifest['matches_manifest'])
        hashes[str(match)]=sha(match)
        identity['production_matches_unchanged']=hashes[str(match)]==manifest['matches_sha256']
        if paths.get('matches'):
            identity['matches_same_as_controlled_run']=paths['matches']==match.resolve() and sha(paths['matches'])==manifest['matches_sha256']
        identity['not_same_algorithm_as_E1']=True
        identity['difference']='Historical top-3 joint-start policy versus one shared family-specific start per controlled condition.'
    return out,identity


def diagnostic_summary(results, arrays):
    def stat(values):
        values=np.array([v for v in values if v is not None and np.isfinite(v)],float)
        return dict(n=len(values),mean=float(values.mean()) if len(values) else None,
            median=float(np.median(values)) if len(values) else None,
            p90=float(np.percentile(values,90)) if len(values) else None,
            max=float(values.max()) if len(values) else None)
    result={'methods':{},'initializations':{},'shared_timings':{},'gate_effects':[]}
    for family in ('R','E'):
        entries=[r['initializations'][family] for r in results.values()]
        result['initializations'][family]=dict(available=sum(bool(x.get('available')) for x in entries),
            on_ready=sum(bool(x.get('on_initialization',{}).get('available')) for x in entries),
            failure_reasons=dict(Counter(str(x.get('reason')) for x in entries if not x.get('available'))),
            generation_ms=stat([x.get('generation_ms') for x in entries]))
    for key in ('shared_monocular_ms','all_initialization_ms','all_methods_optimization_ms','solver_wall_ms'):
        result['shared_timings'][key]=stat([r.get('timing',{}).get(key) for r in results.values()])
    for j,method in enumerate(METHODS):
        entries=[r['methods'][method] for r in results.values()]
        d=dict(statuses=dict(Counter(str(x.get('status')) for x in entries)),
            terminations=dict(Counter(str(x.get('termination_reason','not_started')) for x in entries)),
            starts=stat([x.get('initializations_tried') for x in entries]),
            em_iterations=stat([x.get('em_iterations') for x in entries]),
            optimizer_nfev=stat([x.get('optimizer_nfev_total') for x in entries]),
            residual_function_calls=stat([x.get('optimizer_residual_calls_total') for x in entries]),
            internal_solves_at_nfev60=sum(t.get('nfev',0)>=60 for x in entries for t in x.get('trace',[])),
            internal_solves=sum(len(x.get('trace',[])) for x in entries),
            optimization_ms=stat([x.get('timing',{}).get('optimization_ms') for x in entries]),
            total_ms=stat([x.get('timing',{}).get('total_ms') for x in entries]),
            candidate_nll=stat([x.get('objective_components',{}).get('candidate_nll') for x in entries]),
            reduced_rank=stat([x.get('common_gate',{}).get('diagnostics',{}).get('reduced_rank') for x in entries]),
            reduced_condition=stat([x.get('common_gate',{}).get('diagnostics',{}).get('reduced_condition') for x in entries]),
            background_median_px=stat([x.get('background_gate',{}).get('diagnostics',{}).get('rgb_median_error_px') for x in entries]),
            background_p90_px=stat([x.get('background_gate',{}).get('diagnostics',{}).get('rgb_p90_error_px') for x in entries]))
        result['methods'][method]=d
        common=arrays['valid']['common'][:,j];final=arrays['valid']['final'][:,j];rejected=common&~final
        values=arrays['add'][:,j]
        result['gate_effects'].append(dict(method=method,common_valid=int(common.sum()),final_valid=int(final.sum()),
            rejected_by_background=int(rejected.sum()),rejected_ADD_below_50=int((rejected&(values<50)).sum()),
            rejected_ADD_mean_mm=stat(values[rejected])['mean'],
            kept_ADD_mean_mm=stat(values[final])['mean']))
    return result


def evaluate_run(run_dir, config_path, output_dir, allow_partial=False):
    run_dir=Path(run_dir).resolve();out=Path(output_dir).resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Output must remain in new extension')
    if out.exists():raise FileExistsError(out)
    config,paths=config_paths(config_path)
    hashes={str(Path(config_path).resolve()):sha(config_path),str(Path(__file__)):sha(__file__)}
    for filename in ('plotting.py','reporting.py'):
        p=Path(__file__).parent/filename;hashes[str(p)]=sha(p)
    result_path=run_dir/'pair_results.jsonl'
    for p in [result_path,paths['pairs'],paths['reference']]:hashes[str(p)]=sha(p)
    results=jsonl(result_path);pairs=jsonl(paths['pairs']);refs=jsonl(paths['reference'])
    if not set(results)<=set(pairs) or not set(results)<=set(refs):raise ValueError('Missing input/reference pairs')
    expected=config.get('evaluation',{}).get('expected_pairs',2596)
    full=len(results)==len(pairs)==expected
    if config.get('evaluation',{}).get('require_full',True) and not full and not allow_partial:raise ValueError('Full run missing pairs')
    arrays=load_arrays(results,pairs,refs)
    evaluated=evaluate_arrays(arrays,config.get('bootstrap',config.get('evaluation',{}).get('bootstrap',{})))
    production,production_identity=production_reference(paths,set(results),hashes)
    diagnostics=diagnostic_summary(results,arrays)
    out.mkdir(parents=True,exist_ok=False)
    write_csv(out/'summary_metrics.csv',evaluated['summary'])
    write_csv(out/'paired_comparisons.csv',evaluated['paired'])
    write_csv(out/'bootstrap_sensitivity.csv',evaluated['sensitivity'])
    write_csv(out/'method_bootstrap_sensitivity.csv',evaluated['method_sensitivity'])
    write_csv(out/'per_observation_metrics.csv',arrays['rows'])
    write_csv(out/'failures.csv',arrays['failures'])
    write_csv(out/'timings.csv',arrays['timings'])
    write_csv(out/'production_reference.csv',production)
    write_json(out/'curves.json',evaluated['curves'])
    write_json(out/'bootstrap.json',evaluated['bootstrap'])
    write_json(out/'production_identity.json',production_identity)
    write_json(out/'diagnostics.json',diagnostics)
    np.savez_compressed(out/'evaluation_arrays.npz',add_before_gate=arrays['add'],
        common_valid=arrays['valid']['common'],final_valid=arrays['valid']['final'],
        pre_eligible=arrays['eligible'],timestamps=arrays['timestamps'],pair_ids=np.array(arrays['ids']),
        **{track+'_errors':v for track,v in arrays['pcks'].items()})
    np.savez_compressed(out/'bootstrap_indices.npz',**{f'block_{seconds}_{k}':v for seconds,record in evaluated['bootstrap_data'].items() for k,v in record.items()})
    failure_counts=Counter((r['method'],r['gate'],r['reason']) for r in arrays['failures'])
    exclusions=Counter(reason for pid in results if not results[pid]['pre_eligible'] for reason in results[pid].get('pre_eligible_reasons',[]))
    meta=dict(schema_version=1,role='evaluation_only',dataset='UR_dual_2596',full_sequence=full,
        n_pairs=len(results),n_camera_observations=len(results)*2,n_pre_eligible_pairs=int(arrays['eligible'].sum()),
        pre_eligible_exclusion_reasons=dict(exclusions),run_dir=str(run_dir),config_path=str(Path(config_path).resolve()),
        paths=paths,gate_policies=['final','common'],populations=['full','pre_eligible'],
        ADD_thresholds_mm=THRESHOLDS,strict_threshold_operator='<',failure_error='infinity',
        conditional_statistics=['mean','median','p90'],reference_is_independent_gt=False,
        bootstrap=evaluated['bootstrap'],summary_rows=len(evaluated['summary']),paired_rows=len(evaluated['paired']),
        failure_counts=[dict(method=m,gate=g,reason=r,n_pairs=n) for (m,g,r),n in sorted(failure_counts.items())],
        input_sha256=hashes,input_hashes_unchanged=all(sha(p)==h for p,h in hashes.items()))
    if not meta['input_hashes_unchanged']:raise ValueError('Evaluation input changed')
    write_json(out/'evaluation.json',meta)
    from plotting import build_plots
    build_plots(out,arrays,evaluated,results,pairs,refs)
    from reporting import build_report
    build_report(out,meta,evaluated,production)
    return dict(output=str(out),pairs=len(results),pre_eligible_pairs=int(arrays['eligible'].sum()),
                full_sequence=full,summary_rows=len(evaluated['summary']),passed=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--allow-partial',action='store_true',help='Explicit smoke-only evaluation; report is labelled partial.')
    args=p.parse_args();print(json.dumps(evaluate_run(args.run_dir,args.config,args.output,args.allow_partial),ensure_ascii=False))
