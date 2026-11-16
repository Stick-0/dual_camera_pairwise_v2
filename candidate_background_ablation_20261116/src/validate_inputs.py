#!/usr/bin/env python3
"""Audit all immutable UR_dual_2596 observations and stored RoMa matches.

No model construction, neural forward, voting, new candidate generation, or
matching is run. Deterministic filters are recomputed on saved observations.
"""
import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
NEW = Path(__file__).resolve().parents[1]
REPO = NEW.parent
EXT = REPO / 'dual_camera_pairwise_v2'
EXP = EXT / 'experiments/ur_dual_2596_20261114'
DATA = REPO / 'Data/real/Dual_UR/UR_dual_2596'
OUT = NEW / 'validation'
for name, suffix in {'XDG_CACHE_HOME':'xdg', 'MPLCONFIGDIR':'mpl', 'TORCH_HOME':'torch',
                     'TORCH_EXTENSIONS_DIR':'torch_extensions', 'HF_HOME':'hf',
                     'HUGGINGFACE_HUB_CACHE':'hf/hub', 'NUMBA_CACHE_DIR':'numba',
                     'CUDA_CACHE_PATH':'cuda', 'PIP_CACHE_DIR':'pip', 'TMPDIR':'tmp'}.items():
    path = NEW / 'cache/input_validation' / suffix
    path.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(path)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


def guard(event, args):
    path = None
    if event == 'open' and not isinstance(args[0], int):
        mode, flags = args[1:3]
        if (mode and any(c in mode for c in 'wax+')) or (flags and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)):
            path = args[0]
    elif event in {'os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod', 'os.utime'}:
        path = args[0]
    elif event in {'os.rename', 'os.link', 'os.symlink'}:
        for p in args[:2]:
            if not Path(os.fsdecode(p)).resolve().is_relative_to(NEW):
                raise PermissionError(f'Validation cannot mutate original: {p}')
    if path is not None and not isinstance(path, int):
        if not Path(os.fsdecode(path)).resolve().is_relative_to(NEW):
            raise PermissionError(f'Validation cannot mutate original: {path}')


sys.addaudithook(guard)
import numpy as np
sys.path.insert(0, str(EXT))
from pairwise.bridge import ObservationStore, _validate_current, _fk
from pairwise.candidates import prepare_candidates
from pairwise.matching import load_pair_matches, filter_matches, _array_hash, SOURCES
from pairwise.solver import _settings, _match_arrays, DEFAULTS as SOLVER_DEFAULTS
from pairwise.matching import DEFAULTS as MATCH_DEFAULTS
from pairwise.runtime import identity

HASHES = {}
CAMERAS = ('camera1', 'camera2')
ROBOT_SHA = '7b5e518c15a54a7e5146d8b658a2d71f3e185866b8be69f9e8409df2ccc10d8c'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source(path, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    if expected is not None:
        assert digest == expected, f'File SHA mismatch: {path}'
    HASHES[str(path)] = digest
    return path


def read(path, expected=None):
    return json.loads(source(path, expected).read_text())


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def check_weights(contract):
    checkpoint = source(contract['checkpoint'], ROBOT_SHA)
    roma = source(EXT / 'weights/romav2.0.1.pt', SOURCES['roma_v2']['checkpoint_sha256_expected'])
    # The official default descriptor is pretrained=False at construction;
    # f.* DINOv3 parameters are subsequently loaded from this complete RoMa file.
    import torch
    state = torch.load(roma, map_location='cpu', weights_only=True)
    keys = sorted(k for k in state if k.startswith('f.'))
    assert keys and all(isinstance(state[k], torch.Tensor) for k in keys)
    tensor_hashes = {}
    total = 0
    for key in keys:
        tensor = state[key].detach().cpu().contiguous()
        data = tensor.view(torch.uint8).numpy().tobytes()
        digest = hashlib.sha256(str(tensor.dtype).encode() + str(tuple(tensor.shape)).encode() + data).hexdigest()
        tensor_hashes[key] = {'dtype': str(tensor.dtype), 'shape': list(tensor.shape), 'sha256': digest}
        total += tensor.numel()
    del state
    subset_sha = identity(tensor_hashes)
    dino_source = EXT / 'vendor/dinov3-adc254450203739c8149213a7a69d8d905b4fcfa'
    source(dino_source / 'dinov3/hub/backbones.py')
    source(dino_source / 'LICENSE.md')
    source(EXT / 'vendor/RoMaV2/src/romav2/features.py')
    source(EXT / 'vendor/RoMaV2/src/romav2/romav2.py')
    archive = source(EXT / 'vendor/dinov3_adc254450203739c8149213a7a69d8d905b4fcfa.zip')
    commit = subprocess.check_output(['git', '-C', str(EXT / 'vendor/RoMaV2'), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == SOURCES['roma_v2']['commit']
    assert not subprocess.check_output(['git', '-C', str(EXT / 'vendor/RoMaV2'), 'diff', '--name-only'], text=True).strip()
    result = {'robot_checkpoint': str(checkpoint), 'robot_checkpoint_sha256': sha(checkpoint),
              'roma_checkpoint': str(roma), 'roma_checkpoint_sha256': sha(roma),
              'roma_source_commit': commit, 'roma_source_worktree_clean': True,
              'dinov3_source_commit': 'adc254450203739c8149213a7a69d8d905b4fcfa',
              'dinov3_source_archive_sha256': sha(archive),
              'dinov3_weight_storage': 'f.* tensor subset embedded in complete romav2.0.1.pt; no separate DINO checkpoint is loaded by the default configuration',
              'dinov3_tensor_count': len(keys), 'dinov3_parameter_elements': total,
              'dinov3_tensor_manifest_identity': subset_sha, 'dinov3_tensor_manifest': tensor_hashes,
              'tensor_hash_rule': 'SHA256(torch dtype string + tuple shape string + contiguous raw bytes); aggregate canonical JSON identity',
              'weights_loaded_only_on_CPU_for_hash_audit': True, 'models_instantiated': False,
              'torch_version': torch.__version__, 'torch_cuda_available': bool(torch.cuda.is_available())}
    write('weights.json', result)
    return {k:v for k,v in result.items() if k != 'dinov3_tensor_manifest'}


def main():
    started = time.perf_counter()
    contract = read(EXP / 'ingest/full/input_contract.json')
    contract_identity = identity(contract)
    producer = read(EXP / 'ingest/full/producer.json')
    production = read(EXP / 'runs/roma_v2_full2596/manifest.json')
    config = read(EXP / 'runs/roma_v2_full2596/resolved_config.json')
    cfg = _settings(config)
    report_path = source(EXT / 'releases/current_20261114/document/CURRENT_IMPLEMENTATION_REPORT.html')
    report_text = report_path.read_text()
    for rel, expected in contract['source_sha256'].items():
        source(REPO / rel, expected)
    for rel, expected in production['source_code_sha256'].items():
        source(EXT / rel, expected)
    for rel, expected in contract['recorder_sha256'].items():
        source(DATA / rel, expected)
    assert cfg['max_nfev'] == 60 and 'max_nfev=60' in report_text
    assert cfg['max_rgb_matches'] == 80 and cfg['huber_delta'] == 3 and cfg['lambda_rgb'] == 1
    assert cfg['rgb_noise_std_px'] == 1.5
    assert cfg['candidate_options']['kernel_std_px'] == 2 and cfg['candidate_options']['outlier_probability'] == .05
    assert cfg['candidate_options']['max_em_iterations'] == 10
    assert producer['voting']['num_hypotheses'] == 8 and producer['strict_checkpoint_loading']
    assert producer['amp'] is False and producer['device'] == 'cuda' and producer['dtype'] == 'float32'
    config_audit = {'passed': True, 'report_path': str(report_path), 'report_sha256': sha(report_path),
                    'production_resolved_config': config,
                    'default_vs_executed': [
                        {'field': 'solver.max_nfev', 'source_default': SOLVER_DEFAULTS['max_nfev'], 'executed_and_reported': 60,
                         'resolution': 'Freeze 60 per internal solve in all four conditions'},
                        {'field': 'matching.mask_dilation_px', 'source_default': MATCH_DEFAULTS['mask_dilation_px'], 'executed': 50,
                         'resolution': 'Reuse and verify the fixed stored 50px current mask filtering'},
                        {'field': 'solver.max_joint_starts', 'production': 3, 'controlled_ablation': 1,
                         'resolution': 'Intentional experimental factor control, fixed before outcomes; R/E starts selected once then shared OFF/ON'}],
                    'unchanged_candidate_likelihood': {'sigma_px': 2, 'epsilon': .05, 'fixed_support_prior': True, 'max_em_iterations': 10, 'original_hypotheses': 8},
                    'production_code_hashes_match_current_sources': True,
                    'runtime_configure_runtime_called': False}
    write('config_and_report_audit.json', config_audit)
    weights = check_weights(contract)
    write('critical_sources_ready.json', {'passed': True, 'weights': weights, 'source_sha256': HASHES,
                                         'all_frames_and_matches_complete': False})
    print(json.dumps({'stage': 'critical_sources_and_weights_passed', 'elapsed_s': time.perf_counter()-started}), flush=True)

    obs_path = source(EXP / 'ingest/full/observations.json')
    store = ObservationStore(obs_path)
    source(store.metadata['candidate_manifest_path'], store.metadata['candidate_manifest_sha256'])
    pairs_path = source(store.metadata['pairs_path'], store.metadata['pairs_sha256'])
    pairs = [json.loads(line) for line in pairs_path.open() if line.strip()]
    manifest = read(EXP / 'matches/roma_v2/manifest.json', production['matches_sha256'])
    expected_ids = [f'{k:06d}' for k in range(2596)]
    assert [p['pair_id'] for p in pairs] == expected_ids
    assert set(manifest['pairs']) == set(expected_ids)
    assert len(store.metadata['frames']) == len(store.candidates['frames']) == 5192
    assert store.metadata['robot_checkpoint_sha256'] == contract['checkpoint_sha256'] == ROBOT_SHA
    assert store.metadata['source_cache_identity'] == contract_identity == producer['input_contract_identity']
    assert production['observations_identity'] == store.metadata['identity']
    contract_frames = {r['camera_id']+'/'+r['frame_id']:r for r in contract['frames']}
    recorder = {r['sample_id']:r for r in (json.loads(line) for line in (DATA/'frames.jsonl').open())}
    for cam in CAMERAS:
        settings = read(DATA / cam / '_camera_settings.json')
        assert np.array_equal(settings['K'], store.metadata['cameras'][cam]['K'])
        assert np.array_equal(settings['distortion'], store.metadata['cameras'][cam]['D'])
        assert not settings['rgb_rectified'] and settings['intrinsics_source'] == 'calibration_json'

    counts = collections.Counter()
    candidate_hist, selected_hist, mask_valid_hist = collections.Counter(), collections.Counter(), collections.Counter()
    max_fk_error, max_interp_error = 0., 0.
    timestamps, camera_skews = [], []
    with (OUT / 'frame_contracts.jsonl').open('w') as frame_out, (OUT / 'pair_contracts.jsonl').open('w') as pair_out:
        for pi, pair in enumerate(pairs):
            pid = pair['pair_id']
            observations = store.pair(pair)
            q = []
            for ci, cam in enumerate(CAMERAS):
                key = cam + '/' + pid
                camera, raw = pair['cameras'][cam], observations[cam]
                row = store.metadata['frames'][key]
                baseline = contract_frames[key]
                committed = read(EXP / f'ingest/full/frames/{cam}_{pid}.json')
                source(row['arrays_path'], row['arrays_sha256'])
                source(row['image_path'], row['image_sha256'])
                assert Path(row['image_path']) == DATA / recorder[pid]['cameras'][cam]['rgb']
                assert identity(raw['hypotheses']) == identity(committed['hypotheses'])
                assert row['arrays_sha256'] == committed['arrays_sha256']
                assert row['image_sha256'] == baseline['image_sha256']
                assert np.array_equal(raw['q_rad'], baseline['q_rad'])
                assert np.array_equal(raw['keypoints_base_m'], camera['keypoints_base_m'])
                assert np.array_equal(raw['keypoints_base_m'], baseline['keypoints_base_m'])
                assert np.array_equal(camera['K'], store.metadata['cameras'][cam]['K'])
                assert np.array_equal(camera['D'], store.metadata['cameras'][cam]['D'])
                _validate_current(row, raw)
                fk_error = float(np.max(np.abs(_fk(raw['q_rad'])-raw['keypoints_base_m'])))
                max_fk_error = max(max_fk_error, fk_error)
                inter = row['recorded_joint_interpolation']
                qi = np.asarray(inter['before_q_rad']) + inter['interpolation_fraction']*(np.asarray(inter['after_q_rad'])-inter['before_q_rad'])
                max_interp_error = max(max_interp_error, float(np.max(np.abs(qi-raw['q_rad']))))
                annotation = read(DATA / recorder[pid]['cameras'][cam]['gt'])
                # Only measured joints and capture time are inspected; pose and
                # object/keypoint labels are never read into estimation inputs.
                assert np.array_equal(annotation['joint_angles_rad'], raw['q_rad'])
                assert annotation['capture_metadata']['camera']['capture_monotonic_s'] == row['timestamp_s']
                assert recorder[pid]['camera_capture_monotonic_s'][cam] == row['timestamp_s']
                assert row['seed'] == ci*2596 + pi and row['q_association'] == 'per_frame_exact'
                assert row['observation_stage'] == 'raw_voting' and row['source_device'] == 'cuda'
                assert row['source_dtype'] == 'float32'
                assert np.array_equal(row['preprocessing_transform']['M_model_to_native'], np.eye(2))
                assert np.array_equal(row['preprocessing_transform']['b_model_to_native'], np.zeros(2))
                assert raw['foreground_exclusion_mask'].shape == (480, 640)
                prepared = prepare_candidates(raw, camera, cfg['candidate_options'])
                for k, (h, mode) in enumerate(zip(raw['hypotheses'], prepared['modes'])):
                    uv, support = np.asarray(h['uv']), np.asarray(h['original_scores'])
                    assert uv.shape == (8,2) and support.shape == (8,)
                    assert np.isfinite(uv).all() and np.isfinite(support).all() and (support >= 0).all()
                    candidate_hist[len(uv)] += 1
                    assert np.array_equal(mode['mu'], uv[mode['original_indices']])
                    if mode['valid']:
                        s = support[mode['original_indices']].astype(float)
                        assert np.array_equal(mode['alpha'], s/s.sum())
                    counts['raw_hypotheses'] += len(uv)
                    counts['retained_hypotheses'] += len(mode['mu'])
                counts['images'] += 1
                counts['raw_valid_keypoints'] += int(raw['valid_keypoints'].sum())
                q.append(raw['q_rad'])
                record = {'pair_id': pid, 'camera': cam, 'arrays_sha256': row['arrays_sha256'],
                          'image_sha256': row['image_sha256'], 'candidate_coordinates_support_ids_sha256': identity(raw['hypotheses']),
                          'valid_mask_sha256': _array_hash(raw['valid_keypoints']),
                          'q_sha256': _array_hash(raw['q_rad']), 'fk_sha256': _array_hash(raw['keypoints_base_m']),
                          'camera_intrinsics_identity': identity({k:camera[k] for k in ('K','D','width','height')}),
                          'timestamp_s': row['timestamp_s'], 'raw_candidate_ids': list(range(8)),
                          'retained_candidate_ids': [m['original_indices'].tolist() for m in prepared['modes']],
                          'prepared_candidates_identity': identity(prepared)}
                frame_out.write(json.dumps(record, separators=(',', ':'))+'\n')
            counts['distinct_q1_q2_pairs'] += int(not np.array_equal(*q))
            matches = load_pair_matches(manifest, pair, observations)
            mr = manifest['pairs'][pid]
            source(mr['arrays_path'], mr['arrays_sha256'])
            assert mr['current_frame_only'] and mr['model_training'] is False
            assert mr['inference_config']['mask_dilation_px'] == 50
            assert mr['inference_config']['mask_shape_policy'] == 'original'
            valid, reasons, _, _ = filter_matches(matches['points_c1_native'], matches['points_c2_native'],
                                                   matches['match_score'], matches['bidirectional_error_px'],
                                                   [observations[c]['foreground_exclusion_mask'] for c in CAMERAS],
                                                   [(480,640),(480,640)], mr['inference_config'])
            assert np.array_equal(valid, matches['valid_mask'])
            assert np.array_equal(reasons, matches['rejection_reason'])
            p1, p2, info = _match_arrays(matches, cfg)
            indices = info['optimizer_selected_indices']
            assert len(indices) == len(set(indices)) == 80
            assert np.array_equal(p1, matches['points_c1_native'][indices])
            assert np.array_equal(p2, matches['points_c2_native'][indices])
            production_result = read(EXP / f'runs/roma_v2_full2596/results/{pid}.json')
            for method in ('dual_candidates', 'dual_mean'):
                old = production_result['methods'][method]['quality']['match_input']['optimizer_selected_indices']
                assert indices == old, f'Production selected background differs: {pid}/{method}'
            selected_hist[len(indices)] += 1
            mask_valid_hist[int(valid.sum())] += 1
            counts['matches_pairs'] += 1
            t1, t2 = [pair['cameras'][c]['timestamp_s'] for c in CAMERAS]
            midpoint = (t1+t2)/2
            timestamps.append(midpoint);camera_skews.append(t2-t1)
            record = {'pair_id': pid, 'timestamp_camera1_s': t1, 'timestamp_camera2_s': t2,
                      'pair_timestamp_s': midpoint, 'pair_timestamp_definition': 'mean of the two recorded capture_monotonic_s',
                      'matches_arrays_sha256': mr['arrays_sha256'], 'optimizer_selected_indices': indices,
                      'optimizer_points_identity': identity({'indices':indices,'camera1':p1,'camera2':p2}),
                      'input_candidate_identity': identity({c:observations[c]['hypotheses'] for c in CAMERAS}),
                      'filter_recomputed_exact': True, 'production_subset80_exact': True}
            pair_out.write(json.dumps(record, separators=(',', ':'))+'\n')
            if (pi+1) % 250 == 0:
                frame_out.flush();pair_out.flush()
                print(json.dumps({'stage':'full_input_audit','pairs_complete':pi+1,'elapsed_s':time.perf_counter()-started}),flush=True)

    assert counts['images'] == 5192 and counts['matches_pairs'] == 2596
    dt = np.diff(timestamps)
    assert (dt > 0).all()
    gpu = subprocess.run(['nvidia-smi'], text=True, capture_output=True)
    result = {'passed': True, 'dataset': 'UR_dual_2596', 'pairs': 2596, 'counts': dict(counts),
              'candidate_count_histogram': dict(candidate_hist), 'selected_match_count_histogram': dict(selected_hist),
              'filtered_match_count_min_max': [min(mask_valid_hist),max(mask_valid_hist)],
              'all_5192_RGB_and_NPZ_SHA256_verified': True, 'all_5192_original_candidates_exact': True,
              'all_camera_specific_q_and_FK_verified': True, 'max_fk_absolute_error_m': max_fk_error,
              'max_joint_interpolation_absolute_error_rad': max_interp_error,
              'all_match_identity_mask_image_contracts_verified': True,
              'all_deterministic_score_cycle_mask_grid_filters_recomputed_exact': True,
              'all_fixed_subset80_coordinates_and_indices_match_production': True,
              'evaluation_reference_poses_used': False,
              'annotation_access_scope': 'Validator only: original recorded joint_angles_rad and capture_metadata.camera.capture_monotonic_s; camera_data transforms and objects labels never feed estimation',
              'weights': weights, 'nvidia_smi_exit_code': gpu.returncode,
              'nvidia_smi_stdout': gpu.stdout, 'nvidia_smi_stderr': gpu.stderr,
              'execution_mode': 'CPU numerical solver may reuse the fully verified immutable prior CUDA FP32 robot observations and CUDA RoMa matching cache; no new GPU inference/matching was executed',
              'new_model_inference': False, 'new_matching': False, 'new_candidates': False,
              'all_writes_confined_to_new_extension': True, 'original_runtime_configure_called': False,
              'raw_timestamp_available': True, 'pair_timestamp_source': 'Recorded camera capture_monotonic_s, averaged per pair',
              'timestamp_span_s': float(timestamps[-1]-timestamps[0]),
              'interpair_dt_min_median_max_s': [float(dt.min()),float(np.median(dt)),float(dt.max())],
              'camera2_minus_camera1_min_median_max_s': [float(np.min(camera_skews)),float(np.median(camera_skews)),float(np.max(camera_skews))],
              'source_sha256': HASHES, 'source_hash_count': len(HASHES),
              'audit_wall_s': time.perf_counter()-started}
    write('input_audit.json', result)
    write('critical_sources_ready.json', {'passed': True, 'weights': weights, 'source_sha256': HASHES,
                                         'all_frames_and_matches_complete': True})
    print(json.dumps({k:result[k] for k in ('passed','pairs','counts','max_fk_absolute_error_m','audit_wall_s')}),flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write('input_audit_failure.json', {'passed':False,'exception_type':type(exc).__name__,'error':str(exc)})
        raise
