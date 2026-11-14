#!/usr/bin/env python3
"""Immutable, resumable UR recorder import and original CUDA FP32 raw voting.

No v1 runtime is imported. Pose labels are never estimator inputs: this importer
allowlists only recorder encoder/timestamp fields from recorder JSON containers.
The separate --prepare-reference action alone opens object pose/keypoint labels.
All artifacts, including library caches, remain in this new ingest directory.
"""
import argparse
from dataclasses import asdict
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(EXT))
from pairwise.runtime import _audit_mutation, identity, jsonable, output_path, sha256, write_json, REPO_ROOT, require_original_sources
REPO = REPO_ROOT

JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
SOURCE_FILES = ['robotpose/network3.py', 'robotpose/vote2.py', 'robotpose/my_pnp.py',
                'robotpose/infer_live.py', 'ur5e_capture/geometry.py']
CHECKPOINT_SHA256 = '7b5e518c15a54a7e5146d8b658a2d71f3e185866b8be69f9e8409df2ccc10d8c'
ARRAY_FIELDS = ('keypoints_uv', 'covariance_uv', 'valid_keypoints', 'q_rad',
                'keypoints_base_m', 'foreground_exclusion_mask',
                'keypoints_model_uv', 'covariance_model_uv')


def configure_runtime(output):
    if not output.is_relative_to(HERE):
        raise ValueError('All importer output must be inside this new ingest directory')
    output.mkdir(parents=True, exist_ok=True)
    for name, suffix in {'XDG_CACHE_HOME': 'cache/runtime', 'MPLCONFIGDIR': 'cache/mpl',
            'TORCH_HOME': 'cache/torch', 'TORCH_EXTENSIONS_DIR': 'cache/torch_extensions',
            'HF_HOME': 'cache/hf', 'HUGGINGFACE_HUB_CACHE': 'cache/hf/hub',
            'NUMBA_CACHE_DIR': 'cache/numba', 'CUDA_CACHE_PATH': 'cache/cuda',
            'PIP_CACHE_DIR': 'cache/pip', 'TMPDIR': 'tmp'}.items():
        directory = output / suffix
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(directory)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.addaudithook(_audit_mutation)


def read_json(path):
    return json.loads(Path(path).read_text())


def immutable_json(path, value):
    """Resume only after exact semantic comparison; never rewrite an artifact."""
    if path.exists():
        if identity(read_json(path)) != identity(value):
            raise ValueError(f'Existing artifact differs from current contract: {path}')
    else:
        write_json(path, value)


def load_source(name, path):
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path.resolve():
            raise ValueError(f'Original module collision: {name}')
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def checked_dataset_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Invalid dataset input: {relative}')
    return path


def camera_settings(root, cid):
    import numpy as np
    path = root / cid / '_camera_settings.json'
    data = read_json(path)
    if data['camera_id'] != cid or data['intrinsics_source'] != 'calibration_json':
        raise ValueError('Expected recorder top-level calibration_json')
    if data['rgb_rectified'] is not False:
        raise ValueError('Expected native distorted images, matching the inspected recorder')
    K, D = np.asarray(data['K'], float), np.asarray(data['distortion'], float)
    if K.shape != (3, 3) or D.size not in (4, 5, 8, 12, 14):
        raise ValueError('Malformed OpenCV calibration')
    if not np.isfinite(K).all() or not np.isfinite(D).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError('Invalid finite calibration/focal length')
    return {'K': K.tolist(), 'D': D.tolist(), 'width': int(data['width']),
            'height': int(data['height']), 'distortion_model': 'opencv_brown',
            'pixel_space': 'native_distorted', 'calibration_source': str(path),
            'intrinsics_source': 'recorder_top_level_calibration_json_OpenCV'}


def recorder_frame(root, cid, record, session, seed):
    import numpy as np
    annotation = checked_dataset_path(root, record['cameras'][cid]['gt'])
    # This JSON container also contains labels. Only these recorder fields are
    # selected; objects/camera pose/annotated 3D and 2D keypoints are not accessed.
    decoded = read_json(annotation)
    capture = decoded['capture_metadata']
    if str(capture['sample_id']) != str(record['sample_id']) or decoded['camera_data']['camera_id'] != cid:
        raise ValueError('Recorder sample or camera identity mismatch')
    robot, camera = capture['robot'], capture['camera']
    q = np.asarray(decoded['joint_angles_rad'], float)
    del decoded, capture
    t = float(camera['capture_monotonic_s'])
    before, after = float(robot['before_monotonic_s']), float(robot['after_monotonic_s'])
    if not np.isfinite([t, before, after, robot['target_monotonic_s']]).all():
        raise ValueError('Nonfinite recorder association time')
    if abs(t-float(record['camera_capture_monotonic_s'][cid])) > 1e-6 or abs(t-robot['target_monotonic_s']) > 1e-8:
        raise ValueError('Camera/encoder target timestamp mismatch')
    if robot['method'] == 'exact' and before == after and abs(t-before) <= 1e-9:
        fraction = 0.
    elif robot['method'] == 'linear_interpolation' and before <= t <= after and after > before:
        fraction = (t-before)/(after-before)
    else:
        raise ValueError('Invalid recorder interpolation bracket')
    max_gap = float(session['joint_max_gap_ms']) / 1000.
    if not math.isfinite(max_gap) or max_gap <= 0 or max(t-before, after-t, after-before) > max_gap + 1e-9:
        raise ValueError('Recorder association exceeds joint_max_gap_ms')
    if abs(fraction-float(robot['interpolation_fraction'])) > 1e-7:
        raise ValueError('Interpolation fraction mismatch')
    qa, qb = np.asarray(robot['before_q_rad'], float), np.asarray(robot['after_q_rad'], float)
    if any(v.shape != (6,) or not np.isfinite(v).all() for v in (q, qa, qb)):
        raise ValueError('Joint encoder vector must contain six finite radians')
    if not np.allclose(q, qa + fraction*(qb-qa), rtol=0, atol=1e-9):
        raise ValueError('Encoder q differs from original interpolation')
    interpolation = {k: robot[k] for k in ('method', 'target_monotonic_s',
        'before_monotonic_s', 'after_monotonic_s', 'before_q_rad', 'after_q_rad', 'interpolation_fraction')}
    image = checked_dataset_path(root, record['cameras'][cid]['rgb'])
    fk = load_source('_pairwise_v2_original_fk', REPO/'ur5e_capture/geometry.py')
    return {'frame_id': str(record['sample_id']), 'camera_id': cid,
        'image_path': str(image), 'image_sha256': sha256(image), 'timestamp_s': t,
        'timestamp_source': f'{annotation}:capture_metadata.camera.capture_monotonic_s; {camera["timestamp_basis"]}',
        'q_rad': q.tolist(), 'keypoints_base_m': fk.ur5e_keypoints_from_joints(q).tolist(),
        'q_source': f'{annotation}:joint_angles_rad; recorder interpolation at current camera capture time',
        'q_association': 'per_frame_exact', 'recorded_joint_interpolation': interpolation,
        'seed': seed, 'split': 'all', 'observation_stage': 'raw_voting'}


def prepare_dataset(args):
    import numpy as np
    import yaml
    root = args.dataset.resolve()
    config = yaml.safe_load(args.config.read_text())
    checkpoint = args.checkpoint.resolve()
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise ValueError('Checkpoint differs from requested, frozen ur5e_new2/last.pt')
    session_full = read_json(root/'session.json')
    session = {k: session_full[k] for k in ('robot', 'joint_names', 'joint_max_gap_ms')}
    del session_full
    if session['robot'].lower() != 'ur5e' or session['joint_names'] != JOINT_NAMES:
        raise ValueError('Unsupported robot or joint ordering')
    records = [json.loads(line) for line in (root/'frames.jsonl').read_text().splitlines() if line.strip()]
    records.sort(key=lambda row: int(row['index']))
    if len(records) != 2596 or len({r['sample_id'] for r in records}) != len(records) or len({r['index'] for r in records}) != len(records):
        raise ValueError('Expected exactly 2596 unique recorder pairs')
    total_count = len(records)
    records = records[:args.limit] if args.limit is not None else records
    cameras = {cid: camera_settings(root, cid) for cid in ('camera1', 'camera2')}
    frames = []
    for ci, cid in enumerate(cameras):
        rows = [recorder_frame(root, cid, record, session, ci*total_count+idx) for idx, record in enumerate(records)]
        if not np.all(np.diff([row['timestamp_s'] for row in rows]) > 0):
            raise ValueError('Per-camera recorder timestamps must strictly increase')
        frames.extend(rows)
    contract = {'schema_version': 2, 'dataset': str(root), 'checkpoint': str(checkpoint),
        'checkpoint_sha256': checkpoint_hash, 'n_pairs': len(records), 'n_images': len(frames),
        'source_pair_count': total_count, 'split': 'all', 'no_dev_final_split': True,
        'source_sha256': {f: sha256(REPO/f) for f in SOURCE_FILES},
        'recorder_sha256': {str(p.relative_to(root)): sha256(p) for p in
            [root/'session.json', root/'frames.jsonl', root/'camera1/_camera_settings.json', root/'camera2/_camera_settings.json']},
        'cameras': cameras, 'frames': frames,
        'network': 'robotpose.network3.RobotJointDetection', 'strict_checkpoint_loading': True,
        'preprocessing': {'input_size': [640, 480], 'normalize': True, 'coordinate_convention': 'legacy_scale_only'},
        'voting': {'num_hypotheses': 8, 'other_options': 'unchanged original PVNetVoteConfig defaults'},
        'seed_policy': 'original camera-major ordinal convention; camera1=index, camera2=2596+index',
        'device': 'cuda', 'dtype': 'float32', 'amp': False, 'temporal_EMPnp': False,
        'no_reference_fields': True, 'annotation_container_role': 'only allowlisted timestamp and encoder metadata read',
        'candidate_policy': config['candidates']}
    immutable_json(args.output/'input_contract.json', contract)
    pairs = []
    lookup = {(row['camera_id'], row['frame_id']): row for row in frames}
    for record in records:
        fid = str(record['sample_id'])
        pair = {'schema_version': 2, 'pair_id': fid, 'split': 'all', 'cameras': {},
            'pairing_provenance': str(root/'frames.jsonl')+':sample_id='+fid,
            'robot_checkpoint_sha256': checkpoint_hash,
            'preprocessing_identity': identity(contract['preprocessing']),
            'voting_identity': identity(contract['voting'])}
        for cid, calibration in cameras.items():
            row = lookup[(cid, fid)]
            pair['cameras'][cid] = dict(calibration, **{key: row[key] for key in
                ('frame_id', 'image_path', 'image_sha256', 'timestamp_s', 'q_rad', 'keypoints_base_m')},
                q_provenance=row['q_source'], timestamp_provenance=row['timestamp_source'],
                observation_key=cid+'/'+fid, current_mask_shape=[calibration['height'], calibration['width']])
        pair['timestamp_delta_s'] = pair['cameras']['camera2']['timestamp_s']-pair['cameras']['camera1']['timestamp_s']
        if abs(1000*pair['timestamp_delta_s']-float(record['pair_skew_ms'])) > 1e-5:
            raise ValueError('Recorder pair skew does not match capture timestamps')
        pairs.append(pair)
    path = args.output/'pairs.jsonl'
    if path.exists():
        existing = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if identity(existing) != identity(pairs):
            raise ValueError('Existing pairs manifest differs')
    else:
        with path.open('x') as stream:
            for pair in pairs:
                stream.write(json.dumps(jsonable(pair), allow_nan=False)+'\n')
    return config, contract, pairs


class FrozenRawProducer:
    def __init__(self, checkpoint, threads=4):
        import torch
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required; CPU fallback is prohibited')
        torch.set_num_threads(threads)
        self.device = torch.device('cuda')
        for name in ('network3', 'vote2', 'my_pnp'):
            load_source(name, REPO/'robotpose'/f'{name}.py')
        original = load_source('_2596_original_live', REPO/'robotpose/infer_live.py')
        self.voting = sys.modules['vote2']
        self.vote_cfg = self.voting.PVNetVoteConfig(num_hypotheses=8)
        self.model = sys.modules['network3'].RobotJointDetection(pretrained=False)
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        self.model.load_state_dict(ckpt.get('model', ckpt), strict=True)
        del ckpt
        self.model.to(self.device).eval().to(memory_format=torch.channels_last)
        self.transform = original.make_transform(480, 640, normalize=True)

    def infer(self, row, camera):
        import cv2
        import numpy as np
        torch = self.torch
        image = cv2.imread(row['image_path'])
        if image is None or image.shape[:2] != (camera['height'], camera['width']):
            raise ValueError('RGB dimensions do not match native calibration')
        torch.manual_seed(row['seed'])
        torch.cuda.manual_seed_all(row['seed'])
        cv2.setRNGSeed(row['seed'])
        with torch.inference_mode():
            start = time.perf_counter()
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            x = self.transform(rgb).unsqueeze(0).to(self.device).contiguous(memory_format=torch.channels_last)
            seg, vec = self.model(x)
            if seg.dtype != torch.float32 or vec.dtype != torch.float32:
                raise ValueError('Expected CUDA FP32 model outputs without autocast')
            torch.cuda.synchronize()
            model_s = time.perf_counter()-start
            start = time.perf_counter()
            mu, cov, extra = self.voting.pvnet_vote(seg, vec, self.vote_cfg, return_debug=True)
            torch.cuda.synchronize()
            vote_s = time.perf_counter()-start
            uv_model = mu[0].cpu().numpy().astype(float)
            cov_model = cov[0].cpu().numpy().astype(float)
            valid = extra['valid'][0].cpu().numpy().copy()
            probs = torch.sigmoid(seg[0]).cpu().float().numpy()
            H, W = seg.shape[-2:]
            M = np.diag([camera['width']/W, camera['height']/H])
            uv = uv_model @ M.T
            covariance = np.einsum('ab,...bc,dc->...ad', M, cov_model, M)
            hypotheses = [{'uv': (extra['hyps'][0][k].cpu().numpy().astype(float) @ M.T).tolist(),
                'original_scores': extra['scores'][0][k].cpu().numpy().astype(float).tolist()} for k in range(7)]
        mask = cv2.resize((probs.max(axis=0) >= self.vote_cfg.seg_thr).astype('uint8'),
            (camera['width'], camera['height']), interpolation=cv2.INTER_NEAREST)
        arrays = {'keypoints_uv': uv, 'covariance_uv': covariance, 'valid_keypoints': valid,
            'q_rad': np.asarray(row['q_rad'], float), 'keypoints_base_m': np.asarray(row['keypoints_base_m'], float),
            'foreground_exclusion_mask': mask, 'keypoints_model_uv': uv_model, 'covariance_model_uv': cov_model}
        metadata = {k: value for k, value in row.items() if k not in ARRAY_FIELDS}
        metadata.update(hypotheses=hypotheses, invalid_reasons=['' if value else 'original_voting_invalid' for value in valid],
            pixel_space=camera['pixel_space'], preprocessing_transform={'M_model_to_native': M.tolist(),
                'b_model_to_native': [0., 0.], 'input_size': [640, 480], 'RGB': True,
                'resize': 'torchvision tensor Resize antialias=True', 'normalize_imagenet': True,
                'pixel_center_convention': 'original infer_live and dataloader scale-only coordinates; no added half-pixel offset',
                'crop': None, 'padding': None},
            timing_s={'model_preprocessing': model_s, 'voting': vote_s}, source_device='cuda', source_dtype='float32')
        return arrays, metadata


def produce(args, config, contract, pairs):
    import numpy as np
    from pairwise.bridge import ObservationStore, _validate_current, separate_candidates
    source_identity = identity(contract)
    frame_dir = args.output/'frames'
    frame_dir.mkdir(exist_ok=True)
    pending = []
    rows = {}
    for row in contract['frames']:
        key = row['camera_id']+'/'+row['frame_id']
        path = frame_dir/(key.replace('/', '_')+'.json')
        if path.exists():
            saved = read_json(path)
            if saved['input_contract_identity'] != source_identity or saved['image_sha256'] != row['image_sha256']:
                raise ValueError('Resume frame belongs to another input contract')
            if sha256(saved['arrays_path']) != saved['arrays_sha256']:
                raise ValueError('Resume raw array hash mismatch')
            with np.load(saved['arrays_path'], allow_pickle=False) as data:
                _validate_current(saved, {key: data[key] for key in data.files})
            rows[key] = saved
        else:
            pending.append(row)
    print(json.dumps({'stage': 'raw_observation_start', 'total': len(contract['frames']),
        'resume_frames': len(rows), 'pending': len(pending), 'device': 'cuda', 'dtype': 'float32'}), flush=True)
    producer = FrozenRawProducer(args.checkpoint, args.cpu_threads) if pending else None
    actual_voting = asdict(producer.vote_cfg) if producer else read_json(args.output/'producer.json')['voting']
    producer_meta = {'network': contract['network'], 'source_sha256': contract['source_sha256'],
        'checkpoint_sha256': contract['checkpoint_sha256'], 'voting': actual_voting,
        'strict_checkpoint_loading': True, 'device': 'cuda', 'dtype': 'float32', 'amp': False,
        'input_contract_identity': source_identity}
    if producer:
        producer_meta.update(torch_version=producer.torch.__version__,
            gpu_name=producer.torch.cuda.get_device_name(0), cuda_version=producer.torch.version.cuda)
    if (args.output/'producer.json').exists():
        previous = read_json(args.output/'producer.json')
        for key in ('voting', 'checkpoint_sha256', 'source_sha256', 'input_contract_identity'):
            if identity(previous[key]) != identity(producer_meta[key]):
                raise ValueError('Resume producer contract differs')
    else:
        write_json(args.output/'producer.json', producer_meta)
    start = time.perf_counter()
    for index, row in enumerate(pending):
        key = row['camera_id']+'/'+row['frame_id']
        arrays, metadata = producer.infer(row, contract['cameras'][row['camera_id']])
        _validate_current(metadata, arrays)
        # Random attempt suffix permits recovery after an interrupted NPZ write
        # without replacing any existing artifact. JSON is the commit marker.
        array_path = frame_dir/(key.replace('/', '_')+'_'+uuid.uuid4().hex[:8]+'.npz')
        with array_path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        metadata.update(arrays_path=str(array_path), arrays_sha256=sha256(array_path),
                        input_contract_identity=source_identity)
        write_json(frame_dir/(key.replace('/', '_')+'.json'), metadata)
        rows[key] = metadata
        if index % args.progress_every == 0 or index+1 == len(pending):
            print(json.dumps({'stage': 'raw_observation_progress', 'done': len(rows),
                'total': len(contract['frames']), 'last': key,
                'elapsed_s': time.perf_counter()-start, 'model_s': metadata['timing_s']['model_preprocessing'],
                'voting_s': metadata['timing_s']['voting']}), flush=True)
    metadata = {'schema_version': 2, 'pairs_path': str(args.output/'pairs.jsonl'),
        'pairs_sha256': sha256(args.output/'pairs.jsonl'), 'source_cache': str(args.output),
        'source_cache_identity': source_identity, 'robot_checkpoint_sha256': contract['checkpoint_sha256'],
        'cameras': contract['cameras'], 'frames': rows, 'source_arrays_read_only': True,
        'source_array_keys_allowlisted': list(ARRAY_FIELDS[:6]), 'no_temporal_pose_fields': True,
        'no_reference_fields': True, 'candidates_available': all(len(row['hypotheses']) == 7 for row in rows.values()),
        'candidate_kernel_covariance_available': False, 'candidate_score_policy': 'normalized_nonnegative_vote_support',
        'preprocessing': contract['preprocessing'], 'voting': actual_voting, 'source_device': 'cuda',
        'source_dtype': 'float32', 'fresh_robot_forward_executed': True,
        'input_contract_path': str(args.output/'input_contract.json')}
    observation_path = args.output/'observations.json'
    candidate_path = args.output/'candidates.json'
    if not observation_path.exists():
        if candidate_path.exists():
            # Resume a crash between candidate and observation manifests without
            # replacing the already committed candidate file.
            candidate = read_json(candidate_path)
            expected = {key: {'camera_id': row['camera_id'], 'frame_id': row['frame_id'],
                'image_sha256': row['image_sha256'], 'arrays_sha256': row['arrays_sha256'],
                'hypotheses': row['hypotheses']} for key, row in rows.items()}
            if identity(candidate['frames']) != identity(expected) or candidate['source_cache_identity'] != source_identity:
                raise ValueError('Existing candidate manifest differs')
            for row in metadata['frames'].values():
                row.pop('hypotheses')
            metadata.update(candidate_manifest_path=str(candidate_path), candidate_manifest_sha256=sha256(candidate_path),
                candidate_manifest_identity=candidate['identity'], candidate_storage='separate_verified_original_hypotheses')
            metadata['identity'] = identity(metadata)
        else:
            metadata = separate_candidates(metadata, config, candidate_path)
        write_json(observation_path, metadata)
    store = ObservationStore(observation_path)
    for pair in pairs:
        store.pair(pair)
    final = {'passed': True, 'pairs': len(pairs), 'images': len(rows), 'split': 'all',
        'observation_manifest': str(observation_path), 'observation_sha256': sha256(observation_path),
        'candidate_manifest': str(candidate_path), 'candidate_sha256': sha256(candidate_path),
        'checkpoint_sha256': contract['checkpoint_sha256'], 'CUDA_FP32': True,
        'all_ObservationStore_pairs_validated': True, 'annotation_pose_used_for_inference': False,
        'source_sha256_after': {f: sha256(REPO/f) for f in SOURCE_FILES}}
    if final['source_sha256_after'] != contract['source_sha256'] or sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('Original source or checkpoint changed during import')
    immutable_json(args.output/'validation.json', final)
    print(json.dumps(final), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=REPO/'Data/real/Dual_UR/UR_dual_2596')
    parser.add_argument('--checkpoint', type=Path, default=REPO/'trained_models/ur5e_new2/last.pt')
    parser.add_argument('--config', type=Path, default=EXT/'configs/ur_dual_2596_pairwise.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='Optional initial pair subset for a distinct smoke output directory')
    parser.add_argument('--prepare-only', action='store_true', help='Validate inputs and produce only all-split pair and input manifests')
    parser.add_argument('--prepare-reference', action='store_true', help='Separate evaluation-only reference action; does not run model inference')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--progress-every', type=int, default=20)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.limit is not None and not 1 <= args.limit <= 2596:
        parser.error('--limit must be between 1 and 2596')
    if args.progress_every < 1 or args.cpu_threads < 1:
        parser.error('Progress interval and CPU thread count must be positive')
    require_original_sources(SOURCE_FILES)
    configure_runtime(args.output)
    if args.prepare_reference:
        from pairwise.bridge import read_pairs
        from pairwise.evaluation import prepare_reference
        pairs = read_pairs(args.output/'pairs.jsonl')
        print(json.dumps(prepare_reference(pairs, args.dataset, args.output/'reference.jsonl')), flush=True)
        return
    config, contract, pairs = prepare_dataset(args)
    print(json.dumps({'stage': 'dataset_validated', 'pairs': len(pairs), 'split': 'all',
        'pairs_path': str(args.output/'pairs.jsonl'), 'input_contract_identity': identity(contract)}), flush=True)
    if not args.prepare_only:
        produce(args, config, contract, pairs)


if __name__ == '__main__':
    main()
