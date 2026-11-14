"""Read-only, allowlisted bridge to frozen raw voting observations.

Never imports the previous extension or reads its pose/calibration outputs.
Reference annotations are not opened by this module. Encoder and timestamp
provenance was validated by the recorder importer and is independently checked
here using only whitelisted current-observation metadata.
"""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from .runtime import EXT_ROOT, REPO_ROOT, identity, output_path, sha256, write_json, jsonable, require_original_sources

ARRAY_KEYS = ('keypoints_uv','covariance_uv','valid_keypoints','q_rad','keypoints_base_m','foreground_exclusion_mask')
META_KEYS = ('frame_id','camera_id','image_path','timestamp_s','timestamp_source','q_source',
             'q_association','recorded_joint_interpolation','pixel_space','preprocessing_transform',
             'hypotheses','invalid_reasons','timing_s','seed','observation_stage')


def load_config(path):
    import yaml
    config=yaml.safe_load(Path(path).read_text())
    if config.get('schema_version')!=2:
        raise ValueError('schema_version must be 2')
    expected={'pose_mode':'per_pair','geometry_mode':'current_pair','temporal_prior':False,
        'use_static_base_pose':False,'aggregate_robot_frames':False,'reuse_previous_robot_pose':False,
        'reuse_previous_relative_pose':False,'use_legacy_B_initialization':False,'use_reference_for_inference':False}
    if any(config.get('estimation',{}).get(k)!=v for k,v in expected.items()):
        raise ValueError('Current-pair contract cannot be relaxed')
    if config.get('observations',{}).get('robot_candidate_epipolar_gate')!='disabled':
        raise ValueError('Asynchronous robot epipolar gating is unsupported; keep disabled')
    for key in ('use_sequence_union_mask','use_temporal_background_composite','training_enabled','use_match_score_as_inverse_variance'):
        if config.get('matching',{}).get(key,False):
            raise ValueError(f'Unsupported matching option: {key}')
    return config




def _fk(q):
    require_original_sources(['ur5e_capture/geometry.py'])
    name='_pairwise_v2_original_fk'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,REPO_ROOT/'ur5e_capture/geometry.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return np.asarray(sys.modules[name].ur5e_keypoints_from_joints(q),float)


def _arrays(path,digest):
    if sha256(path)!=digest:
        raise ValueError(f'Raw arrays changed: {path}')
    with np.load(path,allow_pickle=False) as data:
        return {key:data[key].copy() for key in ARRAY_KEYS}


def _validate_current(row,arrays):
    shapes={'keypoints_uv':(7,2),'covariance_uv':(7,2,2),'valid_keypoints':(7,),
            'q_rad':(6,),'keypoints_base_m':(7,3)}
    for name,shape in shapes.items():
        if arrays[name].shape!=shape:
            raise ValueError(f'{name} has unexpected shape')
    valid=arrays['valid_keypoints'].astype(bool)
    if not all(np.isfinite(arrays[key][valid]).all() for key in ('keypoints_uv','covariance_uv','keypoints_base_m')):
        raise ValueError('Nonfinite valid raw observation')
    if not np.allclose(_fk(arrays['q_rad']),arrays['keypoints_base_m'],rtol=0,atol=1e-12):
        raise ValueError('Cached FK differs from unchanged original encoder FK')
    interpolation=row['recorded_joint_interpolation']
    if abs(interpolation['target_monotonic_s']-row['timestamp_s'])>1e-8:
        raise ValueError('Camera and encoder association times differ')
    qa=np.array(interpolation['before_q_rad']);qb=np.array(interpolation['after_q_rad'])
    fraction=interpolation['interpolation_fraction']
    if not np.allclose(qa+fraction*(qb-qa),arrays['q_rad'],rtol=0,atol=1e-9):
        raise ValueError('Encoder interpolation provenance mismatch')
    if row['observation_stage']!='raw_voting':
        raise ValueError('Only raw observations accepted')






def read_pairs(path,split='all',limit=None,pair_ids=None):
    with Path(path).open() as file:pairs=[json.loads(line) for line in file if line.strip()]
    if split!='all':pairs=[p for p in pairs if p['split']==split]
    if pair_ids is not None:
        wanted=set(pair_ids);pairs=[p for p in pairs if p['pair_id'] in wanted]
        if {p['pair_id'] for p in pairs}!=wanted:raise ValueError('Unknown pair ID for selected split')
    return pairs[:limit] if limit is not None else pairs


def separate_candidates(metadata, config, candidate_output, source_manifest=None):
    """Create a new physical raw-candidate cache without changing numeric inputs.

    The caller writes the returned observation metadata to a NEW path. Existing
    embedded manifests remain readable; no original manifest is edited in place.
    """
    observation = copy.deepcopy(metadata)
    observation.pop('identity', None)
    candidate_frames = {}
    for key, row in observation['frames'].items():
        hypotheses = row.pop('hypotheses', None)
        candidate_frames[key] = {
            'camera_id': row['camera_id'], 'frame_id': row['frame_id'],
            'image_sha256': row['image_sha256'], 'arrays_sha256': row['arrays_sha256'],
            'hypotheses': hypotheses,
        }
    candidate = {
        'schema_version': 2, 'kind': 'original_voting_candidates',
        'source_cache': observation['source_cache'],
        'source_cache_identity': observation['source_cache_identity'],
        'robot_checkpoint_sha256': observation['robot_checkpoint_sha256'],
        'pairs_sha256': observation['pairs_sha256'], 'frames': candidate_frames,
        'raw_coordinates_and_scores_unchanged': True,
        'coordinate_space': 'native_pixels',
        'mode_policy': 'none_original_hypotheses_kept; invalidity and score normalization applied on copies at refinement',
        'component_covariance_available': False,
        'kernel_policy': copy.deepcopy(config.get('candidates', {})),
        'kernel_policy_role': 'frozen refinement settings, not a measured per-component covariance',
        'score_semantics': 'original nonnegative segmentation-weighted angular vote support',
        'posterior_in_raw_cache': False,
    }
    if source_manifest is not None:
        candidate['source_observation_manifest_path'] = str(Path(source_manifest).resolve())
        candidate['source_observation_manifest_sha256'] = sha256(source_manifest)
        observation['source_observation_manifest_path'] = str(Path(source_manifest).resolve())
        observation['source_observation_manifest_sha256'] = sha256(source_manifest)
    candidate['identity'] = identity(candidate)
    candidate_output = output_path(candidate_output)
    write_json(candidate_output, candidate)
    observation.update(candidate_manifest_path=str(candidate_output),
                       candidate_manifest_sha256=sha256(candidate_output),
                       candidate_manifest_identity=candidate['identity'],
                       candidate_storage='separate_verified_original_hypotheses',
                       candidate_schema_amendment='storage only; original coordinates, scores, masks and robot observations unchanged')
    observation['identity'] = identity(observation)
    return observation




class ObservationStore:
    def __init__(self,manifest):
        self.path=Path(manifest).resolve();self.metadata=json.loads(self.path.read_text())
        if identity({k:v for k,v in self.metadata.items() if k!='identity'})!=self.metadata['identity']:
            raise ValueError('Observation manifest identity mismatch')
        self.candidates = None
        path = self.metadata.get('candidate_manifest_path')
        if path:
            candidate_path = Path(path).resolve()
            if sha256(candidate_path) != self.metadata.get('candidate_manifest_sha256'):
                raise ValueError('Candidate manifest SHA-256 mismatch')
            candidates = json.loads(candidate_path.read_text())
            if identity({k:v for k,v in candidates.items() if k!='identity'}) != candidates.get('identity'):
                raise ValueError('Candidate manifest identity mismatch')
            if candidates.get('identity') != self.metadata.get('candidate_manifest_identity'):
                raise ValueError('Linked candidate identity mismatch')
            if candidates.get('kind') != 'original_voting_candidates':
                raise ValueError('Expected original voting candidate manifest')
            for key in ('source_cache_identity', 'robot_checkpoint_sha256', 'pairs_sha256'):
                if candidates.get(key) != self.metadata.get(key):
                    raise ValueError(f'Candidate source mismatch: {key}')
            if candidates['frames'].keys() != self.metadata['frames'].keys():
                raise ValueError('Candidate/observation frame coverage mismatch')
            for key, row in self.metadata['frames'].items():
                candidate = candidates['frames'][key]
                for field in ('camera_id', 'frame_id', 'image_sha256', 'arrays_sha256'):
                    if candidate.get(field) != row.get(field):
                        raise ValueError(f'Candidate source field mismatch: {key}/{field}')
                if 'hypotheses' in row and identity(row['hypotheses']) != identity(candidate.get('hypotheses')):
                    raise ValueError('Embedded and separated original candidates disagree')
            self.candidates = candidates
            self.candidate_provenance = {'storage': 'separate_verified_original_hypotheses',
                'manifest_path': str(candidate_path), 'sha256': self.metadata['candidate_manifest_sha256'],
                'identity': candidates['identity']}
        else:
            self.candidate_provenance = {'storage': 'legacy_embedded_compatibility',
                'manifest_path': str(self.path), 'sha256': sha256(self.path),
                'note': 'Historical v2 manifest, original candidates embedded in each observation; no silent neighboring-cache discovery.'}

    def pair(self,pair):
        observations={}
        for cid,camera in pair['cameras'].items():
            row=self.metadata['frames'][camera['observation_key']]
            if row['image_path']!=camera['image_path'] or row['image_sha256']!=camera['image_sha256']:
                raise ValueError('Pair/observation image identity mismatch')
            raw={k:copy.deepcopy(row[k]) for k in META_KEYS if k in row}
            if self.candidates is not None:
                hypotheses = self.candidates['frames'][camera['observation_key']]['hypotheses']
                if hypotheses is not None:
                    raw['hypotheses'] = copy.deepcopy(hypotheses)
            raw.update(_arrays(row['arrays_path'],row['arrays_sha256']))
            if not np.array_equal(raw['q_rad'],camera['q_rad']):raise ValueError('Pair/observation q mismatch')
            observations[cid]=raw
        return observations
