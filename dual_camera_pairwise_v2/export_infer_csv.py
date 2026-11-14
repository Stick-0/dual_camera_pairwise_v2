#!/usr/bin/env python3
"""Export completed V2 predictions with the original infer_dir CSV column layout.

No inference, matching, reference labels or fallback-to-stereo mixing occurs.
"""
import argparse
import csv
import json
from contextlib import ExitStack
from pathlib import Path

from pairwise.runtime import configure_runtime,output_path,sha256,write_json
configure_runtime()

import numpy as np
from scipy.spatial.transform import Rotation
from pairwise.bridge import ObservationStore,read_pairs
from pairwise.geometry import project,transform

KEYPOINT_NAMES=[f'ur5e_link{k}' for k in range(7)]
POSE_COLUMNS=['quat_x','quat_y','quat_z','quat_w','tvec_x','tvec_y','tvec_z','reproj_mean','pnp_success']
CSV_COLUMNS=['image']+[f'{k}_x' for k in KEYPOINT_NAMES]+[f'{k}_y' for k in KEYPOINT_NAMES]+POSE_COLUMNS
DUAL_METHODS={'dual_mean','dual_candidates'}


def _pose(value):
    if value is None:return None
    matrix=np.asarray(value,float)
    if matrix.shape!=(4,4) or not np.isfinite(matrix).all():return None
    R=matrix[:3,:3]
    if not np.allclose(matrix[3],[0,0,0,1],atol=1e-8):return None
    if not np.allclose(R.T@R,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(R),1.,atol=1e-5):return None
    return matrix


def _accepted_pose(result,camera_id):
    if result.get('status') in ('geometry_failed','metric_scale_unobservable','joint_pose_failed',
                               'blocked_matcher_dependency','blocked_missing_pretrained_weights','candidates_unavailable'):
        return None
    return _pose(result.get('poses',{}).get(camera_id))


def _valid_stereo(result):
    stereo=result.get('stereo_pose')
    return (result.get('status')=='stereo_refined' and isinstance(stereo,dict)
            and result.get('quality',{}).get('actual_rgb_correspondences_used',0)>0
            and all(_pose(stereo.get(c)) is not None for c in ('camera1','camera2')))


def _blank_nonfinite(value):
    return float(value) if np.isfinite(value) else ''


def _csv_row(image_name,uv,T,reprojection_mean):
    uv=np.asarray(uv,float)
    coordinates=[_blank_nonfinite(v) for v in np.r_[uv[:,0],uv[:,1]]]
    if T is None:
        pose_values=['']*8+[0]
    else:
        q=Rotation.from_matrix(T[:3,:3]).as_quat()
        pose_values=q.tolist()+T[:3,3].tolist()+[_blank_nonfinite(reprojection_mean),1]
    return [image_name]+coordinates+pose_values


def export_csv(run,pairs_path,observations_path,output):
    run=Path(run).resolve()
    manifest_path=run/'manifest.json' if run.is_dir() else run
    manifest=json.loads(manifest_path.read_text())
    if not manifest.get('completed'):
        raise ValueError('Export requires a completed run manifest')
    if manifest.get('pairs_sha256')!=sha256(pairs_path):
        raise ValueError('Run/pair manifest hash mismatch')
    store=ObservationStore(observations_path)
    if manifest.get('observations_identity')!=store.metadata['identity']:
        raise ValueError('Run/observation identity mismatch')
    pairs={p['pair_id']:p for p in read_pairs(pairs_path)}
    sources=manifest.get('results',[])
    if not sources or len({r['pair_id'] for r in sources})!=len(sources):
        raise ValueError('Completed run must have nonempty unique result rows')
    methods=list(manifest['methods'])
    destination=output_path(output);destination.mkdir(parents=True,exist_ok=False)
    artifacts={method:{} for method in methods}
    counts={method:{c:{'rows':0,'pose_success':0,'stereo_refined':0} for c in ('camera1','camera2')} for method in methods}
    status_columns=['pair_id','split','method','camera_id','image','status','reason','pose_success',
                    'stereo_refined','fallback_available','raw_valid_keypoints','candidate_selected_valid',
                    'source_result_sha256']
    with ExitStack() as stack:
        writers={}
        for method in methods:
            for cid in ('camera1','camera2'):
                stages=['raw_voting','pose_projected']+(['candidate_selected'] if method.endswith('_candidates') else [])
                artifacts[method][cid]={}
                for stage in stages:
                    path=destination/f'{method}_{cid}_{stage}.csv'
                    writer=csv.writer(stack.enter_context(path.open('x',newline='')))
                    writer.writerow(CSV_COLUMNS);writers[(method,cid,stage)]=writer
                    artifacts[method][cid][stage]=str(path)
        status_path=destination/'row_statuses.csv'
        status_writer=csv.DictWriter(stack.enter_context(status_path.open('x',newline='')),fieldnames=status_columns)
        status_writer.writeheader()
        for source in sources:
            pid=source['pair_id'];pair=pairs[pid]
            result_path=Path(source['result_path'])
            if sha256(result_path)!=source['sha256']:
                raise ValueError(f'Prediction result changed: {result_path}')
            record=json.loads(result_path.read_text())
            if record['pair_id']!=pid or record.get('split')!=pair['split']:
                raise ValueError('Result/pair identity or split mismatch')
            if record.get('config_identity')!=manifest.get('config_identity'):
                raise ValueError('Result/run configuration mismatch')
            observations=store.pair(pair)
            for method in methods:
                prediction=record['methods'][method]
                stereo=_valid_stereo(prediction) if method in DUAL_METHODS else False
                if method in DUAL_METHODS and prediction.get('status')=='stereo_refined' and not stereo:
                    raise ValueError(f'False stereo success: {pid}/{method}')
                for cid in ('camera1','camera2'):
                    raw=observations[cid]
                    T=_accepted_pose(prediction,cid)
                    if method in DUAL_METHODS:
                        T=_pose(prediction['stereo_pose'][cid]) if stereo else None
                        if stereo:
                            supplied=_pose(prediction.get('poses',{}).get(cid))
                            if supplied is None or not np.allclose(T,supplied,atol=1e-8):
                                raise ValueError('poses and genuine stereo_pose disagree')
                    raw_uv=np.array(raw['keypoints_uv'],float,copy=True)
                    raw_valid=np.array(raw['valid_keypoints'],bool,copy=True)
                    raw_uv[~raw_valid]=np.nan
                    projected=np.full((7,2),np.nan)
                    reprojection_mean=np.nan
                    if T is not None:
                        camera_points=transform(T,raw['keypoints_base_m'])
                        projected=project(camera_points,pair['cameras'][cid])
                        projected[camera_points[:,2]<=.001]=np.nan
                        finite=raw_valid & np.isfinite(raw_uv).all(1) & np.isfinite(projected).all(1)
                        if finite.any():reprojection_mean=float(np.mean(np.linalg.norm(raw_uv[finite]-projected[finite],axis=1)))
                    image_name=Path(pair['cameras'][cid]['image_path']).name
                    writers[(method,cid,'raw_voting')].writerow(_csv_row(image_name,raw_uv,T,reprojection_mean))
                    writers[(method,cid,'pose_projected')].writerow(_csv_row(image_name,projected,T,reprojection_mean))
                    candidate_valid=np.zeros(7,bool)
                    if method.endswith('_candidates'):
                        candidate=np.full((7,2),np.nan)
                        value=prediction.get('candidate_selected_uv',{}).get(cid)
                        if T is not None and value is not None:
                            candidate=np.asarray(value,float).copy()
                            candidate_valid=np.asarray(prediction.get('candidate_selected_valid',{}).get(cid,[False]*7),bool)
                            candidate_valid &= raw_valid & np.isfinite(candidate).all(1)
                            candidate[~candidate_valid]=np.nan
                        writers[(method,cid,'candidate_selected')].writerow(_csv_row(image_name,candidate,T,reprojection_mean))
                    detail=prediction.get('mono_quality',{}).get(cid,{})
                    status_writer.writerow({'pair_id':pid,'split':pair['split'],'method':method,'camera_id':cid,
                        'image':image_name,'status':prediction.get('status'),'reason':prediction.get('reason') or (detail.get('reason') if T is None and method not in DUAL_METHODS else ''),
                        'pose_success':int(T is not None),'stereo_refined':int(stereo),
                        'fallback_available':int(_pose((prediction.get('fallback_pose') or {}).get(cid)) is not None),
                        'raw_valid_keypoints':json.dumps(raw_valid.tolist()),'candidate_selected_valid':json.dumps(candidate_valid.tolist()),
                        'source_result_sha256':source['sha256']})
                    counts[method][cid]['rows']+=1
                    counts[method][cid]['pose_success']+=int(T is not None)
                    counts[method][cid]['stereo_refined']+=int(stereo)
    result={'schema_version':2,'completed':True,'source_run_manifest':str(manifest_path),
        'source_run_manifest_sha256':sha256(manifest_path),'pairs_manifest_sha256':sha256(pairs_path),
        'observations_manifest_sha256':sha256(observations_path),'reference_accessed':False,
        'original_infer_dir_layout_source':'robotpose/infer_dir.py:677-733',
        'columns':CSV_COLUMNS,'keypoint_order':KEYPOINT_NAMES,'coordinates':'original native distorted pixels',
        'quaternion_order':'xyzw','translation_unit':'m','reprojection_error_unit':'native pixels',
        'reproj_mean_definition':'mean Euclidean raw valid UV to final pose projection residual; no temporal correction; same pose column on all CSV stages',
        'pnp_success_definition':'valid provided pose for this method/camera; column name retained for infer_dir compatibility, not a claim that every solver is PnP',
        'invalid_raw_policy':'blank coordinates for original invalid flags; unmodified numeric raw arrays remain in source cache',
        'failed_pose_policy':'blank quaternion/translation/reprojection with pnp_success=0; rows retained',
        'stereo_policy':'only status stereo_refined with actual RGB factors and both valid stereo poses; operational fallback excluded',
        'stage_meanings':{'raw_voting':'copied current raw mean with original invalid flags blanked',
            'candidate_selected':'original candidate selected by final posterior, outlier/invalid blank',
            'pose_projected':'FK(q_camera) projected by final pose, not new raw detections'},
        'artifacts':artifacts,'row_statuses':str(status_path),'counts':counts}
    result['artifact_sha256']={str(path.relative_to(destination)):sha256(path) for path in sorted(destination.glob('*.csv'))}
    write_json(destination/'manifest.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('run','pairs','observations','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    result=export_csv(args.run,args.pairs,args.observations,args.output)
    print(json.dumps({'output':str(Path(args.output).resolve()),'counts':result['counts'],'completed':True},indent=2))


if __name__=='__main__':main()
