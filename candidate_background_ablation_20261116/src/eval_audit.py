"""Independent audit: stored transforms -> FK ADD/PCK -> every summary and CI.

This intentionally does not import the evaluator's metric/aggregation/bootstrap
functions. Bootstrap uses a direct pair-weight matrix, unlike its block-sum path.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import cv2
import numpy as np


def audit(directory,output):
    directory=Path(directory).resolve();output=Path(output).resolve()
    root=Path(__file__).resolve().parents[1]
    assert output.is_relative_to(root) and not output.exists()
    metadata=json.loads((directory/'evaluation.json').read_text())
    paths=metadata['paths']
    with np.load(directory/'evaluation_arrays.npz') as archive:data={k:archive[k] for k in archive.files}
    ids=list(data['pair_ids']);index={pid:i for i,pid in enumerate(ids)}
    pairs={r['pair_id']:r for r in map(json.loads,Path(paths['pairs']).open())}
    refs={r['pair_id']:r for r in map(json.loads,Path(paths['reference']).open())}
    methods=['R0','R1','E0','E1'];cameras=['camera1','camera2'];thresholds=np.arange(0,101,5)
    checks=0;max_add_difference=0.;max_pck_difference=0.
    def check(actual,expected,tol=1e-9):
        nonlocal checks
        assert np.isclose(actual,expected,atol=tol,rtol=1e-10,equal_nan=True),(actual,expected)
        checks+=1
    def read(path):return list(csv.DictReader(Path(path).open()))
    def value(row,key):return float(row[key]) if row.get(key) not in ('',None) else np.nan
    count=0
    for result in map(json.loads,(Path(metadata['run_dir'])/'pair_results.jsonl').open()):
        pid=result['pair_id'];i=index[pid];count+=1
        if count%650==0:print(json.dumps({'audited_pairs':count}),flush=True)
        expected_time=(pairs[pid]['cameras']['camera1']['timestamp_s']+pairs[pid]['cameras']['camera2']['timestamp_s'])/2
        check(data['timestamps'][i],expected_time)
        assert bool(data['pre_eligible'][i])==result['pre_eligible']
        for j,m in enumerate(methods):
            record=result['methods'][m]
            for k,c in enumerate(cameras):
                cam=pairs[pid]['cameras'][c];ref=refs[pid]['cameras'][c]
                xyz=np.asarray(cam['keypoints_base_m']);hom=np.c_[xyz,np.ones(7)]
                estimate=record['poses'][c];reference=np.asarray(ref['T_camera_from_base'])
                add=np.inf;uv=np.full((7,2),np.nan)
                if estimate is not None:
                    estimate=np.asarray(estimate);points=(estimate@hom.T).T[:,:3];target=(reference@hom.T).T[:,:3]
                    add=np.linalg.norm(points-target,axis=1).mean()*1000
                    valid=np.isfinite(points).all(1)&(points[:,2]>.001)
                    if valid.any():uv[valid]=cv2.projectPoints(points[valid],np.zeros(3),np.zeros(3),np.asarray(cam['K']),np.asarray(cam['D']))[0].reshape(-1,2)
                check(data['add_before_gate'][i,j,k],add,tol=1e-6)
                if np.isfinite(add):max_add_difference=max(max_add_difference,abs(data['add_before_gate'][i,j,k]-add))
                for g in ['common','final']:assert bool(data[g+'_valid'][i,j,k])==record[g+'_valid'];checks+=1
                refuv=np.asarray(ref['projected_keypoints_uv']);inside=np.isfinite(refuv).all(1)&(refuv[:,0]>=0)&(refuv[:,0]<cam['width'])&(refuv[:,1]>=0)&(refuv[:,1]<cam['height'])
                raw=record['raw_inputs'][c]
                for track,pred,flag in [('raw',raw['keypoints_uv'],raw['valid_keypoints']),('map',record['candidate_selected_uv'][c],record['candidate_selected_valid'][c]),('projected',uv,None)]:
                    expected=np.full(7,np.nan);expected[inside]=np.inf
                    if pred is not None:
                        pred=np.asarray(pred,float);ok=inside&np.isfinite(pred).all(1)
                        if flag is not None:ok&=np.asarray(flag,bool)
                        expected[ok]=np.linalg.norm(pred[ok]-refuv[ok],axis=1)
                    actual=data[track+'_errors'][i,j,k]
                    assert np.array_equal(np.isnan(actual),np.isnan(expected))
                    assert np.array_equal(np.isinf(actual),np.isinf(expected))
                    finite=np.isfinite(expected)
                    if finite.any():
                        diff=float(np.max(np.abs(actual[finite]-expected[finite])));max_pck_difference=max(max_pck_difference,diff);assert diff<1e-7
                    checks+=7
    assert count==metadata['n_pairs']==len(ids)
    summary=read(directory/'summary_metrics.csv');paired=read(directory/'paired_comparisons.csv')
    method_sensitivity=read(directory/'method_bootstrap_sensitivity.csv');contrast_sensitivity=read(directory/'bootstrap_sensitivity.csv')
    # All source pair rows remain on the same original time axis.
    sums=[];denominators=[];rowspec=[]
    for row in summary:
        j=methods.index(row['method']);cs=[0,1] if row['camera']=='pooled' else [cameras.index(row['camera'])]
        include=np.ones(len(ids),bool) if row['population']=='full' else data['pre_eligible']
        good=data[row['gate']+'_valid'][:,j][:,cs]
        raw=data['add_before_gate'][:,j][:,cs];e=np.where(good,raw,np.inf)
        selected=e[include].ravel();finite=selected[np.isfinite(selected)]
        check(value(row,'n_total'),len(selected));check(value(row,'n_valid'),len(finite));check(value(row,'n_failed'),len(selected)-len(finite))
        curve=(selected[:,None]<thresholds).mean(0)
        check(value(row,'ADD50'),np.mean(selected<50));check(value(row,'ADD_AUC'),np.trapezoid(curve,thresholds)/100)
        for key,target in [('ADD_mean_mm',finite.mean() if len(finite) else np.nan),('ADD_median_mm',np.median(finite) if len(finite) else np.nan),('ADD_p90_mm',np.percentile(finite,90) if len(finite) else np.nan)]:check(value(row,key),target)
        for track in ('raw','map','projected'):
            error=data[track+'_errors'][:,j][:,cs][include].copy()
            before=error[~np.isnan(error)];check(value(row,track+'_PCK5_before_gate'),np.mean(before<5))
            if track!='raw':error[(~good[include])[...,None]&~np.isnan(error)]=np.inf
            ev=error[~np.isnan(error)];check(value(row,track+'_PCK5'),np.mean(ev<5));check(value(row,track+'_reference_points'),len(ev))
        whole=np.broadcast_to(include[:,None],e.shape);valid=whole&np.isfinite(e)
        curveper=(e[...,None]<thresholds).astype(float)
        perauc=(curveper[...,:-1]+curveper[...,1:]).sum(-1)*.025
        for metric,num,den in [('ADD50',((e<50)*whole).sum(1),whole.sum(1)),('ADD_AUC',(perauc*whole).sum(1),whole.sum(1)),('ADD_mean_mm',np.where(valid,e,0).sum(1),valid.sum(1))]:
            sums.append(num);denominators.append(den);rowspec.append(('method',row,metric))
    for row in paired:
        cs=[0,1] if row['camera']=='pooled' else [cameras.index(row['camera'])]
        included=np.ones(len(ids),bool) if row['population']=='full' else data['pre_eligible']
        error=np.where(data[row['gate']+'_valid'],data['add_before_gate'],np.inf)[:,:,cs]
        whole=np.broadcast_to(included[:,None],error[:,0].shape)
        if row['contrast'].startswith('('):
            valid=np.isfinite(error).all(1)&whole
            safe=np.where(valid[:,None,:],error,0);delta=safe[:,3]-safe[:,2]-safe[:,1]+safe[:,0]
        else:
            a=error[:,methods.index(row['before'])];b=error[:,methods.index(row['after'])]
            valid=np.isfinite(a)&np.isfinite(b)&whole
            delta=np.where(valid,b,0)-np.where(valid,a,0)
        check(value(row,'n_common_valid'),valid.sum());check(value(row,'n_denominator'),whole.sum())
        d=delta[valid];check(value(row,'paired_ADD_delta_mm'),d.mean() if len(d) else np.nan)
        sums.append(delta.sum(1));denominators.append(valid.sum(1));rowspec.append(('contrast',row,'paired_ADD_delta_mm'))
        if not row['contrast'].startswith('('):
            check(value(row,'improved_fraction'),np.mean(d<0));check(value(row,'worsened_fraction'),np.mean(d>0))
            for metric,num in [('ADD50_delta',(((b<50).astype(float)-(a<50))*whole).sum(1)),('ADD_AUC_delta',((((b[...,None]<thresholds).astype(float)[...,:-1]+(b[...,None]<thresholds).astype(float)[...,1:]-(a[...,None]<thresholds).astype(float)[...,:-1]-(a[...,None]<thresholds).astype(float)[...,1:]).sum(-1)*.025)*whole).sum(1))]:
                check(value(row,metric),num.sum()/whole.sum());sums.append(num);denominators.append(whole.sum(1));rowspec.append(('contrast',row,metric))
    numer=np.stack(sums,axis=1);denom=np.stack(denominators,axis=1)
    bootstrap=np.load(directory/'bootstrap_indices.npz');ci_checks=0
    for seconds in (1.,.5,2.):
        assignment=bootstrap[f'block_{seconds}_assignment'];counts=bootstrap[f'block_{seconds}_counts']
        # Explicit weight of each original pair, equivalent to replicating it.
        weights=counts[:,assignment].astype(float)
        ns=weights@numer;ds=weights@denom
        samples=np.divide(ns,ds,out=np.full_like(ns,np.nan),where=ds>0)
        for col,(kind,row,metric) in enumerate(rowspec):
            table=method_sensitivity if kind=='method' else contrast_sensitivity;identity_key='method' if kind=='method' else 'contrast'
            target=next(r for r in table if float(r['block_seconds'])==seconds and all(r[k]==row[k] for k in ('gate','population','camera',identity_key)))
            vals=samples[:,col];finite=vals[np.isfinite(vals)]
            lower,upper=np.percentile(finite,[2.5,97.5]) if len(finite) else [np.nan,np.nan]
            check(value(target,metric+'_ci_low'),lower);check(value(target,metric+'_ci_high'),upper)
            check(value(target,metric+'_bootstrap_valid'),len(finite));ci_checks+=1
    result=dict(passed=True,n_pairs=len(ids),pose_camera_rows=count*8,summary_rows=len(summary),paired_rows=len(paired),
        direct_pose_and_PCK_checks=True,all_full_and_preeligible_gates_checked=True,
        bootstrap_CI_metrics_checked=ci_checks,bootstrap_independent_pair_weights=True,
        checks=checks,max_ADD_absolute_difference_mm=max_add_difference,max_PCK_absolute_difference_px=max_pck_difference,
        evaluation_directory=str(directory),inference_rerun=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with output.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation-dir',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();audit(a.evaluation_dir,a.output)
