"""Official, frozen RGB matchers; current-frame endpoint masks only.

No pose, FK, reference or previous-frame input is accepted by this module.
All raw samples and the reason for each exclusion survive in the NPZ cache.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import subprocess

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .runtime import EXT_ROOT, identity, output_path, pair_seed, sha256, write_json

DEFAULTS = dict(sample_before_filter=4096, max_spatially_selected_matches=300,
                mask_dilation_px=8, min_score=0.2, cycle_threshold_px=3.0,
                grid_cell_px=16, min_spacing_px=8.0, seed=0,
                roma_setting='fast', roma_bidirectional=True,
                background_noise_std_px=1.5, device='cuda',mask_shape_policy='original')
SOURCES = {
    'roma_v2': dict(repo='https://github.com/Parskatt/RoMaV2',
                    commit='95c9968145c8906b7b59383258e9f73b02853d89',
                    checkpoint='romav2.0.1.pt',
                    checkpoint_sha256_expected='1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7',
                    checkpoint_url='https://github.com/Parskatt/RoMaV2/releases/download/v2.0.1/romav2.0.1.pt'),
    'loftr_indoor_ds': dict(repo='https://github.com/zju3dv/LoFTR',
                    commit='df7ca80f917334b94cfbe32cc2901e09a80e70a8',
                    checkpoint='indoor_ds.ckpt',
                    checkpoint_sha256_expected='d73c54720370ba0690cd477e38404a626e881292cc0be09a6801da0cc6f53198',
                    checkpoint_url='https://cmp.felk.cvut.cz/~mishkdmy/models/loftr_indoor.ckpt',
                    checkpoint_provenance='Original indoor (not indoor_new) checkpoint mirror in Kornia, linked by official LoFTR README'),
}


def _array_hash(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def current_exclusion_mask(mask, config=None):
    """Optional conservative hull fills missed robot links using this mask only."""
    cfg=DEFAULTS | (config or {})
    result=np.asarray(mask,bool).astype(np.uint8).copy()
    policy=cfg['mask_shape_policy']
    if policy=='current_mask_convex_hull':
        y,x=np.nonzero(result)
        if len(x)>=3:
            hull=cv2.convexHull(np.column_stack((x,y)).astype(np.int32))
            cv2.fillConvexPoly(result,hull,1)
    elif policy!='original':raise ValueError(f'Unsupported current mask policy {policy}')
    radius=int(cfg['mask_dilation_px'])
    if radius<0:raise ValueError('Negative mask dilation')
    if radius:result=cv2.dilate(result,np.ones((2*radius+1,2*radius+1),np.uint8))
    return result.astype(bool)


def filter_matches(points1, points2, scores, cycle_errors, masks, sizes, config=None):
    """Deterministic staged rejection; neither input pixels nor masks are altered."""
    cfg = DEFAULTS | (config or {})
    p1, p2 = np.asarray(points1, float), np.asarray(points2, float)
    scores, cycle_errors = np.asarray(scores, float), np.asarray(cycle_errors, float)
    n = len(p1)
    reasons = np.full(n, 'accepted', dtype='<U40')
    counts = {'raw_candidates': n}
    def reject(condition, name):
        reasons[(reasons == 'accepted') & condition] = name
        counts['after_'+name] = int(np.count_nonzero(reasons == 'accepted'))
    reject(~np.isfinite(np.c_[p1,p2,scores]).all(1), 'nonfinite')
    bounds = np.zeros(n, bool)
    for p, (h,w) in zip((p1,p2), sizes):
        bounds |= (p[:,0]<0)|(p[:,0]>=w)|(p[:,1]<0)|(p[:,1]>=h)
    reject(bounds, 'outside_image')
    reject(scores < cfg['min_score'], 'low_score')
    reject(~np.isfinite(cycle_errors)|(cycle_errors > cfg['cycle_threshold_px']), 'bidirectional_inconsistent')
    masked = np.zeros(n, bool)
    for p, mask, (h,w) in zip((p1,p2), masks, sizes):
        mask = np.asarray(mask, bool)
        if mask.shape != (h,w):
            raise ValueError(f'Current mask shape {mask.shape} differs from native image {(h,w)}')
        dilated = current_exclusion_mask(mask,cfg)
        rounded = np.rint(np.nan_to_num(p, nan=-1)).astype(np.int64)
        masked |= dilated[np.clip(rounded[:,1],0,h-1),np.clip(rounded[:,0],0,w-1)].astype(bool)
    reject(masked, 'current_robot_foreground')
    cell = int(cfg['grid_cell_px']); minimum = float(cfg['min_spacing_px'])
    accepted, seen1, seen2 = [], set(), set()
    # Stable score sort, with raw index as tie-breaker.
    for k in np.argsort(-scores, kind='stable'):
        if reasons[k] != 'accepted':
            continue
        c1, c2 = tuple(np.floor(p1[k]/cell).astype(int)), tuple(np.floor(p2[k]/cell).astype(int))
        if c1 in seen1 or c2 in seen2:
            reasons[k] = 'duplicate_grid_cell'; continue
        if accepted and (np.linalg.norm(p1[accepted]-p1[k],axis=1).min() < minimum or
                         np.linalg.norm(p2[accepted]-p2[k],axis=1).min() < minimum):
            reasons[k] = 'spatial_neighbour'; continue
        if len(accepted) >= int(cfg['max_spatially_selected_matches']):
            reasons[k] = 'selection_budget'; continue
        accepted.append(k); seen1.add(c1); seen2.add(c2)
    valid = reasons == 'accepted'
    counts['spatially_selected'] = int(valid.sum())
    counts['rejections'] = {str(k):int(v) for k,v in zip(*np.unique(reasons,return_counts=True))}
    coverage = []
    for p,(h,w) in zip((p1,p2), sizes):
        coverage.append(float(cv2.contourArea(cv2.convexHull(p[valid].astype(np.float32)))/(h*w)) if valid.sum()>=3 else 0.)
    return valid, reasons, counts, coverage


class Matcher:
    """One frozen model instance, no cross-pair state; seeds reset for every pair."""
    def __init__(self, backend, config=None):
        self.backend = backend
        self.config = DEFAULTS | (config or {})
        self.model = None
        self.load_time_ms = 0.
        self.provenance = dict(SOURCES.get(backend, {}))
        if backend not in SOURCES and backend != 'sift_diagnostic':
            raise ValueError(f'Unsupported backend {backend}')
        if backend == 'sift_diagnostic':
            self.provenance.update(matcher_name=backend, checkpoint_sha256=None,
                                   source_commit=None, opencv_version=cv2.__version__,
                                   inference_config=self.config,
                                   resize_crop_padding_transforms={'resize':None,'crop':None,'padding':None},
                                   score_role='diagnostic_binary_ratio_pass_not_precision')
            return
        sys.path.insert(0,str(EXT_ROOT/'environments/deps'))
        import torch
        if self.config['device'].startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('blocked_matcher_dependency: CUDA unavailable in this process; do not silently use CPU')
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('highest')
        checkpoint = EXT_ROOT/'weights'/SOURCES[backend]['checkpoint']
        if not checkpoint.is_file():
            raise FileNotFoundError(f'blocked_missing_pretrained_weights: {checkpoint}')
        checkpoint_digest=sha256(checkpoint)
        if checkpoint_digest!=SOURCES[backend]['checkpoint_sha256_expected']:
            raise ValueError('Checkpoint differs from explicitly pinned pretrained weights')
        source_dir=EXT_ROOT/'vendor'/('RoMaV2' if backend=='roma_v2' else 'LoFTR')
        actual_commit=subprocess.check_output(['git','-C',str(source_dir),'rev-parse','HEAD'],text=True).strip()
        if actual_commit!=SOURCES[backend]['commit']:raise ValueError('Official source commit differs from pinned source')
        source_clean=not bool(subprocess.check_output(['git','-C',str(source_dir),'diff','--name-only'],text=True).strip())
        if not source_clean:raise ValueError('Official matcher source has local modifications')
        start = time.perf_counter()
        if backend == 'roma_v2':
            sys.path.insert(0,str(EXT_ROOT/'vendor/RoMaV2/src'))
            from romav2 import RoMaV2
            self.model = RoMaV2(RoMaV2.Cfg(setting=self.config['roma_setting']))
            self.model.bidirectional = bool(self.config['roma_bidirectional'])
            if not self.model.bidirectional:
                raise ValueError('This adapter requires bidirectional RoMa predictions for cycle checking')
            transform = {'resize_lr_hw':[self.model.H_lr,self.model.W_lr],
                         'resize_hr_hw':[self.model.H_hr,self.model.W_hr],
                         'interpolation':'bicubic_antialias_align_corners_false',
                         'crop':None,'padding':None,
                         'output_mapping':'official to_pixel_coordinates: (normalized+1)*native_size/2'}
            self.provenance['backbone_source_commit'] = 'adc254450203739c8149213a7a69d8d905b4fcfa'
        else:
            # Import only the official src/loftr package, avoiding generic src namespace collisions.
            source = EXT_ROOT/'vendor/LoFTR/src/loftr'
            spec = importlib.util.spec_from_file_location('_pairwise_official_loftr',source/'__init__.py',submodule_search_locations=[str(source)])
            mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
            self.model = mod.LoFTR(config=copy.deepcopy(mod.default_cfg))
            state = torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict']
            self.model.load_state_dict(state, strict=True)
            self.model.to(self.config['device']).eval()
            transform = {'resize':'native_size_floor_multiple_8','interpolation':'opencv_INTER_AREA',
                         'color':'BGR_to_gray','crop':None,'padding':None,
                         'output_mapping':'multiply native_width/resize_width, native_height/resize_height'}
        self.model.eval()
        for parameter in self.model.parameters(): parameter.requires_grad_(False)
        self.load_time_ms = 1000*(time.perf_counter()-start)
        self.provenance.update(matcher_name=backend,source_commit=SOURCES[backend]['commit'],
            checkpoint_path=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
            source_worktree_clean=source_clean,
            torch_version=torch.__version__,device=self.config['device'],
            model_training=False,strict_checkpoint_loading=True,resize_crop_padding_transforms=transform,
            inference_config=self.config,score_role='selection_only_not_pixel_precision')

    def match_pair(self, pair_id, images, masks):
        """images/masks are two-item camera1/camera2 mappings in native coordinates."""
        images = [str(images[c]) for c in ('camera1','camera2')]
        masks = [np.asarray(masks[c],bool).copy() for c in ('camera1','camera2')]
        rgb = [cv2.imread(p,cv2.IMREAD_COLOR) for p in images]
        if any(im is None for im in rgb): raise FileNotFoundError(images)
        sizes = [im.shape[:2] for im in rgb]
        seed = pair_seed(pair_id,int(self.config['seed']))
        cv2.setRNGSeed(seed); np.random.seed(seed)
        start = time.perf_counter()
        extra = {}
        if self.backend == 'roma_v2':
            import torch
            import torch.nn.functional as F
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                preds = self.model.match(*images)
                matches, scores, _, _ = self.model.sample(preds,int(self.config['sample_before_filter']))
                p1,p2 = self.model.to_pixel_coordinates(matches,*sizes[0],*sizes[1])
                def sample_warp(warp,coords):
                    return F.grid_sample(warp.permute(0,3,1,2),coords[None,None],align_corners=False,mode='bilinear',padding_mode='border')[0,:,0].T
                forward = sample_warp(preds['warp_AB'],matches[:,:2])
                reverse = sample_warp(preds['warp_BA'],matches[:,2:])
                d1 = (reverse-matches[:,:2])*torch.tensor(sizes[0][::-1],device=matches.device)/2
                d2 = (forward-matches[:,2:])*torch.tensor(sizes[1][::-1],device=matches.device)/2
                cycles = torch.maximum(d1.norm(dim=1),d2.norm(dim=1))
                p1,p2,scores,cycles = [x.float().cpu().numpy() for x in (p1,p2,scores,cycles)]
        elif self.backend == 'loftr_indoor_ds':
            import torch
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            inputs=[]; scales=[]
            for im in rgb:
                h,w=im.shape[:2]; h8,w8=(h//8)*8,(w//8)*8
                gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
                gray=cv2.resize(gray,(w8,h8),interpolation=cv2.INTER_AREA)
                inputs.append(torch.from_numpy(gray)[None,None].to(self.config['device'],dtype=torch.float32)/255.)
                scales.append(np.array([w/w8,h/h8]))
            with torch.inference_mode():
                fwd={'image0':inputs[0],'image1':inputs[1]}; self.model(fwd)
                rev={'image0':inputs[1],'image1':inputs[0]}; self.model(rev)
            p1=fwd['mkpts0_f'].cpu().numpy()*scales[0]; p2=fwd['mkpts1_f'].cpu().numpy()*scales[1]
            scores=fwd['mconf'].cpu().numpy()
            rp1=rev['mkpts1_f'].cpu().numpy()*scales[0];rp2=rev['mkpts0_f'].cpu().numpy()*scales[1]
            if len(rp1):
                _,indices=cKDTree(np.c_[rp1,rp2]).query(np.c_[p1,p2])
                cycles=np.maximum(np.linalg.norm(p1-rp1[indices],axis=1),np.linalg.norm(p2-rp2[indices],axis=1))
            else: cycles=np.full(len(p1),np.inf)
            extra['reverse_candidates']=len(rp1)
        else:
            detector=cv2.SIFT_create(nfeatures=6000)
            detected=[detector.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),None) for im in rgb]
            extra['features_detected']=[len(x[0]) for x in detected]
            p1,p2,scores,cycles=[],[],[],[]
            if all(x[1] is not None for x in detected):
                matcher=cv2.BFMatcher(cv2.NORM_L2)
                fw=matcher.knnMatch(detected[0][1],detected[1][1],k=2)
                rv=matcher.knnMatch(detected[1][1],detected[0][1],k=2)
                reverse={a.queryIdx:a.trainIdx for pair in rv if len(pair)==2 for a,b in [pair] if a.distance<.75*b.distance}
                good=[a for pair in fw if len(pair)==2 for a,b in [pair] if a.distance<.75*b.distance]
                extra['descriptor_ratio_passed']=len(good)
                for a in good:
                    p1.append(detected[0][0][a.queryIdx].pt);p2.append(detected[1][0][a.trainIdx].pt)
                    scores.append(1.);cycles.append(0. if reverse.get(a.trainIdx)==a.queryIdx else float('inf'))
            p1,p2=np.asarray(p1).reshape(-1,2),np.asarray(p2).reshape(-1,2)
            scores,cycles=np.asarray(scores),np.asarray(cycles)
        valid,reasons,counts,coverage=filter_matches(p1,p2,scores,cycles,masks,sizes,self.config)
        elapsed=1000*(time.perf_counter()-start)
        provenance=self.provenance | {'pair_id':str(pair_id),'image_sha256':[sha256(p) for p in images],
              'current_mask_sha256':[_array_hash(m) for m in masks],
              'applied_mask_sha256':[_array_hash(current_exclusion_mask(m,self.config)) for m in masks],
              'applied_mask_policy':self.config['mask_shape_policy'],
              'native_image_size_hw':sizes,'pixel_space':'native_distorted','pair_seed':seed,
              'matching_runtime_ms':elapsed,'current_frame_only':True,'mask_before_matching':False,
              'mask_provenance':'exact current raw voting foreground mask copied; configured current-mask shape/dilation only in endpoint filtering',
              'counts':counts|extra,'spatial_convex_hull_coverage':coverage}
        return dict(points_c1_native=p1,points_c2_native=p2,match_score=scores,
                    bidirectional_error_px=cycles,valid_mask=valid,rejection_reason=reasons,
                    provenance=provenance,status='matches_available' if valid.sum()>=5 else 'insufficient_cross_view_matches')


def match_pair(pair_id, images, masks, config=None, backend='roma_v2', matcher=None):
    return (matcher or Matcher(backend,config)).match_pair(pair_id,images,masks)


def save_match(result, directory, images=None, overlay=False):
    directory=output_path(directory);directory.mkdir(parents=True,exist_ok=True)
    pair_id=result['provenance']['pair_id']; arrays_path=directory/(pair_id+'.npz')
    if arrays_path.exists(): raise FileExistsError(arrays_path)
    keys=('points_c1_native','points_c2_native','match_score','bidirectional_error_px','valid_mask','rejection_reason')
    np.savez_compressed(arrays_path,**{k:result[k] for k in keys})
    metadata=result['provenance']|{'status':result['status'],'arrays_path':str(arrays_path),
              'arrays_sha256':sha256(arrays_path),'cache_identity':identity(result['provenance'])}
    if overlay and images:
        metadata['overlay_path']=str(save_overlay(result,images,directory/(pair_id+'_overlay.jpg')))
        metadata['patch_context_path']=str(save_patch_context(result,images,directory/(pair_id+'_patches.jpg')))
    json_path=directory/(pair_id+'.json');write_json(json_path,metadata)
    return metadata|{'metadata_path':str(json_path)}


def save_overlay(result, images, path):
    ims=[cv2.imread(str(images[c])) for c in ('camera1','camera2')]
    canvas=np.concatenate(ims,axis=1); width=ims[0].shape[1]
    ids=np.flatnonzero(result['valid_mask'])
    for order,k in enumerate(ids):
        color=tuple(int(v) for v in cv2.cvtColor(np.uint8([[[order*179//max(1,len(ids)),220,255]]]),cv2.COLOR_HSV2BGR)[0,0])
        a=np.rint(result['points_c1_native'][k]).astype(int);b=np.rint(result['points_c2_native'][k]).astype(int)+[width,0]
        cv2.line(canvas,tuple(a),tuple(b),color,1,cv2.LINE_AA)
        cv2.circle(canvas,tuple(a),3,color,-1);cv2.circle(canvas,tuple(b),3,color,-1)
    cv2.putText(canvas,f"{result['provenance']['matcher_name']} pair {result['provenance']['pair_id']} selected {len(ids)}",(10,24),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(canvas,f"{result['provenance']['matcher_name']} pair {result['provenance']['pair_id']} selected {len(ids)}",(10,24),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
    path=output_path(path);cv2.imwrite(str(path),canvas);return path


def save_patch_context(result, images, path):
    ims=[cv2.imread(str(images[c])) for c in ('camera1','camera2')]
    ids=np.flatnonzero(result['valid_mask'])
    ids=ids[np.linspace(0,len(ids)-1,min(12,len(ids))).astype(int)] if len(ids) else []
    rows=[]
    for k in ids:
        patches=[]
        for im,p in zip(ims,(result['points_c1_native'][k],result['points_c2_native'][k])):
            patch=cv2.getRectSubPix(im,(65,65),tuple(map(float,p)))
            cv2.drawMarker(patch,(32,32),(0,255,0),cv2.MARKER_CROSS,9,1)
            patches.append(cv2.resize(patch,(130,130)))
        rows.append(np.concatenate(patches,axis=1))
    canvas=np.concatenate(rows,axis=0) if rows else np.zeros((130,260,3),np.uint8)
    path=output_path(path);cv2.imwrite(str(path),canvas);return path


def geometry_diagnostics(result, cameras):
    """Reference-free E/H diagnostics, not an extra optimizer prior or success gate."""
    valid=result['valid_mask'];p1=result['points_c1_native'][valid];p2=result['points_c2_native'][valid]
    out={'selected':len(p1),'ransac_threshold_px':1.5,'homography_threshold_px':3.,
         'essential_inliers':0,'homography_inliers':0,'essential_positive_depth':0}
    if len(p1)<5:return out
    c1,c2=cameras['camera1'],cameras['camera2']
    K1,K2=np.asarray(c1['K'],float),np.asarray(c2['K'],float)
    n1=cv2.undistortPoints(p1[:,None].astype(float),K1,np.asarray(c1['D'],float))[:,0]
    n2=cv2.undistortPoints(p2[:,None].astype(float),K2,np.asarray(c2['D'],float))[:,0]
    focal=float(np.mean([K1[0,0],K1[1,1],K2[0,0],K2[1,1]]))
    threshold=1.5/focal;out['ransac_threshold_normalized']=threshold
    cv2.setRNGSeed(int(result['provenance']['pair_seed']))
    E,mask=cv2.findEssentialMat(n1,n2,np.eye(3),method=cv2.RANSAC,prob=.999,threshold=threshold,maxIters=5000)
    if E is not None and mask is not None:
        mask=mask.astype(np.uint8);out['essential_inliers']=int(mask.sum())
        recovered=[]
        for piece in np.asarray(E).reshape(-1,3,3):
            count,R,t,pmask=cv2.recoverPose(piece,n1,n2,np.eye(3),mask=mask.copy())
            recovered.append((int(count),R,t,pmask))
        count,R,t,pmask=max(recovered,key=lambda x:x[0]);out['essential_positive_depth']=count
        out['essential_relative_rotation']=R.tolist();out['essential_translation_unit']=t.ravel().tolist()
    if len(p1)>=4:
        # Normalize distortion first and convert to nominal focal pixel scale for diagnostic H only.
        _,hm=cv2.findHomography(n1*focal,n2*focal,cv2.RANSAC,3.,maxIters=5000,confidence=.999)
        out['homography_inliers']=int(hm.sum()) if hm is not None else 0
    return out


def run_matching(pairs, observations_manifest, config, backend, output_manifest, overlay_ids=None):
    """Run a selected list of current pairs. Existing per-pair caches validate before reuse."""
    from .bridge import ObservationStore, read_pairs
    if isinstance(pairs,(str,Path)): pairs=read_pairs(pairs)
    output_manifest=output_path(output_manifest)
    if output_manifest.exists():raise FileExistsError(output_manifest)
    output_manifest.parent.mkdir(parents=True,exist_ok=True)
    cfg=DEFAULTS | config.get('matching',config)
    store=ObservationStore(observations_manifest)
    start=time.perf_counter();matcher=None;error=None
    try:matcher=Matcher(backend,cfg)
    except (ImportError,FileNotFoundError,RuntimeError) as exc:error=f'{type(exc).__name__}: {exc}'
    rows={};overlay_ids=set(overlay_ids or [])
    if error:
        status='blocked_missing_pretrained_weights' if 'pretrained_weights' in error else 'blocked_matcher_dependency'
        result={'schema_version':2,'backend':backend,'status':status,'reason':error,
                'pairs':{p['pair_id']:{'status':status,'reason':error} for p in pairs},
                'matching_config':cfg,'observations_manifest':str(observations_manifest),
                'observations_manifest_sha256':sha256(observations_manifest)}
        result['identity']=identity(result)
        write_json(output_manifest,result);return result
    directory=output_manifest.parent/'pairs'
    for index,pair in enumerate(pairs):
        pid=pair['pair_id']; observations=store.pair(pair)
        images={c:pair['cameras'][c]['image_path'] for c in ('camera1','camera2')}
        masks={c:observations[c]['foreground_exclusion_mask'] for c in ('camera1','camera2')}
        old=directory/(pid+'.json')
        if old.exists():
            metadata=json.loads(old.read_text())
            expected={'image_sha256':[sha256(images[c]) for c in ('camera1','camera2')],
                      'current_mask_sha256':[_array_hash(masks[c]) for c in ('camera1','camera2')],
                      'inference_config':cfg,'checkpoint_sha256':matcher.provenance.get('checkpoint_sha256'),
                      'source_commit':matcher.provenance.get('source_commit'),'matcher_name':backend}
            if any(metadata.get(k)!=v for k,v in expected.items()):raise ValueError(f'Match cache identity mismatch: {pid}')
            if sha256(metadata['arrays_path'])!=metadata['arrays_sha256']:raise ValueError(f'Match arrays mutated: {pid}')
            rows[pid]=metadata|{'metadata_path':str(old)}
        else:
            result=matcher.match_pair(pid,images,masks)
            result['provenance']['geometry_diagnostics']=geometry_diagnostics(result,pair['cameras'])
            rows[pid]=save_match(result,directory,images,overlay=pid in overlay_ids)
        if index%25==0 or index+1==len(pairs):
            print(json.dumps({'backend':backend,'completed':index+1,'total':len(pairs),'pair_id':pid,
                  'selected':rows[pid]['counts']['spatially_selected'],'elapsed_s':round(time.perf_counter()-start,2)}),flush=True)
    result={'schema_version':2,'backend':backend,'status':'executed','pairs':rows,
            'matching_config':cfg,'matcher_provenance':matcher.provenance,
            'model_load_time_ms':matcher.load_time_ms,'wall_runtime_ms':1000*(time.perf_counter()-start),
            'observations_manifest':str(Path(observations_manifest).resolve()),
            'observations_manifest_sha256':sha256(observations_manifest),
            'reference_accessed':False,'pair_state_reused':False}
    result['identity']=identity(result)
    write_json(output_manifest,result);return result


def load_match(manifest, pair_id):
    if isinstance(manifest,(str,Path)):manifest=json.loads(Path(manifest).read_text())
    row=manifest['pairs'][str(pair_id)]
    if 'arrays_path' not in row:
        return {'status':row['status'],'reason':row.get('reason'),'points_c1_native':np.empty((0,2)),
                'points_c2_native':np.empty((0,2)),'match_score':np.empty(0),'valid_mask':np.empty(0,bool),
                'rejection_reason':np.empty(0,dtype='<U40'),'provenance':row}
    extra={'status','arrays_path','arrays_sha256','cache_identity','overlay_path','patch_context_path','metadata_path'}
    if identity({k:v for k,v in row.items() if k not in extra})!=row['cache_identity']:
        raise ValueError('Pair match metadata identity differs')
    if sha256(row['arrays_path'])!=row['arrays_sha256']:raise ValueError('Match cache content changed')
    with np.load(row['arrays_path'],allow_pickle=False) as data:result={k:data[k].copy() for k in data.files}
    return result|{'status':row['status'],'provenance':row}


def load_pair_matches(manifest, pair, observations=None):
    """Validate pair image identity and matcher identity before returning copied arrays.

    Pass current observations to additionally rehash masks. Without that argument,
    the matching manifest's immutable observation-manifest hash remains the mask
    provenance binding; no arbitrary nearby cache is ever discovered.
    """
    if isinstance(manifest,(str,Path)):manifest=json.loads(Path(manifest).read_text())
    result=load_match(manifest,pair['pair_id']);row=result['provenance']
    if 'arrays_path' not in row:return result
    camera_order=('camera1','camera2')
    expected_images=[pair['cameras'][c]['image_sha256'] for c in camera_order]
    if row['image_sha256']!=expected_images:raise ValueError('Current pair/match image identity differs')
    for key in ('matcher_name','source_commit','checkpoint_sha256','inference_config'):
        expected=manifest['matcher_provenance'][key]
        if row[key]!=expected:raise ValueError(f'Pair/matcher {key} differs')
    if observations is not None:
        expected_masks=[_array_hash(np.asarray(observations[c]['foreground_exclusion_mask'],bool)) for c in camera_order]
        if row['current_mask_sha256']!=expected_masks:raise ValueError('Current pair/match mask identity differs')
        if 'applied_mask_sha256' in row:
            expected_applied=[_array_hash(current_exclusion_mask(observations[c]['foreground_exclusion_mask'],row['inference_config'])) for c in camera_order]
            if row['applied_mask_sha256']!=expected_applied:raise ValueError('Applied current mask differs')
    return result
