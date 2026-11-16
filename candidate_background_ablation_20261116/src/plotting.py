"""Standalone scientific figures and real-image error examples for the ablation."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from evaluate import METHODS, CAMERAS, CONTRASTS, THRESHOLDS, project, pose_array, write_json, sha

COLORS={'R0':'#64748b','R1':'#087e8b','E0':'#d38b14','E1':'#af3d6f'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})


def save_figure(fig, out, name):
    fig.savefig(out/(name+'.png'),dpi=150,bbox_inches='tight')
    fig.savefig(out/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)


def temporal_correlations(arrays):
    dt=float(np.median(np.diff(arrays['timestamps']))) if len(arrays['timestamps'])>1 else 1.
    errors=np.where(arrays['valid']['final'],arrays['add'],np.inf)
    records=[];series={}
    for label,before,after in CONTRASTS:
        a=errors[:,METHODS.index(before)];b=errors[:,METHODS.index(after)]
        valid=np.isfinite(a)&np.isfinite(b)
        delta=np.where(valid,b,0)-np.where(valid,a,0)
        y=np.divide(delta.sum(1),valid.sum(1),out=np.full(len(errors),np.nan),where=valid.sum(1)>0)
        series[label]=y
        for seconds in (.5,1.,2.,5.,10.):
            lag=max(1,int(round(seconds/dt)))
            ok=np.isfinite(y[:-lag])&np.isfinite(y[lag:]) if lag<len(y) else np.zeros(0,bool)
            n=int(ok.sum());corr=None;actual=None
            if n>=3:
                x,z=y[:-lag][ok],y[lag:][ok]
                if np.std(x)>0 and np.std(z)>0:corr=float(np.corrcoef(x,z)[0,1])
                actual=float(np.median(arrays['timestamps'][lag:][ok]-arrays['timestamps'][:-lag][ok]))
            records.append(dict(contrast=label,requested_lag_s=seconds,index_lag=lag,
                actual_median_lag_s=actual,n_pair_pairs=n,correlation=corr))
    return records,series


def build_plots(out, arrays, evaluated, results, pairs, refs):
    out=Path(out);figdir=out/'figures';figdir.mkdir()
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    for row,gate in enumerate(('final','common')):
        for col,pop in enumerate(('full','pre_eligible')):
            ax=axes[row,col]
            for method in METHODS:
                curve=evaluated['curves'][f'{gate}/{pop}/pooled'][method]
                ax.plot(THRESHOLDS,100*np.array(curve['ADD']),label=method,color=COLORS[method],lw=2)
            n=len(arrays['ids'])*2 if pop=='full' else int(arrays['eligible'].sum())*2
            ax.set_title(f'{gate.capitalize()} gate / {pop} / denominator={n}')
            ax.axvline(50,color='#cbd5e1',ls='--',lw=.8);ax.grid(alpha=.2);ax.set_ylim(0,102)
            ax.set_xlabel('ADD threshold (mm)');ax.set_ylabel('Accepted poses below threshold (%)')
    axes[0,0].legend(ncol=4,loc='lower right')
    fig.tight_layout();save_figure(fig,figdir,'add_curves')

    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    primary=[r for r in evaluated['paired'] if r['gate']=='final' and r['population']=='full' and r['camera']=='pooled' and not r['contrast'].startswith('(')]
    for ax,metric,title,scale in zip(axes,('paired_ADD_delta_mm','ADD50_delta','ADD_AUC_delta'),
            ('Joint-valid ADD difference (mm)','All attempts: ADD50 difference (pp)','All attempts: AUC difference (pp)'),(1,100,100)):
        ax.axvline(0,color='#94a3b8',lw=1)
        for i,r in enumerate(primary):
            value=r[metric];lo=r.get(metric+'_ci_low');hi=r.get(metric+'_ci_high')
            if value is None:continue
            ax.scatter([value*scale],[i],c=COLORS[r['after']],s=42,zorder=3)
            if lo is not None and hi is not None:ax.plot([lo*scale,hi*scale],[i,i],c=COLORS[r['after']],lw=2)
        ax.set_yticks(range(len(primary)),[r['contrast'] for r in primary]);ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_title(title)
        ax.set_xlabel('Point estimate and 95% time-block bootstrap CI')
    fig.tight_layout();save_figure(fig,figdir,'paired_effects')
    correlations,series=temporal_correlations(arrays)
    write_json(out/'temporal_correlation.json',dict(method='Original ordered pair lag; invalid values remain missing, never compressed.',records=correlations))
    fig,axes=plt.subplots(4,1,figsize=(12,9),sharex=True)
    t=arrays['timestamps']-arrays['timestamps'][0]
    for ax,(label,y) in zip(axes,series.items()):
        ax.plot(t,y,lw=.7,color='#087e8b');ax.axhline(0,c='#64748b',lw=.6);ax.set_ylabel(label+'\nADD delta mm');ax.grid(alpha=.2)
    axes[-1].set_xlabel('Original pair timestamp relative to first pair (seconds)')
    fig.tight_layout();save_figure(fig,figdir,'temporal_effects')

    meta=json.loads((out/'evaluation.json').read_text());paths=meta['paths'];candidate_path=paths.get('candidates')
    if not candidate_path and paths.get('observations'):candidate_path=str(Path(paths['observations']).parent/'candidates.json')
    candidates=json.loads(Path(candidate_path).read_text()) if candidate_path and Path(candidate_path).exists() else None
    source_hashes={str(candidate_path):sha(candidate_path)} if candidates else {}
    examples=[]
    for label,before,after in CONTRASTS:
        a=METHODS.index(before);b=METHODS.index(after)
        common=arrays['valid']['final'][:,a].all(1)&arrays['valid']['final'][:,b].all(1)
        delta=arrays['add'][:,b].mean(1)-arrays['add'][:,a].mean(1)
        for direction in ('improved','worsened'):
            indices=[i for i in np.flatnonzero(common) if np.isfinite(delta[i]) and (delta[i]<0 if direction=='improved' else delta[i]>0)]
            if not indices:
                examples.append(dict(contrast=label,direction=direction,available=False,reason='No pair with both cameras valid and requested signed difference'))
                continue
            index=sorted(indices,key=lambda i:((delta[i] if direction=='improved' else -delta[i]),arrays['ids'][i]))[0]
            pid=arrays['ids'][index];result=results[pid];name=f'example_{label}_{direction}_{pid}'
            fig,axes=plt.subplots(2,3,figsize=(15,7.5))
            for row,camera in enumerate(CAMERAS):
                cam=pairs[pid]['cameras'][camera];reference=np.asarray(refs[pid]['cameras'][camera]['projected_keypoints_uv'])
                image=plt.imread(cam['image_path']);source_hashes[cam['image_path']]=sha(cam['image_path'])
                raw=result['methods'][before].get('raw_inputs',{}).get(camera,{})
                mean=np.asarray(raw.get('keypoints_uv',[]),float);rawvalid=np.asarray(raw.get('valid_keypoints',[False]*7),bool)
                candidate=candidates['frames'].get(cam.get('observation_key',camera+'/'+pid),{}) if candidates else {}
                for col,method in enumerate((None,before,after)):
                    ax=axes[row,col];ax.imshow(image);ax.axis('off');ax.set_xlim(0,cam.get('width',640));ax.set_ylim(cam.get('height',480),0)
                    ax.plot(reference[:,0],reference[:,1],color='#50e291',ls=':',lw=1,alpha=.8)
                    ax.scatter(reference[:,0],reference[:,1],marker='o',facecolors='none',edgecolors='#50e291',s=38,lw=1)
                    if method is None:
                        for hyp in candidate.get('hypotheses',[]):
                            uv=np.asarray(hyp['uv']);ax.scatter(uv[:,0],uv[:,1],s=8,c='#dddddd',alpha=.65)
                        if mean.shape==(7,2):ax.scatter(mean[rawvalid,0],mean[rawvalid,1],marker='+',s=55,c='#ffad32',lw=1.5)
                        ax.set_title(camera+' / unchanged raw candidates + mean')
                    else:
                        m=result['methods'][method];selected=np.asarray(m.get('candidate_selected_uv',{}).get(camera,[]),float)
                        valid=np.asarray(m.get('candidate_selected_valid',{}).get(camera,[False]*7),bool)
                        if selected.shape==(7,2):ax.scatter(selected[valid,0],selected[valid,1],marker='s',facecolors='none',edgecolors='#f3dd58',s=45,lw=1)
                        uv=project(pose_array(m.get('poses',{}).get(camera)),np.asarray(cam['keypoints_base_m']),cam)
                        ax.plot(uv[:,0],uv[:,1],c='#37d6ff',lw=1.2);ax.scatter(uv[:,0],uv[:,1],marker='x',s=40,c='#37d6ff',lw=1.3)
                        ax.set_title(f'{camera} / {method} / ADD {arrays["add"][index,METHODS.index(method),row]:.2f} mm')
            fig.suptitle(f'{label} / {direction} / pair {pid} / mean delta {delta[index]:+.2f} mm',fontsize=14)
            legend=[Line2D([],[],marker='o',ls=':',c='#50b978',markerfacecolor='none',label='Annotation reference'),
                    Line2D([],[],marker='+',ls='',c='#d48212',label='Raw voting mean'),
                    Line2D([],[],marker='s',ls='',c='#9b8500',markerfacecolor='none',label='Posterior MAP candidate'),
                    Line2D([],[],marker='x',ls='-',c='#008cbb',label='Estimated pose projection')]
            fig.legend(handles=legend,loc='lower center',ncol=4);fig.tight_layout(rect=(0,.04,1,.96))
            save_figure(fig,figdir,name)
            examples.append(dict(contrast=label,direction=direction,available=True,pair_id=pid,delta_ADD_mm=float(delta[index]),
                before=before,after=after,before_ADD_mm=arrays['add'][index,a].tolist(),after_ADD_mm=arrays['add'][index,b].tolist(),
                image='figures/'+name+'.png',selection_rule='Within full/final both-camera common-valid pairs, largest signed pair-mean ADD improvement/worsening; pair_id tie break. Evaluation-only visualization, not tuning.'))
    write_json(out/'examples.json',dict(examples=examples,input_sha256=source_hashes,input_hashes_unchanged=all(sha(p)==d for p,d in source_hashes.items())))
