"""Plot every chronological R0/R1 camera observation without dropping frames.

Only writes into this new visualization extension. Source evaluation CSV is
read-only. A frame index identifies an original paired capture, not a selected
success subset. Accepted ADD uses final_valid; rejected poses are gaps and rugs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
for variable, relative in {'MPLCONFIGDIR': 'cache/add_graphs/mpl',
                           'XDG_CACHE_HOME': 'cache/add_graphs/xdg',
                           'TMPDIR': 'cache/add_graphs/tmp'}.items():
    directory = ROOT / relative
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(directory)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

METHODS = ('R0', 'R1')
CAMERAS = ('camera1', 'camera2')
DEFAULT_SOURCE = ROOT.parent / 'candidate_background_ablation_20261116/reports/final/per_observation_metrics.csv'
EXPECTED_PAIRS = 2596
EXPECTED_PAIR_IDS = {f'{i:06d}' for i in range(EXPECTED_PAIRS)}
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.titlesize': 14, 'axes.labelsize': 12,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#9aabb4', 'axes.labelcolor': '#284654',
    'text.color': '#183542', 'xtick.color': '#49636f', 'ytick.color': '#49636f',
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'path.simplify': False, 'agg.path.chunksize': 0,
    'svg.hashsalt': 'candidate-background-all-frames-20261120',
})


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def bool_value(value):
    if str(value).lower() in ('true', '1'): return True
    if str(value).lower() in ('false', '0'): return False
    raise ValueError(f'Unknown boolean: {value!r}')


def original_metadata(path):
    stat = Path(path).stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, mode=stat.st_mode)


def read_source(source):
    records = {}
    source_rows = 0
    for row in csv.DictReader(Path(source).open(newline='')):
        source_rows += 1
        if row['method'] not in METHODS: continue
        if row['camera'] not in CAMERAS: raise ValueError('Unexpected camera ID')
        pid = row['pair_id']
        key = (row['method'], row['camera'])
        pair = records.setdefault(pid, {})
        if key in pair: raise ValueError(f'Duplicate row: {pid}/{key}')
        valid = bool_value(row['final_valid'])
        final = float(row['final_ADD_mm']) if row['final_ADD_mm'] else math.inf
        if valid and (not math.isfinite(final) or final < 0):
            raise ValueError(f'Invalid accepted ADD: {pid}/{key}')
        if valid:
            assert bool_value(row['common_valid'])
            assert final == float(row['add_before_gate_mm']) == float(row['common_ADD_mm'])
        pair[key] = dict(timestamp_s=float(row['timestamp_s']), valid=valid,
                         final_ADD_mm=final if valid else math.inf)
    if set(records) != EXPECTED_PAIR_IDS:
        raise ValueError('The full 2596-pair population is required; no subset is allowed')
    keys = {(method, camera) for method in METHODS for camera in CAMERAS}
    for pid, pair in records.items():
        if set(pair) != keys: raise ValueError(f'Incomplete pair: {pid}')
        times = [record['timestamp_s'] for record in pair.values()]
        if not all(math.isfinite(t) and t == times[0] for t in times):
            raise ValueError(f'Inconsistent original pair timestamps: {pid}')
    ids = sorted(records, key=lambda pid: (records[pid][('R0', 'camera1')]['timestamp_s'], pid))
    times = np.array([records[pid][('R0', 'camera1')]['timestamp_s'] for pid in ids])
    assert np.all(np.diff(times) >= 0)
    values, masks = {}, {}
    for method, camera in sorted(keys):
        mask = np.array([records[pid][(method, camera)]['valid'] for pid in ids], bool)
        value = np.array([records[pid][(method, camera)]['final_ADD_mm'] for pid in ids], float)
        values[(method, camera)] = np.where(mask, value, np.nan)
        masks[(method, camera)] = mask
    return ids, times, values, masks, source_rows


def setup_axis(ax, n, ymax):
    ax.set_xlim(1, n)
    ax.set_ylim(0, ymax)
    ax.set_xticks([1, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, n])
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.grid(axis='y', color='#dce5e9', linewidth=.75)
    ax.grid(axis='x', color='#edf1f3', linewidth=.55)
    ax.axhline(50, color='#8b5050', ls=(0, (5, 4)), linewidth=1.,
               label='50 mm reference', zorder=1)
    ax.set_xlabel('Frame index (chronological paired capture, 1–2,596)')
    ax.set_ylabel('ADD (mm)')
    ax.tick_params(axis='both', labelsize=10)


def add_curve(ax, x, values, valid, color, label, trace_checks):
    line, = ax.plot(x, values, color=color, linewidth=.8, alpha=.9,
                    marker='.', markersize=1.8, markeredgewidth=0,
                    label=label, zorder=3, rasterized=False)
    np.testing.assert_array_equal(line.get_xdata(), x)
    np.testing.assert_allclose(line.get_ydata(), values, rtol=0, atol=0, equal_nan=True)
    assert len(line.get_path().vertices) == len(x)
    if (~valid).any():
        # Axis-relative rug preserves the frame location while avoiding fake ADD.
        ax.scatter(x[~valid], np.full((~valid).sum(), .018), transform=ax.get_xaxis_transform(),
                   marker='x', s=25, c=color, linewidths=1.1,
                   label=label+' rejected (gap/rug)', zorder=5)
    trace_checks.append(dict(label=label, x_points=len(x), finite_y_points=int(np.isfinite(values).sum()),
        rejected_gap_points=int((~valid).sum()), path_vertices=len(line.get_path().vertices),
        all_x_and_y_equal_original=True))


def save(fig, root, name, traces, common_ylim):
    outputs = []
    for suffix in ('png', 'svg'):
        path = root / 'figures' / (name+'.'+suffix)
        if path.exists(): raise FileExistsError(path)
        kwargs = dict(dpi=180)
        if suffix == 'svg': kwargs['metadata'] = {'Date': None, 'Creator': 'Robotpose all-frame ADD visualization'}
        fig.savefig(path, **kwargs)
        outputs.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size))
    width, height = (fig.get_size_inches()*180).astype(int)
    for ax in fig.axes:
        assert tuple(ax.get_ylim()) == tuple(common_ylim)
    plt.close(fig)
    return dict(name=name, pixels=[int(width), int(height)], dpi=180,
                y_limits_mm=list(common_ylim), traces=traces, files=outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    source = args.source.resolve()
    before_hash, before_metadata = sha(source), original_metadata(source)
    ids, times, values, masks, source_rows = read_source(source)
    output_csv = ROOT / 'data/add_by_frame.csv'
    validation_path = ROOT / 'validation/add_graphs.json'
    names = ['R0_ADD_all_frames', 'R1_ADD_all_frames', 'R0_R1_ADD_comparison']
    targets = [output_csv, validation_path] + [ROOT/'figures'/(name+'.'+ext) for name in names for ext in ('png','svg')]
    for path in targets:
        if path.exists() and path != output_csv: raise FileExistsError(f'Will not overwrite: {path}')
        assert path.resolve().is_relative_to(ROOT)
        path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['frame_index', 'pair_id', 'timestamp_s']
    for method in METHODS:
        for camera in CAMERAS: fields.extend([f'{method}_{camera}_ADD_mm', f'{method}_{camera}_valid'])
    if not output_csv.exists():
        with output_csv.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for i, pid in enumerate(ids):
                row = dict(frame_index=i+1, pair_id=pid, timestamp_s=float(times[i]))
                for method in METHODS:
                    for camera in CAMERAS:
                        key=(method,camera)
                        row[f'{method}_{camera}_ADD_mm']=float(values[key][i]) if masks[key][i] else ''
                        row[f'{method}_{camera}_valid']=bool(masks[key][i])
                writer.writerow(row)
    # Round-trip the mapping and every finite value; no precision-losing formatting.
    readback=list(csv.DictReader(output_csv.open()))
    assert len(readback)==EXPECTED_PAIRS
    assert list(readback[0])==fields
    for i,row in enumerate(readback):
        assert int(row['frame_index'])==i+1 and row['pair_id']==ids[i]
        assert float(row['timestamp_s'])==times[i]
        for method in METHODS:
            for camera in CAMERAS:
                key=(method,camera);flag=bool_value(row[f'{method}_{camera}_valid'])
                assert flag==masks[key][i]
                if flag:assert float(row[f'{method}_{camera}_ADD_mm'])==values[key][i]
                else:assert row[f'{method}_{camera}_ADD_mm']==''
    maximum=max(float(np.nanmax(value)) for value in values.values())
    common_ylim=(0.,maximum*1.05)
    x=np.arange(1,len(ids)+1)
    camera_colors={'camera1':'#087e8b','camera2':'#d06a2b'}
    method_colors={'R0':'#687b90','R1':'#087e8b'}
    plots=[]
    for method,state in [('R0','OFF'),('R1','ON')]:
        fig,ax=plt.subplots(figsize=(20,6),layout='constrained')
        fig.suptitle(f'{method}  |  PnP initialization + background {state}',fontsize=19,fontweight='bold')
        ax.set_title('All 2,596 paired frames · camera-specific 7-point FK ADD · final accepted poses',fontsize=12,pad=15,loc='left')
        setup_axis(ax,len(ids),common_ylim[1]);traces=[]
        for camera in CAMERAS:
            add_curve(ax,x,values[(method,camera)],masks[(method,camera)],camera_colors[camera],camera.replace('camera','Camera '),traces)
        ax.legend(loc='lower right',bbox_to_anchor=(1.,1.013),borderaxespad=0,
                  frameon=True,facecolor='white',edgecolor='#d4dfe4',ncol=3)
        plots.append(save(fig,ROOT,method+'_ADD_all_frames',traces,common_ylim))
    fig,axes=plt.subplots(2,1,figsize=(20,10),layout='constrained',sharex=True,sharey=True)
    fig.suptitle('R0 vs R1  |  Background OFF vs ON from the same PnP initialization',fontsize=19,fontweight='bold')
    traces=[]
    for ax,camera in zip(axes,CAMERAS):
        setup_axis(ax,len(ids),common_ylim[1]);ax.set_title(camera.replace('camera','Camera ')+' · all 2,596 paired frames',loc='left',pad=10)
        for method in METHODS:
            add_curve(ax,x,values[(method,camera)],masks[(method,camera)],method_colors[method],method+(' · background OFF' if method=='R0' else ' · background ON'),traces)
        ax.legend(loc='lower right',bbox_to_anchor=(1.,1.013),borderaxespad=0,
                  frameon=True,facecolor='white',edgecolor='#d4dfe4',ncol=3)
    axes[0].set_xlabel('');axes[0].tick_params(labelbottom=True)
    plots.append(save(fig,ROOT,'R0_R1_ADD_comparison',traces,common_ylim))
    summaries=[]
    for method in METHODS:
        for camera in CAMERAS:
            v=values[(method,camera)];finite=v[np.isfinite(v)]
            summaries.append(dict(method=method,camera=camera,total=len(v),valid=len(finite),failed=int(np.isnan(v).sum()),
                mean_ADD_mm=float(finite.mean()) if len(finite) else None,median_ADD_mm=float(np.median(finite)) if len(finite) else None,
                min_ADD_mm=float(finite.min()) if len(finite) else None,max_ADD_mm=float(finite.max()) if len(finite) else None,
                ADD50=float(np.sum(v<50)/len(v)),max_frame_index=int(np.nanargmax(v)+1),max_pair_id=ids[int(np.nanargmax(v))]))
    assert sha(source)==before_hash and original_metadata(source)==before_metadata
    validation=dict(passed=True,source_csv=str(source),source_sha256=before_hash,source_metadata=before_metadata,
        source_preserved_content_and_metadata=True,source_rows=source_rows,selected_camera_rows=len(ids)*4,
        n_pairs=len(ids),methods=list(METHODS),cameras=list(CAMERAS),
        sorting='Original pair timestamp_s ascending, then pair_id stable lexical tie break',
        x_axis='1-based frame index in unchanged full pair sequence',pair_id_first=ids[0],pair_id_last=ids[-1],
        pair_ids_match_000000_to_002595=(ids==[f'{i:06d}' for i in range(EXPECTED_PAIRS)]),
        timestamp_span_s=float(times[-1]-times[0]),original_timestamp_deltas_s=dict(min=float(np.diff(times).min()),median=float(np.median(np.diff(times))),max=float(np.diff(times).max())),
        csv_roundtrip_all_values_exact=True,output_csv=str(output_csv),output_csv_sha256=sha(output_csv),
        shared_y_limits_mm=list(common_ylim),global_maximum_ADD_mm=maximum,clipping=False,smoothing=False,downsampling=False,
        matplotlib_path_simplify=False,matplotlib_agg_path_chunksize=0,all_original_samples_have_markers=True,
        failed_pose_policy='NaN line gaps and x-axis-relative x-mark rugs; never zero-filled',
        graph_files=plots,per_camera_statistics=summaries,
        code_sha256=sha(__file__),source_inference_or_solver_rerun=False)
    with validation_path.open('x') as stream:json.dump(validation,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(passed=True,n_pairs=len(ids),selected_camera_rows=len(ids)*4,
        shared_y_limits_mm=common_ylim,failed_camera_rows=sum(r['failed'] for r in summaries),
        csv=str(output_csv),validation=str(validation_path),plots=[r['name'] for r in plots]),ensure_ascii=False))


if __name__=='__main__':main()
