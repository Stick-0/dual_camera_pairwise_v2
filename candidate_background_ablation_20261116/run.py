#!/usr/bin/env python3
"""Immutable-input, experiment-local controlled ablation entry point."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT.parent / 'dual_camera_pairwise_v2'))
from runtime import configure_runtime, identity, jsonable, sha256, write_json, output_path
configure_runtime()

import argparse
import concurrent.futures as futures
import copy
import json
import multiprocessing as mp
import platform
import time
import traceback
import yaml


def prohibit_reference(path):
    forbidden = str(Path(path).resolve())
    def hook(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            name = str(Path(os.fsdecode(args[0])).resolve())
            if name == forbidden:
                raise PermissionError('Evaluation reference is forbidden during estimation')
    sys.addaudithook(hook)


def source_manifest():
    paths = [ROOT / 'run.py', *[ROOT/'src'/name for name in ('runtime.py','controlled_solver.py')],
             *sorted((ROOT.parent / 'dual_camera_pairwise_v2/pairwise').glob('*.py')),
             ROOT.parent / 'ur5e_capture/geometry.py']
    return {str(p): sha256(p) for p in paths}


def solver_config(config):
    return {key: copy.deepcopy(config[key]) for key in
            ('schema_version', 'seed', 'estimation', 'observations', 'matching',
             'geometry', 'candidates', 'solver', 'control') if key in config}


def solve_job(job):
    from controlled_solver import solve_pair_controlled
    pair, observations, matches, cfg, match_error = job
    begin = time.perf_counter()
    try:
        result = solve_pair_controlled(pair, observations, matches, cfg)
    except Exception as error:
        result = {'pair_id': pair['pair_id'], 'pre_eligible': False,
                  'initializations': {}, 'methods': {method: {
                      'poses': {}, 'common_valid': False, 'final_valid': False,
                      'failure_stage': 'unhandled_adapter_exception',
                      'failure_reason': f'{type(error).__name__}: {error}'}
                      for method in ('R0', 'R1', 'E0', 'E1')},
                  'exception_traceback': traceback.format_exc()}
    result['timestamp_s'] = sum(c['timestamp_s'] for c in pair['cameras'].values()) / 2
    result['match_loading_failure'] = match_error
    result['worker_wall_seconds'] = time.perf_counter() - begin
    return jsonable(result)


def infer(config, config_path, output, pair_ids, workers):
    from pairwise.bridge import ObservationStore, read_pairs
    from pairwise.matching import load_pair_matches
    import numpy as np
    import scipy
    import cv2
    output = output_path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'pairs').mkdir()
    prohibit_reference(config['inputs']['reference'])
    start = time.time()
    pairs = read_pairs(config['inputs']['pairs'])
    if pair_ids:
        ids = set(pair_ids)
        pairs = [p for p in pairs if p['pair_id'] in ids]
        if len(pairs) != len(ids): raise ValueError('Missing requested pair ID')
    elif len(pairs) != 2596:
        raise ValueError('Full run must contain exactly 2596 pairs')
    store = ObservationStore(config['inputs']['observations'])
    matches_manifest = json.loads(Path(config['inputs']['matches']).read_text())
    if identity({k:v for k,v in matches_manifest.items() if k != 'identity'}) != matches_manifest['identity']:
        raise ValueError('Match manifest identity mismatch')
    if matches_manifest['observations_manifest_sha256'] != sha256(config['inputs']['observations']):
        raise ValueError('Match/observation binding mismatch')
    cfg = solver_config(config)
    begin_manifest = {'config': config, 'config_path': str(config_path),
        'config_sha256': sha256(config_path), 'source_sha256': source_manifest(),
        'input_sha256': {k: sha256(config['inputs'][k]) for k in ('pairs','observations','matches')},
        'reference_read_in_inference': False, 'reference_read_guard': True,
        'candidate_provenance': store.candidate_provenance,
        'pair_ids': [p['pair_id'] for p in pairs], 'pair_count': len(pairs),
        'workers': workers, 'started_unix_s': start,
        'environment': {'python': sys.version, 'executable': sys.executable,
            'platform': list(os.uname()), 'numpy': np.__version__,
            'scipy': scipy.__version__, 'opencv': cv2.__version__,
            'optimizer': 'CPU SciPy sparse LSMR; immutable cached CUDA observations/matches',
            'BLAS_threads_per_worker': 1, 'write_guard_root': str(ROOT)}}
    write_json(output / 'started.json', begin_manifest)
    write_json(output / 'resolved_config.json', config)
    def jobs():
        for pair in pairs:
            obs = store.pair(pair)
            error = None
            try: matches = load_pair_matches(matches_manifest, pair, obs)
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
                matches = {'status': 'match_loading_failure', 'reason': error}
            yield pair, obs, matches, cfg, error
    completed = {}; counts = {m: 0 for m in ('R0','R1','E0','E1')}
    iterator = iter(jobs())
    with futures.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as executor:
        pending = set()
        for _ in range(2 * workers):
            job = next(iterator, None)
            if job is not None: pending.add(executor.submit(solve_job, job))
        while pending:
            ready, pending = futures.wait(pending, return_when=futures.FIRST_COMPLETED)
            for future in ready:
                result = future.result(); pid = result['pair_id']
                write_json(output / 'pairs' / f'{pid}.json', result)
                completed[pid] = result
                for m in counts: counts[m] += bool(result['methods'][m].get('final_valid', False))
                if len(completed) % 25 == 0 or len(completed) == len(pairs):
                    print(json.dumps({'completed': len(completed), 'total': len(pairs),
                        'elapsed_s': round(time.time()-start, 2), 'valid_pairs': counts}), flush=True)
                job = next(iterator, None)
                if job is not None: pending.add(executor.submit(solve_job, job))
    with (output / 'pair_results.jsonl').open('x') as results_file, (output / 'initializations.jsonl').open('x') as inits_file:
        for p in pairs:
            result = completed[p['pair_id']]
            results_file.write(json.dumps(result, allow_nan=False) + '\n')
            inits_file.write(json.dumps({'pair_id': p['pair_id'],
                'pre_eligible': result.get('pre_eligible', False),
                'initializations': result.get('initializations', {})}, allow_nan=False) + '\n')
    end_sources = source_manifest()
    changes = [p for p,h in begin_manifest['source_sha256'].items() if end_sources.get(p) != h]
    manifest = {**begin_manifest, 'completed_pairs': len(completed),
        'ended_unix_s': time.time(), 'wall_seconds': time.time()-start,
        'final_valid_pairs': counts, 'code_changed_during_run': changes,
        'results_sha256': sha256(output / 'pair_results.jsonl')}
    write_json(output / 'manifest.json', manifest)
    if changes: raise RuntimeError(f'Source changed during run: {changes}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['smoke', 'full', 'evaluate'])
    parser.add_argument('--config', type=Path, default=ROOT/'configs/experiment.yaml')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run-dir', type=Path, default=ROOT/'results/full')
    parser.add_argument('--workers', type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.command == 'evaluate':
        from evaluate import evaluate_run
        value = evaluate_run(args.run_dir, args.config, args.output or ROOT/'reports/full')
        print(json.dumps(jsonable(value), ensure_ascii=False))
    else:
        ids = config['control']['smoke_pair_ids'] if args.command == 'smoke' else None
        value = infer(config, args.config, args.output or ROOT/'results'/args.command,
                      ids, args.workers or config['control']['workers'])
        print(json.dumps({'completed_pairs': value['completed_pairs'], 'wall_seconds': value['wall_seconds']}))


if __name__ == '__main__': main()
