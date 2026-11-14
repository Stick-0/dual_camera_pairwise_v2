#!/usr/bin/env python3
"""Offline current-pair inference, RGB matching, refinement and evaluation."""
import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Avoid nested BLAS thread pools when independent pairs use CPU worker processes.
# This changes only this process environment, never the installed packages.
for _name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_name]='1'

from pairwise.runtime import EXT_ROOT, REPO_ROOT, configure_runtime, identity, jsonable, output_path, sha256, write_json


def _read_matching(path):
    if path is None:return None
    value=json.loads(Path(path).read_text())
    if value.get('identity') and identity({k:v for k,v in value.items() if k!='identity'})!=value['identity']:
        raise ValueError('Matches manifest identity mismatch')
    return value


def _match_for_pair(pair,manifest,observations):
    if manifest is None:return None
    from pairwise.matching import load_pair_matches
    return load_pair_matches(manifest,pair,observations)


def _solve_job(item):
    pair,observations,matched,config,methods=item
    from pairwise.solver import solve_pair
    started=time.perf_counter()
    result=solve_pair(pair['pair_id'],observations,pair['cameras'],matched,config,methods)
    robot_source=sum(o.get('timing_s',{}).get('model_preprocessing',0.) for o in observations.values())*1000
    voting_source=sum(o.get('timing_s',{}).get('voting',0.) for o in observations.values())*1000
    duration=(time.perf_counter()-started)*1000
    matcher_ms=None
    if matched:
        matcher_ms=matched.get('matching_runtime_ms',matched.get('provenance',{}).get('matching_runtime_ms'))
    return {'schema_version':2,'pair_id':pair['pair_id'],'split':pair['split'],'methods':result,
        'observations':{cid:{key:o[key] for key in ('keypoints_uv','valid_keypoints','keypoints_base_m','hypotheses') if key in o} for cid,o in observations.items()},
        'timing':{'robot_forward_ms':0.,'voting_ms':0.,'robot_observations_reused':True,
            'source_robot_forward_ms':robot_source,'source_voting_ms':voting_source,
            'matcher_ms':matcher_ms,'refinement_wall_ms':duration,
            'total_ms':duration+(matcher_ms or 0.),'total_note':'Current matching plus all requested solvers; shared frozen-robot inference reused, recorded separately'},
        'matching':{k:matched.get(k) for k in ('status','counts','coverage','provenance','matching_runtime_ms') if k in matched} if matched else None,
        'config_identity':identity(config)}


def refine(args,config):
    source_paths=[EXT_ROOT/'run.py',*sorted((EXT_ROOT/'pairwise').glob('*.py'))]
    source_hashes_at_start={str(path.relative_to(EXT_ROOT)):sha256(path) for path in source_paths}
    from pairwise.bridge import read_pairs,ObservationStore
    pairs=read_pairs(args.pairs,args.split,args.limit,args.pair_ids)
    if not pairs:raise ValueError('No pairs selected')
    store=ObservationStore(args.observations)
    if store.metadata['pairs_sha256']!=sha256(args.pairs):raise ValueError('Pair manifest and observations differ')
    matches=_read_matching(args.matches)
    if matches is not None:
        from pairwise.matching import DEFAULTS as MATCHING_DEFAULTS
        expected_matching=MATCHING_DEFAULTS | config.get('matching',{})
        if matches.get('matching_config')!=expected_matching:
            raise ValueError('Matching cache settings differ from the selected config')
        if matches.get('observations_manifest_sha256')!=sha256(args.observations):
            raise ValueError('Matching cache uses a different observations manifest')
    methods=config['methods'] if args.methods=='all' else args.methods.split(',')
    folder=output_path(args.output);folder.mkdir(parents=True,exist_ok=False)
    (folder/'results').mkdir()
    write_json(folder/'resolved_config.json',config)
    manifest={'schema_version':2,'config':str(args.config),'config_identity':identity(config),
        'pairs_manifest':str(args.pairs),'pairs_sha256':sha256(args.pairs),
        'observations_manifest':str(args.observations),'observations_identity':store.metadata['identity'],
        'matches_manifest':str(args.matches) if args.matches else None,
        'matches_sha256':sha256(args.matches) if args.matches else None,
        'methods':methods,'split':args.split,'workers':args.workers,'results':[],
        'reference_paths_read':False,'previous_pose_fields_used':False,'current_pair_only':True,
        'source_code_sha256_at_start':source_hashes_at_start,
        'started_unix_s':time.time(),'completed':False}
    def jobs():
        for pair in pairs:
            observations=store.pair(pair)
            yield pair,observations,_match_for_pair(pair,matches,observations),config,methods
    executor=None
    if args.workers>1:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        executor=ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork'))
        results=executor.map(_solve_job,jobs(),chunksize=1)
    else:results=map(_solve_job,jobs())
    started=time.monotonic()
    try:
        for index,result in enumerate(results):
            path=folder/'results'/(result['pair_id']+'.json');write_json(path,result)
            manifest['results'].append({'pair_id':result['pair_id'],'split':result['split'],'result_path':str(path),'sha256':sha256(path)})
            if index%10==0 or index+1==len(pairs):
                statuses={method:value['status'] for method,value in result['methods'].items()}
                print(f'[{index+1}/{len(pairs)}] {result["pair_id"]} {statuses} {time.monotonic()-started:.1f}s',file=sys.stderr,flush=True)
        manifest['completed']=True
    except BaseException as exc:
        manifest['error']=str(exc);manifest['traceback']=traceback.format_exc();raise
    finally:
        if executor:executor.shutdown(wait=True,cancel_futures=True)
        manifest['wall_time_s']=time.monotonic()-started
        manifest['source_code_sha256']={str(path.relative_to(EXT_ROOT)):sha256(path) for path in source_paths}
        manifest['source_code_changed_during_run']=manifest['source_code_sha256']!=source_hashes_at_start
        write_json(folder/'manifest.json',manifest)
    return {'run':str(folder),'pairs':len(manifest['results']),'completed':manifest['completed'],'wall_time_s':manifest['wall_time_s']}



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    default_config=EXT_ROOT/'configs/ur_dual_2596_pairwise.yaml'
    def option(p,name,required=True): p.add_argument('--'+name,required=required)
    def config_option(p): p.add_argument('--config',default=str(default_config))
    def selection(p):
        p.add_argument('--split',choices=['all'],default='all')
        p.add_argument('--limit',type=int)
        p.add_argument('--pair-ids',help='Comma-separated exact IDs')
    p=commands.add_parser('prepare-reference');option(p,'pairs');option(p,'data-root');option(p,'output')
    p=commands.add_parser('match-pairs');option(p,'pairs');option(p,'observations');config_option(p);option(p,'backend');selection(p);option(p,'output')
    p=commands.add_parser('refine-pairs');option(p,'pairs');option(p,'observations');option(p,'matches',False);config_option(p)
    p.add_argument('--methods',default='all');selection(p);option(p,'output');p.add_argument('--workers',type=int,default=1)
    p=commands.add_parser('evaluate-pairs');option(p,'pairs');option(p,'reference-manifest');option(p,'run');option(p,'output');p.add_argument('--split',choices=['all'],default='all')
    p=commands.add_parser('test');p.add_argument('--output',default=None)
    args=parser.parse_args()
    for name in ('data_root','config','pairs','observations','matches','output','reference_manifest','run'):
        if getattr(args,name,None):setattr(args,name,Path(getattr(args,name)).expanduser().resolve())
    if getattr(args,'pair_ids',None):args.pair_ids=args.pair_ids.split(',')
    if getattr(args,'limit',None) is not None and args.limit<=0:parser.error('--limit must be positive')
    if getattr(args,'workers',1)<1:parser.error('--workers must be positive')
    configure_runtime()
    from pairwise.bridge import load_config,read_pairs
    config=load_config(args.config) if getattr(args,'config',None) else None
    if args.command=='prepare-reference':
        from pairwise.evaluation import prepare_reference
        result=prepare_reference(read_pairs(args.pairs),args.data_root,args.output)
    elif args.command=='match-pairs':
        from pairwise.matching import run_matching
        result=run_matching(read_pairs(args.pairs,args.split,args.limit,args.pair_ids),args.observations,config,args.backend,args.output)
    elif args.command=='refine-pairs':result=refine(args,config)
    elif args.command=='evaluate-pairs':
        from pairwise.evaluation import evaluate_pairs
        result=evaluate_pairs(read_pairs(args.pairs),args.reference_manifest,args.run,args.output,args.split)
    else:
        import unittest,io
        from pairwise.runtime import new_directory
        folder=output_path(args.output) if args.output else new_directory('validation','tests')
        folder.mkdir(parents=True,exist_ok=True)
        stream=io.StringIO();test_result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.discover(str(EXT_ROOT/'tests')))
        (folder/'unittest.log').write_text(stream.getvalue());print(stream.getvalue(),file=sys.stderr)
        result={'passed':test_result.wasSuccessful(),'tests':test_result.testsRun,'failures':len(test_result.failures),'errors':len(test_result.errors),'directory':str(folder)}
        write_json(folder/'summary.json',result)
        if not test_result.wasSuccessful():print(json.dumps(result));return 1
    print(json.dumps(jsonable(result),indent=2,ensure_ascii=False));return 0


if __name__=='__main__':
    sys.exit(main())
