"""Required control checks using prespecified real pairs and fixed stored arrays."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run
import argparse
import copy
import json
import numpy as np
import yaml
from controlled_solver import solve_from_fixed_initial, generate_initializations, _context
from pairwise.bridge import ObservationStore, read_pairs
from pairwise.matching import load_pair_matches
from dual_camera_pairwise_v2.pairwise import solver as original
from dual_camera_pairwise_v2.pairwise.geometry import pose_vector, pose
from runtime import write_json, identity


def stable_output(value):
    """Ignore instrumentation clock differences, compare numerical/control results."""
    if isinstance(value, dict):
        return {k: stable_output(v) for k,v in value.items()
                if k not in ('timing','elapsed_ms','generation_ms','total_ms','optimization_ms')}
    if isinstance(value, list): return [stable_output(v) for v in value]
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT/'results/smoke')
    parser.add_argument('--output', type=Path, default=ROOT/'validation/control_checks.json')
    args=parser.parse_args()
    cfg = yaml.safe_load((ROOT/'configs/experiment.yaml').read_text())
    run.prohibit_reference(cfg['inputs']['reference'])
    safe_cfg = run.solver_config(cfg)
    store = ObservationStore(cfg['inputs']['observations'])
    manifest = json.loads(Path(cfg['inputs']['matches']).read_text())
    pairs = {p['pair_id']: p for p in read_pairs(cfg['inputs']['pairs'])}
    checks = []
    for pid in cfg['control']['smoke_pair_ids']:
        pair = pairs[pid]; obs = store.pair(pair)
        matches = load_pair_matches(manifest, pair, obs)
        saved = json.loads((args.run_dir/'pairs'/f'{pid}.json').read_text())
        for family in ('R','E'):
            init = saved['initializations'][family]
            if not init['available']:
                checks.append({'pair_id':pid,'family':family,'test':'available_for_fixed_init_checks','passed':False,'reason':init['reason']})
                continue
            off = saved['methods'][family+'0']; on = saved['methods'][family+'1']
            for key in ('A0','G0'):
                assert np.array_equal(off['initialization'][key], on['initialization'][key])
            assert off['initialization']['responsibility_hash'] == on['initialization']['responsibility_hash'] == init['initial_responsibility_hash']
            assert off['observation_hash'] == on['observation_hash'] == init['observation_hash']
            base = solve_from_fixed_initial(pair, obs, init, False, matches, safe_cfg)
            # Deliberately malformed, impossible background data and nonfinite P0
            # must not even be read by OFF's main solve.
            changed_cfg = copy.deepcopy(safe_cfg)
            changed_cfg['reference'] = {'T_C_B': [[9999]], 'path': '/must/not/read'}
            changed = solve_from_fixed_initial(pair, obs, init, False,
                {'status':'arbitrary', 'points_c1_native':'malformed', 'valid_mask':None},
                changed_cfg, P0=np.full((3,3), np.nan))
            assert identity(stable_output(base)) == identity(stable_output(changed))
            assert base['joint_parameter_count'] == 12 and base['landmark_parameter_count'] == 0
            assert not base['background_gate']['applied']
            assert base['objective_components']['background_reprojection'] == 0
            assert base['objective_components']['background_depth'] == 0
            context = _context(pair, obs, safe_cfg)
            x = np.r_[pose_vector(np.asarray(init['A0'])), pose_vector(np.asarray(init['G0']))]
            fun, pattern, states, _, _ = original._joint_functions(context['data'],
                np.empty((0,2)), np.empty((0,2)), context['cameras'], context['cfg'], x)
            expected = []
            depths = []
            for cid,T in (('camera1',pose(x[:6])),('camera2',pose(x[6:12])@pose(x[:6]))):
                residual, depth = original._robot_residual(T,context['data'][cid],states[cid])
                expected.extend(residual); depths.extend(depth)
            expected = np.r_[expected,original._depth_residual(depths,context['cfg'])]
            assert np.array_equal(fun(x), expected)
            assert pattern.shape == (len(expected),12)
            assert on['fixed_matches']['selection_hash'] == saved['fixed_matches']['selection_hash']
            checks.append({'pair_id':pid,'family':family,'test':'shared_initial_pose_observations_responsibility_and_OFF_invariance',
                'passed':True,'initial_hash':init['pose_array_hash'],'responsibility_hash':init['initial_responsibility_hash'],
                'OFF_result_hash_without_timing':identity(stable_output(base)),
                'OFF_parameter_count':12,'OFF_residual_rows':len(expected),
                'OFF_residual_exactly_robot_candidates_plus_depth':True})
        if saved['initializations']['R']['available'] and saved['initializations']['E']['available']:
            assert saved['methods']['R1']['fixed_matches']['selection_hash'] == saved['methods']['E1']['fixed_matches']['selection_hash']
    # Matching failure must still permit R0; E failure must remain explicit.
    pair = pairs[cfg['control']['smoke_pair_ids'][0]]; obs = store.pair(pair)
    no_bg = generate_initializations(pair, obs, None, safe_cfg)
    assert no_bg['initializations']['R']['available']
    assert not no_bg['initializations']['E']['available']
    off = solve_from_fixed_initial(pair, obs, no_bg['initializations']['R'], False, None, safe_cfg)
    assert off['initializations_tried'] == 1 and off['poses']['camera1'] is not None
    checks.append({'test':'missing_matching_keeps_R0_and_explicit_E_failure','passed':True,'R0_status':off['status']})
    try:
        Path(cfg['inputs']['reference']).read_text()
    except PermissionError:
        checks.append({'test':'reference_read_denied_during_inference','passed':True})
    else: raise AssertionError('Reference read should be denied')
    write_json(args.output,{'passed':all(c['passed'] for c in checks),
        'checks':checks,'prespecified_pairs':cfg['control']['smoke_pair_ids']})
    print(json.dumps({'passed':all(c['passed'] for c in checks),'checks':len(checks)}))


if __name__ == '__main__': main()
