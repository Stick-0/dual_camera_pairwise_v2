"""Exhaustive output-contract audit; no evaluation references or ADD consulted."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run
import json
from collections import Counter
import numpy as np
from controlled_solver import array_hash
from runtime import write_json, identity, sha256


def main():
    directory=ROOT/'results/full'
    manifest=json.loads((directory/'manifest.json').read_text())
    counts=Counter();ids=set();failures=[]
    assert manifest['completed_pairs']==2596 and manifest['pair_count']==2596
    assert not manifest['code_changed_during_run']
    for line in (directory/'pair_results.jsonl').open():
        result=json.loads(line);pid=result['pair_id'];assert pid not in ids;ids.add(pid)
        try:
            assert result['evaluation_reference_used'] is False
            assert result['inference_or_matching_rerun'] is False
            settings=result['effective_settings']
            assert settings['max_nfev']==60 and settings['max_joint_starts']==1
            assert settings['candidate_options']['max_em_iterations']==10
            assert set(result['methods'])=={'R0','R1','E0','E1'}
            prespecified=all(x['available'] and x['on_initialization']['available'] for x in result['initializations'].values())
            assert result['pre_eligible']==prespecified
            counts['pre_eligible']+=prespecified
            for family in ('R','E'):
                init=result['initializations'][family]
                off,on=(result['methods'][family+n] for n in ('0','1'))
                assert off['observation_hash']==on['observation_hash']==result['observation_hash']
                counts[family+'_init_available']+=init['available']
                if init['available']:
                    for key in ('A0','G0'):
                        assert np.array_equal(init[key],off['initialization'][key])
                        assert np.array_equal(init[key],on['initialization'][key])
                        assert array_hash(init[key])==init[key+'_hash']==off['initialization'][key+'_hash']==on['initialization'][key+'_hash']
                    assert init['initial_responsibility_hash']==off['initialization']['responsibility_hash']==on['initialization']['responsibility_hash']
                    assert on['fixed_matches']['selection_hash']==result['fixed_matches']['selection_hash']
                else:
                    assert not off['final_valid'] and not on['final_valid']
                    assert off['initializations_tried']==on['initializations_tried']==0
                for m in (off,on):
                    assert not m['fallback_used']
                    assert m['initializations_tried'] in (0,1)
                    assert len(m['trace'])<=10
                    for step in m['trace']: assert step['nfev']<=60
                    assert not m['final_valid'] or m['common_valid']
                    counts[m['method']+'_common_valid']+=m['common_valid']
                    counts[m['method']+'_final_valid']+=m['final_valid']
                assert off['background_gate']['applied'] is False
                if off['initializations_tried']:
                    assert off['joint_parameter_count']==12 and off['landmark_parameter_count']==0
                    assert off['objective_components']['background_depth']==0
                    assert off['objective_components']['background_reprojection']==0
                if on['initializations_tried']:
                    assert on['joint_parameter_count']==252 and on['landmark_parameter_count']==240
            for mono in result['shared_monocular'].values():
                for proposal in mono['traces']:
                    for step in proposal['trace']:assert step['nfev']<=60
            counts['audited_pairs']+=1
        except Exception as error:
            failures.append({'pair_id':pid,'type':type(error).__name__,'message':str(error)})
    assert len(ids)==2596 and ids=={f'{k:06d}' for k in range(2596)}
    result={'passed':not failures,'counts':dict(counts),'failures':failures,
        'result_sha256':sha256(directory/'pair_results.jsonl'),
        'all_failed_poses_retained_as_rows':True,'evaluation_reference_read':False,
        'prespecified_eligibility_rederived_without_outcomes':True}
    write_json(ROOT/'validation/full_output_contracts.json',result)
    print(json.dumps(result))
    if failures:raise AssertionError(failures)


if __name__=='__main__':main()
