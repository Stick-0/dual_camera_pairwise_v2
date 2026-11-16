"""Link the completed immutable experiment, evaluation and preservation evidence."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run
import csv
import json
from datetime import datetime, timezone
from runtime import sha256,write_json


def main():
    final=ROOT/'reports/final'
    manifest=json.loads((ROOT/'results/full/manifest.json').read_text())
    audit=json.loads((ROOT/'validation/full_output_contracts.json').read_text())
    preservation=json.loads((ROOT/'preservation/after_full.json').read_text())
    evaluation=json.loads((final/'evaluation.json').read_text())
    provenance=json.loads((final/'production_identity.json').read_text())
    assert manifest['completed_pairs']==2596 and audit['passed'] and evaluation['full_sequence']
    assert preservation['passed']
    assert not manifest['code_changed_during_run']
    assert provenance['observations_identity_matches_current']
    # The second evaluation repairs a provenance comparison and expands prose;
    # numerical experiment results and metrics must stay exactly the same.
    same={name:sha256(final/name)==sha256(ROOT/'reports/full'/name) for name in
        ('summary_metrics.csv','paired_comparisons.csv','bootstrap_sensitivity.csv',
         'per_observation_metrics.csv','production_reference.csv')}
    assert all(same.values()),same
    required=[ROOT/'README.md',ROOT/'configs/experiment.yaml',ROOT/'run.py',
        ROOT/'results/full/manifest.json',ROOT/'results/full/started.json',
        ROOT/'results/full/initializations.jsonl',ROOT/'results/full/pair_results.jsonl',
        ROOT/'validation/input_audit.json',ROOT/'validation/weights.json',
        ROOT/'validation/control_checks.json',ROOT/'validation/full_output_contracts.json',
        ROOT/'validation/no_cross_diagnostic.json',ROOT/'validation/mechanism_analysis.json',
        ROOT/'validation/independent_evaluation_audit_final.json',ROOT/'validation/report_structure_qa.json',
        ROOT/'preservation/before_summary.json',ROOT/'preservation/after_full.json',
        *sorted((ROOT/'src').glob('*.py')),*sorted(final.glob('*.json')),*sorted(final.glob('*.csv')),
        final/'ablation_report.html']
    primary=[r for r in csv.DictReader((final/'summary_metrics.csv').open())
        if r['gate']=='final' and r['population']=='full' and r['camera']=='pooled']
    result={'status':'completed_full_2596_pairs','created_utc':datetime.now(timezone.utc).isoformat(),
        'new_experiment_root':str(ROOT),'canonical_report':str(final/'ablation_report.html'),
        'pair_count':2596,'camera_observations':5192,'pre_eligible_pairs':evaluation['n_pre_eligible_pairs'],
        'conditions':['R0','R1','E0','E1'],'one_joint_start_per_condition':True,
        'actual_inference_wall_seconds':manifest['wall_seconds'],
        'cached_CUDA_observations_and_RoMa_matches_reused':True,
        'new_model_inference':False,'new_matching':False,'new_candidates':False,'training':False,
        'optimizer_device':'CPU','originals_preservation':preservation,
        'evaluation_revision_numerically_identical':same,
        'evaluation_revision_reason':'Canonical observation identity excludes its own identity field; expanded evidence-based analysis. No inference or metric change.',
        'primary_results':primary,'artifacts_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in required}}
    write_json(ROOT/'completion_manifest.json',result)
    print(json.dumps({'completed':True,'report':result['canonical_report'],'artifact_count':len(result['artifacts_sha256'])}))


if __name__=='__main__':main()
