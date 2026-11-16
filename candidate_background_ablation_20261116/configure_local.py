#!/usr/bin/env python3
"""Create a new absolute-path local configuration; preserve every original input."""
import argparse
import copy
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import yaml

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robotpose-root', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'configs/experiment.local.yaml')
    args = parser.parse_args()
    original = args.robotpose_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if ROOT.parent != original:
        parser.error('Deploy only this new ablation directory directly under the original Robotpose root. The frozen runner requires that layout. Do not overwrite an existing directory.')
    if not output.is_relative_to(ROOT):
        parser.error('Configuration output must remain inside this new ablation extension.')
    if output.exists():
        parser.error('Output already exists; choose a new --output filename.')
    required = ['ur5e_capture/geometry.py', 'dual_camera_pairwise_v2/pairwise/solver.py',
                'dual_camera_pairwise_v2/pairwise/candidates.py',
                'dual_camera_pairwise_v2/pairwise/preservation.py']
    missing = [str(original/p) for p in required if not (original/p).is_file()]
    if missing:
        parser.error('Original execution sources are required (the GitHub lightweight core alone is insufficient): '+', '.join(missing))
    config = yaml.safe_load((ROOT/'configs/experiment.yaml').read_text())
    config = copy.deepcopy(config)
    # Only path-bearing groups change; all numerical and method settings stay frozen.
    for group in ('source', 'inputs', 'reference'):
        for key, value in config.get(group, {}).items():
            if isinstance(value, str) and '${ROBOTPOSE_ROOT}' in value:
                config[group][key] = value.replace('${ROBOTPOSE_ROOT}', str(original))
    expected = [Path(v) for v in config['inputs'].values()]
    expected += [Path(config['reference'][k]) for k in ('production_run','production_evaluation')]
    absent = [str(p) for p in expected if not p.exists()]
    if absent:
        parser.error('Read-only original observations, match caches, and production references are required: '+', '.join(absent))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    print(output)


if __name__ == '__main__':
    main()
