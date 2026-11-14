"""GPU matcher CLI in the unchanged base Python plus isolated dependency target."""
import argparse
from pathlib import Path

from pairwise.runtime import configure_runtime
configure_runtime()
from pairwise.bridge import load_config, read_pairs
from pairwise.matching import run_matching


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pairs',required=True)
    parser.add_argument('--observations',required=True)
    parser.add_argument('--config',required=True)
    parser.add_argument('--backend',required=True,choices=['roma_v2','loftr_indoor_ds','sift_diagnostic'])
    parser.add_argument('--split',default='all',choices=['all'])
    parser.add_argument('--pair-ids')
    parser.add_argument('--output',required=True)
    parser.add_argument('--overlays',action='store_true')
    args=parser.parse_args()
    pairs=read_pairs(args.pairs,args.split,pair_ids=args.pair_ids.split(',') if args.pair_ids else None)
    config=load_config(args.config)
    result=run_matching(pairs,args.observations,config,args.backend,args.output,
                        overlay_ids=[p['pair_id'] for p in pairs] if args.overlays else None)
    print(f"Saved {len(result['pairs'])} {args.backend} records: {args.output}; status={result['status']}",flush=True)


if __name__=='__main__':main()
