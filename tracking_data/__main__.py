"""Check real dataset paths and one batch: python -m tracking_data config.json."""
import argparse
import json
from .loader import build_loader

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    args=parser.parse_args()
    with open(args.config,encoding='utf-8-sig') as f: config=json.load(f)
    batch=next(iter(build_loader(config)))
    for key,value in batch.items():
        print(key,tuple(value.shape) if hasattr(value,'shape') else value)

if __name__=='__main__': main()
