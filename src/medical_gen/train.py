"""Train a reference generator, independent pathology adaptation, or scratch baseline."""
import argparse
from pathlib import Path


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--resume',type=Path,help='Explicit Lightning checkpoint for resuming this same run')
    args=parser.parse_args()
    from .training_runtime import run_training
    run_training(args.config,args.resume)

if __name__=='__main__':main()
