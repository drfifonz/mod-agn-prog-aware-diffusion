"""Generate individual synthetic images from configured checkpoints."""
import argparse
from pathlib import Path

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--num-images',type=int)
    parser.add_argument('--device',choices=['auto','cuda','cpu','mps'])
    parser.add_argument('--output-root')
    parser.add_argument('--no-wandb',action='store_true')
    parser.add_argument('--model-index',type=int)
    args=parser.parse_args()
    if args.num_images is not None and args.num_images<1:parser.error('--num-images must be positive')
    from .generation_runtime import main as generate
    generate(args)

if __name__=='__main__':main()
