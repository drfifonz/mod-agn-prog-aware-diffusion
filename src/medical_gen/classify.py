"""Run repeated classifier comparisons against fixed real validation and test sets."""
import argparse
import json
from pathlib import Path
from .protocol_config import ClassificationConfig
from .datasets.protocol import build_selection


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true',help='Validate and export selections without torch or training')
    args=parser.parse_args();cfg=ClassificationConfig.load(args.config)
    selection=build_selection(cfg.data)
    output=cfg.output_root.expanduser()/cfg.experiment_name
    if output.exists():raise FileExistsError(f'Refusing to overwrite experiment: {output}')
    summary=selection.export(output/'selection')
    (output/'config.json').write_text(cfg.model_dump_json(indent=2)+'\n')
    if args.prepare_only:
        print(json.dumps(summary,indent=2));return
    from .classification_runtime import run_repeats
    run_repeats(cfg,selection,summary,output)

if __name__=='__main__':main()
