"""Route the unified entry to the existing ground pipeline without changing its recipe.
Do not translate scene-only flags or checkpoint semantics: the ground parser owns them.
"""
import argparse
import importlib.util
from pathlib import Path
import sys

GROUND_ENTRY=Path(__file__).resolve().with_name('run_ground_pipeline.py')

def dispatch_ground(argv, entry=None):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--pipeline',choices=['scene','ground'],default='scene')
    selected,remaining=parser.parse_known_args(argv)
    if selected.pipeline!='ground':return False
    target=Path(entry) if entry is not None else GROUND_ENTRY
    spec=importlib.util.spec_from_file_location('_streetgs_ground_pipeline_entry',target)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    previous=sys.argv
    try:
        sys.argv=[str(target),*remaining]
        module.main()
    finally:
        sys.argv=previous
    return True
