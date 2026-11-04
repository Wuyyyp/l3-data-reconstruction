import json, os, threading
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def write_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f'.{path.name}.{os.getpid()}.{threading.get_ident()}.tmp')
    tmp.write_text(json.dumps(value,indent=2))
    tmp.replace(path)
