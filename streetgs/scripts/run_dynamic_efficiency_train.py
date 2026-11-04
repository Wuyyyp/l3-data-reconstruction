"""Fresh training entrypoint, numpy compatibility only; production training unchanged."""
import sys,os,importlib,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for name in ['multiarray','numeric','umath','_multiarray_umath']:importlib.import_module('numpy.core.'+name)
for name,module in list(sys.modules.items()):
 if name=='numpy.core' or name.startswith('numpy.core.'):sys.modules[name.replace('numpy.core','numpy._core',1)]=module
runpy.run_path(str(ROOT/'train.py'),run_name='__main__')
