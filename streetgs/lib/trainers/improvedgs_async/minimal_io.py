from pathlib import Path
from lib.utils import m18_utils
from lib.datasets import m18_readers
source=Path(m18_utils.__file__).read_text()
a=source.index("    os.makedirs(f'{cfg.model_path}/obj_bounds'")
b=source.index('    # run colmap',a)
source=source[:a]+source[b:]
ns=dict(m18_utils.__dict__);exec(compile(source,m18_utils.__file__,'exec'),ns)
m18_readers.generate_dataparser_outputs=ns['generate_dataparser_outputs']
