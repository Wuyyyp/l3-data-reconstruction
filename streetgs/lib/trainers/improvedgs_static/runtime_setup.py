import sys, os, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
SETTINGS = json.loads(Path(os.environ['STREET_IMPROVEDGS_CONFIG']).read_text())
VARIANT = 'streetgs_improvedgs_static_sky'
CONFIG = dict(backend='gsplat', with_sky=SETTINGS.get('sky', True))
sys.path.insert(0, str(ROOT / 'native_source'))
from backend_bridge import gsplat_render
import scene.methods.densification_stage as ds
render = gsplat_render
ds.render = render
