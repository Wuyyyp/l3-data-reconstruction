import json, os, sys
from pathlib import Path
work=Path(sys.argv[1])
sys.path.insert(0,str(Path(os.environ['DVP_ROOT'])/'l3'))
sys.path.insert(1,str(Path(os.environ['DVP_ROOT'])/'mvs_process'))
from dvp_driver import run_window
os.environ.update(WYS_OLD_CASE=str(work/'geometry'), WYS_CASE_ROOT=str(work))
ply=run_window(next((work/'delivery').iterdir()),work/'dvp',0,100,int(sys.argv[2]),far_only=True)
(work/'dvp_result.json').write_text(json.dumps({'ply':str(ply),'far_only':True}))
