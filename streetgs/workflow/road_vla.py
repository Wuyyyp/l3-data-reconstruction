import json
import sys
from pathlib import Path

sys.path.insert(1,str(Path(__file__).resolve().parent))
from task_queue import write_json
from vla_job import ENGINE
sys.path.insert(0,str(ENGINE))
from roadmask_batch import prepare_case
case=Path(sys.argv[1])
config = case/'config.json'
if not (case/'full/pipeline_config.json').exists():
    data = json.loads(config.read_text())
    data['model']['ground']['sh_degree'] = 0
    write_json(config, data)
prepare_case(config,int(sys.argv[2]))
