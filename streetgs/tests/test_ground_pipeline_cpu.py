"""CPU-only regression tests; no dataset, GPU, or persistent output required."""
from pathlib import Path
import tempfile,os,json,importlib.util,unittest
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class GroundPipelineTests(unittest.TestCase):
    def test_checkpoint_curve_recovery(self):
        runner=module('ground_runner',R/'scripts/run_ground_pipeline.py')
        with tempfile.TemporaryDirectory() as tmp:
            t=Path(tmp);curve=t/'curve.jsonl'
            curve.write_text(''.join(json.dumps(dict(iteration=i))+'\n' for i in range(1,5))+'{"iteration":')
            runner.recover_curve(curve,3)
            self.assertEqual([json.loads(x)['iteration'] for x in curve.read_text().splitlines()],[1,2,3])
            archived=list(t.glob('curve.before_resume_*.jsonl'))
            self.assertEqual(len(archived),1)
            self.assertTrue(archived[0].read_text().endswith('{"iteration":'))
            curve.write_text('{"iteration":1}\nBAD\n{"iteration":3}\n')
            with self.assertRaises(ValueError):runner.recover_curve(curve,3)

    def test_mask_polarity_and_template(self):
        old={k:os.environ.get(k) for k in ['ROAD_PIPELINE_CONFIG','ROAD_STAGE_OUTPUT']}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                t=Path(tmp);ego=t/'ego';ego.mkdir()
                one=np.zeros((8,10),np.uint8);one[4:]=1
                frame=np.zeros_like(one);frame[:4]=255
                Image.fromarray(one).save(ego/'000000_00.png')
                Image.fromarray(frame).save(ego/'000004_00.png')
                cfg=t/'pipeline.json';cfg.write_text(json.dumps(dict(source=str(t),output=str(t/'out'),ego_mask=str(ego))))
                os.environ['ROAD_PIPELINE_CONFIG']=str(cfg);os.environ['ROAD_STAGE_OUTPUT']=str(t/'stage')
                common=module('ground_common',R/'lib/preprocessing/ground_pipeline/common.py')
                self.assertEqual(common.ego_path('000004_00').name,'000004_00.png')
                self.assertEqual(common.ego_path('000010_00').name,'000000_00.png')
                np.testing.assert_array_equal(common.binary_mask(ego/'000000_00.png',(10,8)),one.astype(bool))
                np.testing.assert_array_equal(common.binary_mask(ego/'000004_00.png',(10,8)),frame.astype(bool))
        finally:
            for k,v in old.items():
                if v is None:os.environ.pop(k,None)
                else:os.environ[k]=v

if __name__=='__main__':unittest.main()
