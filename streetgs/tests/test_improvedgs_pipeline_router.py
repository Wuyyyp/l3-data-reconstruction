"""Routing must preserve the ground recipe, booleans and native error handling."""
import importlib.util
from pathlib import Path
import sys,tempfile,unittest,json
R=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('router',R/'scripts/improvedgs_pipeline_router.py')
router=importlib.util.module_from_spec(spec);spec.loader.exec_module(router)
class RouterTests(unittest.TestCase):
 def test_default_scene_does_not_import_ground(self):
  self.assertFalse(router.dispatch_ground(['--scene-mode','static','--gpu','0'],entry='/missing'))
  self.assertFalse(router.dispatch_ground(['--pipeline','scene','--resume','checkpoint.pth'],entry='/missing'))
 def test_ground_forwards_exact_arguments_and_restores_argv(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);entry=p/'entry.py';result=p/'args.json'
   entry.write_text('import json,sys\nfrom pathlib import Path\ndef main():\n Path('+repr(str(result))+').write_text(json.dumps(sys.argv[1:]))\n')
   before=sys.argv
   args=['--config','recipe with spaces.json','--resume','--stage','evaluate','--gpu','2']
   self.assertTrue(router.dispatch_ground(['--pipeline=ground',*args],entry))
   self.assertEqual(json.loads(result.read_text()),args);self.assertIs(sys.argv,before)
 def test_failure_propagates_and_restores_argv(self):
  with tempfile.TemporaryDirectory() as d:
   entry=Path(d)/'entry.py';entry.write_text('def main():\n raise RuntimeError("native failure")\n')
   before=sys.argv
   with self.assertRaisesRegex(RuntimeError,'native failure'):router.dispatch_ground(['--pipeline','ground'],entry)
   self.assertIs(sys.argv,before)
if __name__=='__main__':unittest.main()
