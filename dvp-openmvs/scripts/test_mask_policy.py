import json,sys,tempfile,subprocess,unittest
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'mvs_process'))
from mask_policy import resize_ignore,rig_mask
from prepare_dvp_scene import metadata_matches_request
class Masks(unittest.TestCase):
    def test_thin_invalid_region_survives_downsampling(self):
        a=np.zeros((8,8),np.uint8);a[1,1]=255
        self.assertTrue(resize_ignore(a,(2,2))[0,0])
        self.assertFalse(resize_ignore(np.zeros_like(a),(2,2)).any())
    def test_changed_mask_invalidates_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'meta.json';p.write_text(json.dumps(dict(mask_sha256='old',dvp_depth_range_m=[1,10],depth_range_source='test',lidar_calibration_upper_bound_m=None)))
            request=dict(depth_min_m=1,depth_max_m=10,source='test')
            self.assertTrue(metadata_matches_request(p,request,True,'old'))
            self.assertFalse(metadata_matches_request(p,request,True,'new'))
    def test_lossless_mask_and_explicit_rig(self):
        mask=rig_mask((100,200),[[0,.64],[1,.64]])
        self.assertTrue((mask[80]==255).all());self.assertTrue((mask[40]==0).all())
        ok,buf=cv2.imencode('.png',255-mask);self.assertTrue(ok)
        self.assertTrue(np.array_equal(cv2.imdecode(buf,0),255-mask))
if __name__=='__main__':unittest.main()
