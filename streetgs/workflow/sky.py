from pathlib import Path
import sys,cv2,numpy as np
from dongfeng_registration.semantic_masks import DynamicMaskConfig,Mask2FormerMapillaryPredictor
from prepare_pose_input import CAMERAS
w=Path(sys.argv[1]);cv2.setNumThreads(1)
pred=Mask2FormerMapillaryPredictor(DynamicMaskConfig(cache_root=w/'sky_cache',device='cuda',processor_image_size=1024))
out=w/'sky';out.mkdir(exist_ok=True)
for camera,cid in CAMERAS.items():
    paths=sorted((w/'pose/canonical/images'/camera).glob('*.jpg'));assert len(paths)==101
    for path in paths:
        dest=out/(camera+'_'+path.stem+'.png')
        if dest.exists() and cv2.imread(str(dest),0) is not None:continue
        m=pred.predict(cv2.imread(str(path)))
        ids=[k for k,v in m.class_id_to_name.items() if v.lower().split('--')[-1]=='sky']
        if not cv2.imwrite(str(dest),np.isin(m.labels,ids).astype('uint8')*255):raise IOError(dest)
    print('sky ready',camera,flush=True)
