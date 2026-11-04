"""Same-checkpoint, no-training comparison. Run with --config evaluate.yaml."""
import sys, os, importlib, json, time, hashlib, math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch, numpy as np
from PIL import Image, ImageDraw
# Existing dataset was pickled by NumPy 2; keep current NumPy/Torch untouched.
for name in ['multiarray','numeric','umath','_multiarray_umath']:
    importlib.import_module('numpy.core.'+name)
for name,module in list(sys.modules.items()):
    if name=='numpy.core' or name.startswith('numpy.core.'):
        sys.modules[name.replace('numpy.core','numpy._core',1)]=module
from lib.config import cfg
from lib.utils.camera_utils import Camera
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.utils.general_utils import safe_state
from lib.utils.loss_utils import ssim

out=ROOT/'output/gsplat_stage123';out.mkdir(parents=True,exist_ok=True)
safe_state(False)
assert cfg.mode != 'train'
renderer=StreetGaussianRenderer()

def metric(a,b):
    d=(a-b).float(); mse=float(d.square().mean())
    return dict(psnr=(-10*math.log10(mse) if mse else None),mae=float(d.abs().mean()),max=float(d.abs().max()))

def compare(a,b):
    row={k:metric(a[k],b[k]) for k in ['rgb','acc','depth']}
    row['rgb']['ssim']=float(ssim(a['rgb'],b['rgb']))
    valid=(a['acc']>.1)&(b['acc']>.1)
    ea=a['depth']/a['acc'].clamp_min(1e-8);eb=b['depth']/b['acc'].clamp_min(1e-8)
    row['expected_depth_mae_m']=float((ea-eb).abs()[valid].mean()) if valid.any() else None
    row['depth_valid_pixels']=int(valid.sum())
    for result in [a,b]:
        for k in ['rgb','acc','depth']:
            assert torch.isfinite(result[k]).all(), k
        assert float(result['acc'].min())>=-1e-5 and float(result['acc'].max())<=1.0001
    return row

def call(kind,c,pc,backend):
    cfg.render.backend=backend
    torch.cuda.synchronize();t=time.perf_counter()
    result=getattr(renderer,kind)(c,pc)
    torch.cuda.synchronize()
    elapsed=(time.perf_counter()-t)*1000
    return {k:result[k].detach() for k in ['rgb','acc','depth']},elapsed

def rgb_image(t):
    return Image.fromarray((t.clamp(0,1).permute(1,2,0).cpu().numpy()*255+.5).astype('uint8'))

def panel(name,a,b,gt=None):
    folder=out/'images';folder.mkdir(exist_ok=True)
    for label,r in [('legacy',a),('gsplat',b)]:
        rgb_image(r['rgb']).save(folder/(name+'_'+label+'.png'))
    panels=[('Legacy',rgb_image(a['rgb'])),('gsplat',rgb_image(b['rgb'])),
            ('RGB difference x10',rgb_image((a['rgb']-b['rgb']).abs()*10)),
            ('Alpha difference x10',rgb_image(((a['acc']-b['acc']).abs()*10).expand(3,-1,-1)))]
    width=480; height=round(panels[0][1].height*width/panels[0][1].width)
    sheet=Image.new('RGB',(width*4,height+50),'white');draw=ImageDraw.Draw(sheet)
    for i,(label,img) in enumerate(panels):
        draw.text((i*width+8,6),name+' / '+label,fill='black');sheet.paste(img.resize((width,height)),(i*width,40))
    sheet.save(folder/(name+'_compare.jpg'),quality=95)

class Synthetic:
    max_sh_degree=1;active_sh_degree=1
    def __init__(self):
        self.get_xyz=torch.tensor([[-.5,-.2,4.],[.4,.1,5.],[0.,.5,7.],[0,0,.1],[0,0,-3.]],device='cuda')
        self.get_scaling=torch.tensor([[.13,.2,.1]]*5,device='cuda')
        self.get_rotation=torch.tensor([[1.,0,0,0]]*5,device='cuda')
        self.get_opacity=torch.tensor([[.8],[.6],[.9],[.99],[.99]],device='cuda')
        self.get_features=torch.zeros((5,4,3),device='cuda')
        self.get_features[:,0,:]=torch.tensor([[.8,.2,.1],[.1,.8,.2],[.1,.2,.8],[1,1,1],[1,1,1]],device='cuda')

with torch.no_grad():
    synthetic=[]
    for shifted in [False,True]:
        for white in [False,True]:
            K=np.array([[210.,0,142. if shifted else 160.],[0,215.,105. if shifted else 120.],[0,0,1.]])
            c=Camera(0,np.eye(3),np.array([.1,0,.05]) if shifted else np.zeros(3),
                     2*math.atan(320/420),2*math.atan(240/430),K,torch.zeros(3,240,320),'synthetic')
            results=[]
            for backend in ['legacy','gsplat']:
                cfg.render.backend=backend
                results.append(renderer.render_kernel(c,Synthetic(),white_background=white))
            row=compare(*results);row.update(shifted=shifted,white_background=white)
            synthetic.append(row)
            assert row['rgb']['mae']<2e-4, row
            assert row['depth']['mae']<2e-3, row
    (out/'synthetic.json').write_text(json.dumps(synthetic,indent=2))
    print('SYNTHETIC_PASS',json.dumps(synthetic),flush=True)
    from lib.datasets.dataset import Dataset
    from lib.models.street_gaussian_model import StreetGaussianModel
    from lib.models.scene import Scene
    dataset=Dataset();model=StreetGaussianModel(dataset.scene_info.metadata);scene=Scene(model,dataset)
    cams=sorted(scene.getTrainCameras()+scene.getTestCameras(),key=lambda c:c.image_name)
    selected=[c for c in cams if int(c.image_name.split('_')[0]) in [0,40,99]]
    assert len(selected)==21,len(selected)
    rows=[]
    for c in selected:
        w=1920;h=1080 if c.image_name.endswith(('_09','_10')) else 1280
        K=c.K.cpu().numpy().copy();K[0]*=w/c.image_width;K[1]*=h/c.image_height
        c.image_width=w;c.image_height=h;c.set_intrinsic(K)
        # Full render includes actor transforms, Fourier SH, sky and exposure.
        a,ta=call('render',c,model,'legacy');b,tb=call('render',c,model,'gsplat')
        row=compare(a,b);row.update(image=c.image_name,kind='full',legacy_ms=ta,gsplat_ms=tb)
        gt=np.asarray(Image.open(Path(cfg.source_path)/'images'/(c.image_name+'.jpg')).convert('RGB')).copy()
        gt=torch.from_numpy(gt).cuda().permute(2,0,1).float()/255
        row['legacy_vs_gt']=metric(a['rgb'],gt);row['gsplat_vs_gt']=metric(b['rgb'],gt)
        row['legacy_vs_gt']['ssim']=float(ssim(a['rgb'],gt));row['gsplat_vs_gt']['ssim']=float(ssim(b['rgb'],gt))
        rows.append(row);panel(c.image_name,a,b)
        if c.image_name in ['000040_03','000040_10','000099_09']:
            for kind in ['render_background','render_object']:
                aa,tta=call(kind,c,model,'legacy');bb,ttb=call(kind,c,model,'gsplat')
                r=compare(aa,bb);r.update(image=c.image_name,kind=kind,legacy_ms=tta,gsplat_ms=ttb);rows.append(r)
                panel(c.image_name+'_'+kind,aa,bb)
            # Explicit raw kernel test isolates sky / exposure from rasterization.
            model.set_visibility(list(model.model_name_id.keys()));model.parse_camera(c)
            aa,tta=call('render_kernel',c,model,'legacy');bb,ttb=call('render_kernel',c,model,'gsplat')
            r=compare(aa,bb);r.update(image=c.image_name,kind='raw_kernel',legacy_ms=tta,gsplat_ms=ttb);rows.append(r)
            panel(c.image_name+'_raw_kernel',aa,bb)
            np.savez_compressed(out/(c.image_name+'_layers.npz'),**{
                label+'_'+k:r[k].cpu().numpy() for label,r in [('legacy',aa),('gsplat',bb)] for k in ['rgb','acc','depth']})
        (out/'per_image.json').write_text(json.dumps(rows,indent=2))
        print('RENDER',c.image_name,json.dumps(row),flush=True)
    import gsplat
    result=dict(synthetic=synthetic,rows=rows,checkpoint=str(Path(cfg.trained_model_dir)/'iteration_25000.pth'),
                gsplat=gsplat.__version__,torch=torch.__version__,cuda=torch.version.cuda,
                note='GT metrics are unmasked full-image diagnostics; timings include Python/model work and are not a throughput benchmark.')
    (out/'results.json').write_text(json.dumps(result,indent=2))
    print('COMPLETE',len(rows),flush=True)
