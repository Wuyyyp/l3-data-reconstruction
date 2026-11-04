"""Render every camera at the same signed ego-right translation, without changing actor time."""
from ground_extension import *
from scene.gaussian_model import GaussianModel
from PIL import Image

@torch.no_grad()
def main():
    torch.set_num_threads(4);ck=torch.load(sys.argv[1]);out=Path(sys.argv[2]);offset=float(sys.argv[3]);out.mkdir(parents=True,exist_ok=False)
    (out/'images').mkdir();ds,cams=dataset();assert ck['camera_names']==[c.image_name for c in cams]
    bg=GaussianModel(1,'ours_adam');state=ck['model'];bg.active_sh_degree=state[0];bg.max_sh_degree=state[1]
    for k,v in zip(PARAMS,state[2:8]):setattr(bg,k,v)
    cfg.mode='evaluate';scene=DynamicScene(ds,cams,bg,out,initialize=False)
    if scene.sg.actor_pose is not None:scene.sg.actor_pose.training_setup()
    scene.restore(ck['dynamic']);scene.audit_finite();scene.sg.active_sh_degree=bg.active_sh_degree;iteration=ck['iteration'];del ck,state
    source=Path(cfg.source_path);manifest=json.loads((source.parent/'input_manifest.json').read_text());views={r['name']:r for r in manifest['views']}
    lookup={c.image_name:c for c in ds.test_cameras[1]};assert set(lookup)==set(views) and len(views)==707
    pipe=SimpleNamespace(debug=False,antialiasing=False);background=torch.zeros(3,device='cuda');audit=[]
    for name,c in lookup.items():
        r=views[name];ego=np.loadtxt(source/'ego_pose'/f"{r['frame']:06d}.txt");right=ego[:3,0]/np.linalg.norm(ego[:3,0]);front=lookup[f"{r['frame']:06d}_10"]
        assert np.dot(front.get_extrinsic()[:3,0],right)>.95,'Unexpected ego-right convention'
        base=c.get_extrinsic().copy();moved=base.copy();delta=offset*right;moved[:3,3]+=delta;c.set_extrinsic(moved)
        c.alpha_mask=masks_for_sky(c)['mask'];pkg=scene.render(c,bg,pipe,background,track_gradients=False);rgb=scene.sg.color_correction(c,pkg['render']).clamp(0,1)
        if int(name.split('_')[-1])<9 and manifest.get('geometry_version')!='l3-wys-full-frame-v1':
            rgb[:,:,:125]=0;rgb[:,:,1475:]=0
        Image.fromarray((rgb.cpu().permute(1,2,0).numpy()*255).round().astype('u1')).save(out/'images'/(name+'.png'))
        absolute=np.array(r['c2w']);absolute[:3,3]+=delta;r['c2w']=absolute.tolist();r['image_file']=name+'.png'
        audit.append(dict(name=name,delta_world=delta.tolist(),distance=float(np.linalg.norm(delta))))
        c.set_extrinsic(base);del pkg,rgb
        if len(audit)%100==0:print('SHIFT_RENDER',len(audit),707,flush=True)
    assert len(audit)==707 and all(abs(x['distance']-abs(offset))<1e-6 for x in audit)
    (out/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'render_audit.json').write_text(json.dumps(dict(passed=True,iteration=iteration,offset_ego_right_m=offset,images=707,orientation_unchanged=True,actor_time_unchanged=True,poses=audit),indent=2))

if __name__=='__main__':main()
