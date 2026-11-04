"""Pair BA cameras with unchanged ego-frame LiDAR and annotations for Mesh and DVP."""
import argparse,copy,csv,json,pickle,struct,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2,numpy as np
from PIL import Image
from prepare_pose_input import CAMERAS,U,ROOT
EGO='ego2global_transformation_matrix_camera_reference_offset_optimized'

def cameras_bin(path):
    out={}
    with Path(path).open('rb') as f:
        for _ in range(struct.unpack('<Q',f.read(8))[0]):
            cid,kind,w,h=struct.unpack('<iiQQ',f.read(24));assert kind==1,(cid,kind)
            fx,fy,cx,cy=struct.unpack('<4d',f.read(32))
            assert (w,h)==(1600,900),(w,h)
            out[cid]=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
        assert not f.read(1)
    return out


def projection(points,T,K):
    W=np.linalg.inv(T);q=points@W[:3,:3].T+W[:3,3];uv=q@K.T
    with np.errstate(divide='ignore',invalid='ignore'):xy=uv[:,:2]/uv[:,2:]
    good=np.isfinite(xy).all(1)&(q[:,2]>.1)&(xy[:,0]>=0)&(xy[:,0]<1600)&(xy[:,1]>=0)&(xy[:,1]<900)
    ids=np.flatnonzero(good);pixels=xy[ids].astype(int)
    depth=np.full(1600*900,np.inf,'f4');np.minimum.at(depth,pixels[:,1]*1600+pixels[:,0],q[ids,2])
    depth=depth.reshape(900,1600);depth[~np.isfinite(depth)]=0
    return ids,pixels,q[ids,2],depth


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--sample',required=True);a=p.parse_args()
    work=a.work;pose=work/'pose';delivery=pose/'run/deliverables';out=work/'geometry';out.mkdir(exist_ok=False)
    src=work/'no_prior_converted'
    data=U((delivery/'result_interpolation_optimized.pkl').open('rb')).load();infos=data['infos'];assert len(infos)==101
    ba=delivery/'colmap-ba';rows=list(csv.DictReader((ba/'image_camera_to_world_after_ba.csv').open()))
    lookup={(int(r['frame']),r['camera']):r for r in rows}
    assert len(lookup)==len(rows)==707
    assert set(lookup)=={(f,c) for f in range(101) for c in CAMERAS}
    # Read exactly the model paired with the delivered CSV; no extra BA.
    from geometry_io import model_images,box_polygon
    sparse=ba/'sparse_after';models=model_images(sparse/'images.bin');Ks=cameras_bin(sparse/'cameras.bin')
    sem={}
    for meta in (pose/'cache/semantic_masks').glob('**/metadata/*.json'):
        m=json.loads(meta.read_text());sem[Path(m['source']['path']).parent.name,Path(m['source']['path']).name]=m
    cv2.setNumThreads(1)
    for folder in ['images','ego_pose','extrinsics','intrinsics','lidar_depth','dynamic_mask','fixed_mask','valid_mask','sky_mask']:(out/'converted'/folder).mkdir(parents=True)
    if (src/'pointcloud.npz').is_file():
        with np.load(src/'pointcloud.npz',allow_pickle=True) as z:clouds=z['pointcloud'].item()
    else:clouds={}
    from geometry_io import read_pcd
    for fi,info in enumerate(infos):
        if fi not in clouds:
            clouds[fi]=read_pcd(Path(info['sensors']['lidar']['perception']['data_path']))
    assert sorted(clouds)==list(range(101))
    projections={};views=[];calibrations={};records=[];audits=[]
    audit=json.loads((pose/'input_audit.json').read_text());first=audit['source_range'][0]
    for fi,info in enumerate(infos):
        ego=np.asarray(info[EGO]);assert ego.shape==(4,4) and np.isfinite(ego).all()
        np.savetxt(out/'converted/ego_pose'/f'{fi:06d}.txt',ego)
        points=np.asarray(clouds[fi]);assert points.ndim==2 and points.shape[1]==3 and np.isfinite(points).all()
        world=points@ego[:3,:3].T+ego[:3,3]
        proj=np.full((len(points),21),-1,dtype='i4');slots=np.zeros(len(points),dtype='i4')
        for col,(camera,cid) in enumerate(CAMERAS.items()):
            row=lookup[fi,camera];cam=info['sensors']['cams'][camera];filename=str(int(cam['timestamp']))+'.jpg'
            key=camera+'/'+filename;model=models[key];T=np.array([float(row[f'T_cw_{i}{j}']) for i in range(4) for j in range(4)]).reshape(4,4)
            np.testing.assert_allclose(T,np.linalg.inv(model['w2c']),atol=1e-7,rtol=0)
            K=Ks[model['camera_id']];np.testing.assert_allclose(K,cam['cam_intrinsic'],atol=1e-6)
            name=f'{fi:06d}_{cid:02d}';image=pose/'canonical/images'/camera/filename
            (out/'converted/images'/(name+'.jpg')).symlink_to(image)
            fixed=np.asarray(Image.open(pose/'source_fixed_masks'/f'000000_{cid:02d}.png').convert('L').resize((1600,900),Image.Resampling.NEAREST))>0
            skyfile=work/'sky'/(camera+'_'+Path(filename).stem+'.png')
            sky=cv2.imread(str(skyfile),0);assert sky is not None and sky.shape==(900,1600)
            sky=sky>127
            ids,uv,depths,depth=projection(world,T,K)
            left,_,width,_=cam['source_geometry']['content_xywh']
            inside=(uv[:,0]>=left)&(uv[:,0]<left+width)
            ids,uv,depths=ids[inside],uv[inside],depths[inside]
            depth[:,:left]=0;depth[:,left+width:]=0
            # Native StreetGS colors from the first projection: pack visible cameras first.
            columns=slots[ids]*3
            proj[ids,columns]=cid;proj[ids,columns+1]=uv[:,0];proj[ids,columns+2]=uv[:,1];slots[ids]+=1
            valid=np.isfinite(depth)&(depth>.1)&(depth<=200)&~fixed
            np.save(out/'converted/lidar_depth'/(name+'.npy'),dict(mask=valid,value=depth[valid]))
            dyn=np.zeros((900,1600),'u1')
            for obj in info['objects']:
                box=np.r_[obj['location'],obj['size'],obj['rotation'][2]]
                polygon=box_polygon(box,np.linalg.inv(T)@ego,K,1600,900)
                if polygon is not None:cv2.fillPoly(dyn,[polygon],255)
            for folder,mask in [('dynamic_mask',dyn>0),('fixed_mask',fixed),('valid_mask',~fixed),('sky_mask',sky)]:Image.fromarray(mask.astype('u1')*255).save(out/'converted'/folder/(name+'.png'))
            np.savetxt(out/'converted/intrinsics'/(name+'.txt'),K)
            np.savetxt(out/'converted/extrinsics'/(name+'.txt'),np.linalg.inv(ego)@T)
            np.savetxt(out/'converted/ego_pose'/(name+'.txt'),T)
            cam.update(camera_to_world_optimized=T.tolist(),cam_intrinsic=K.tolist())
            views.append(dict(name=name,frame=fi,original_frame=first+fi,cam=cid,K=K.tolist(),c2w=T.tolist(),width=1600,height=900,image=str(image)))
            records.append(dict(label=name+'.jpg',camera=camera,camera_to_world=T.tolist()))
            calibrations[camera]=dict(K=K.tolist(),width=1600,height=900)
            audits.append(dict(name=name,source_frame=first+fi,lidar_pixels=int(valid.sum()),csv_model_max_error=float(np.max(abs(T-np.linalg.inv(model['w2c']))))))
        projections[fi]=proj
        info['original_gt_boxes']=copy.deepcopy(info['gt_boxes']);info['original_gt_names']=copy.deepcopy(info['gt_names'])
        info['track_id']=[int(o['id']) for o in info['objects']]
        info['gt_boxes']=np.array([o['location']+o['size']+[o['rotation'][2]] for o in info['objects']],float).reshape(-1,7)
        info['gt_names']=[o['type'] for o in info['objects']]
        print('FRAME',fi,flush=True)
    np.savez_compressed(out/'converted/pointcloud.npz',pointcloud=clouds,camera_projection=projections)
    for item in ['track','timestamps.json','timestamps_specific.json']:
        entry=src/item
        if (pose/'supplements'/item).exists():entry=pose/'supplements'/item
        if not entry.exists():raise FileNotFoundError(entry)
        (out/'converted'/item).symlink_to(entry)
    (out/'converted/input_manifest.json').write_text(json.dumps(dict(records=records,calibrations=calibrations),indent=2))
    (out/'input_manifest.json').write_text(json.dumps(dict(views=views,resolution=[1600,900],cameras=list(CAMERAS.values()),world_origin=[0,0,0],source_range=[first,first+100]),indent=2))
    mesh=out/'mesh_input';(mesh/'videos').mkdir(parents=True)
    with (mesh/'result_interpolation_optimized_first101.pkl').open('wb') as f:pickle.dump(data,f,protocol=4)
    manifest=dict(frames=101,source=str(delivery),source_frame_range=[first,first+100],videos=[dict(camera=c,file=f'videos/{c}.mp4') for c in CAMERAS])
    (mesh/'manifest.json').write_text(json.dumps(manifest,indent=2))
    def encode(camera):
        dst=mesh/'videos'/(camera+'.mp4')
        subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-framerate','10','-i',str(out/'converted/images'/f'%06d_{CAMERAS[camera]:02d}.jpg'),'-frames:v','101','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-threads','2',str(dst)],check=True)
    with ThreadPoolExecutor(2) as pool:list(pool.map(encode,CAMERAS))
    prior_deltas=[]
    for view in views:
        pk=src/'ego_pose'/(view['name']+'.txt')
        if pk.is_file():
            prior=np.loadtxt(pk);delta=np.linalg.inv(prior)@np.array(view['c2w'])
            prior_deltas.append([view['name'],float(np.linalg.norm(delta[:3,3])),float(np.degrees(np.arccos(np.clip((np.trace(delta[:3,:3])-1)/2,-1,1))))])
    (out/'samples_vs_ba.json').write_text(json.dumps(dict(columns=['image','translation_m','rotation_deg'],authority='samples initial -> delivered sparse_after',rows=prior_deltas)))
    (out/'geometry_audit.json').write_text(json.dumps(dict(passed=True,views=audits,camera_authority=str(ba/'image_camera_to_world_after_ba.csv'),lidar_compensated=True,pointcloud_source=str(Path(audit['source'])/'compensated_perception')),indent=2))

if __name__=='__main__':main()
