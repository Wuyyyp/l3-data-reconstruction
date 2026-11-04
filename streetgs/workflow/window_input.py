"""Map the delivered PINHOLE model to the samples-v3 training canvas without a new BA."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv,json,pickle,struct,sys,shutil
import numpy as np
import cv2
from PIL import Image
from prepare_pose_input import U,CAMERAS
from l3_wys_canvas import VERSION,geometry,render,write_masks
from sample_inputs import fill_metadata_gaps

def sparse_canvas(src,dst):
    dst.mkdir(parents=True,exist_ok=True)
    by_camera={}
    with (src/'cameras.bin').open('rb') as i,(dst/'cameras.bin').open('wb') as o:
        n,=struct.unpack('<Q',i.read(8));assert n==7;o.write(struct.pack('<Q',n))
        for _ in range(n):
            cid,kind,w,h,fx,fy,cx,cy=struct.unpack('<iiQQ4d',i.read(56));assert kind==1
            # The two front cameras have 16:9 images, the five others have 3:2.
            g=geometry(10 if w*9==h*16 else 0,(w,h))
            A=np.asarray(g['A']);K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
            q=A@K;by_camera[cid]=(g,K)
            o.write(struct.pack('<iiQQ4d',cid,kind,1600,900,q[0,0],q[1,1],q[0,2],q[1,2]))
        assert not i.read(1)
    names={}
    with (src/'images.bin').open('rb') as i,(dst/'images.bin').open('wb') as o:
        head=i.read(8);n,=struct.unpack('<Q',head);assert n==707;o.write(head)
        for _ in range(n):
            rec=i.read(64);o.write(rec);cid=struct.unpack('<i',rec[-4:])[0];s=bytearray()
            while True:
                b=i.read(1)
                if not b:raise EOFError('image name')
                if b==b'\x00':break
                s.extend(b)
            o.write(s+b'\x00');name=s.decode();names[name]=by_camera[cid]
            head=i.read(8);o.write(head);size,=struct.unpack('<Q',head)
            obs=np.frombuffer(i.read(size*24),dtype=[('xy','<f8',(2,)),('pid','<i8')]).copy()
            # COLMAP XYs are expressed with the upper left corner at (0,0).
            # Match exactly the affine map applied to the stored PINHOLE K.
            A=np.array(by_camera[cid][0]['A']);obs['xy']=obs['xy']@A[:2,:2].T+A[:2,2]
            o.write(obs.tobytes())
        assert not i.read(1) and len(names)==707
    for f in src.glob('*.bin'):
        if f.name not in ('images.bin','cameras.bin'):shutil.copyfile(f,dst/f.name)
    return names

def main(w,src):
    pose=w/'pose';delivery=pose/'run/deliverables';ba=delivery/'colmap-ba'
    v=json.loads((src/'window_verification.json').read_text()) if (src/'window_verification.json').exists() else {'status':'PASS','source_range':[int(__import__('os').environ['SOURCE_START']),int(__import__('os').environ['SOURCE_START'])+100]};report=json.loads((src/'colmap-ba/report.json').read_text())
    assert v['status']=='PASS' and report['coverage']=='complete' and len(report['selected_frames'])==101 and report['selected_frames']==list(range(report['selected_frames'][0],report['selected_frames'][0]+101))
    models=sparse_canvas(src/'colmap-ba/sparse_after',ba/'sparse_after')

    base=delivery/'image_extrinsics';base.mkdir(exist_ok=True)
    rows=list(csv.DictReader((src/'image_extrinsics/image_camera_to_world_after.csv').open()))
    after=list(csv.DictReader((src/'colmap-ba/image_camera_to_world_after_ba.csv').open()))
    indices=sorted({int(r['frame']) for r in after})
    assert len(indices)==101 and indices==list(range(indices[0],indices[0]+101))
    # BA may cover an interior window of a longer base PKL/CSV. Match image identities before rebasing.
    rows=[r for r in rows if int(r['frame']) in indices]
    for r in rows+after:r['frame']=str(int(r['frame'])-indices[0])
    with (ba/'image_camera_to_world_after_ba.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(after[0]));writer.writeheader();writer.writerows(after)
    lookup={(int(r['frame']),r['camera']):r for r in rows}
    assert len(lookup)==len(rows)==707 and {(int(r['frame']),r['camera'],r['image_filename']) for r in after}=={(int(r['frame']),r['camera'],r['image_filename']) for r in rows}
    data=U((src/'result_interpolation_optimized.pkl').open('rb')).load()
    front='front_camera_fov30'
    by_image={str(int(info['sensors']['cams'][front]['timestamp']))+'.jpg':info for info in data['infos']}
    data['infos']=[by_image[lookup[i,front]['image_filename']] for i in range(101)]
    if not (src/'window_verification.json').exists() and indices[0]!=0:
        v['source_range']=[indices[0],indices[-1]]
    tasks=[];cv2.setNumThreads(1)
    for fi,info in enumerate(data['infos']):
        lid=info['sensors']['lidar']['perception'];pcd=src/'compensated_perception'/Path(lid['data_path']).name
        assert pcd.is_file() and lid.get('dynamic_compensated');lid['aws_path']=lid['data_path']=str(pcd)
        for name,cid in CAMERAS.items():
            cam=info['sensors']['cams'][name];r=lookup[fi,name];f=r['image_filename']
            assert f==str(int(cam['timestamp']))+'.jpg'
            g,k=models[name+'/'+f];assert g==geometry(cid,tuple(g['source_wh']))
            np.testing.assert_allclose(cam['cam_intrinsic'],k,rtol=0,atol=1e-6)
            dst=pose/'canonical/images'/name/f
            source_path=(src/Path(r['image_path'])).resolve(strict=True)
            tasks.append((source_path,dst,g))
            if fi==0:write_masks(pose/'source_fixed_masks',cid,g)
            nk=(np.array(g['A'])@k).tolist()
            cam.update(cam_intrinsic=nk,cam_intrinsic_resize=nk,source_geometry=g,images_undistorted=True,
                       original_cam_intrinsic=k.tolist(),original_data_path=str(source_path),aws_path=str(dst),data_path=str(dst),
                       source_d=cam.get('d'),d=[0.]*len(cam.get('d',[])),cam_dist={'type':'pinhole',**{z:0. for z in cam.get('cam_dist',{}) if z!='type'}})
            r['image_path']=str(dst)
    def convert(task):
        path,dst,g=task;im=cv2.imread(str(path));assert im is not None,path
        dst.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(dst),render(im,g)):raise IOError(dst)
    with ThreadPoolExecutor(4) as pool:list(pool.map(convert,tasks))
    with (base/'image_camera_to_world_after.csv').open('w') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    fill_metadata_gaps(w/'no_prior_converted',pose,data['infos'])
    with (delivery/'result_interpolation_optimized.pkl').open('wb') as f:pickle.dump(data,f,protocol=4)
    (pose/'input_audit.json').write_text(json.dumps(dict(sample=src.name,source=str(src),geometry_version=VERSION,source_range=v['source_range'],frames=101,images=707,BA_extrinsics='unchanged',intrinsics_and_sparse_observations='same pixel affine as RGB',compensated_pcd=101)))
if __name__=='__main__':main(*map(Path,sys.argv[1:]))

