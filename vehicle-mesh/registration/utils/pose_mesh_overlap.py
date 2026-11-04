"""Speculative image-only SAM3 inference. All geometric association stays post-BA."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

PROJECT = Path(__file__).resolve().parents[3]
WORK_ROOT = Path(os.environ.get('L3_WORK_ROOT',PROJECT/'work')).resolve()
UTILS = Path(__file__).resolve().parent


def cache_dir(work):
    return work/'.mesh_overlap_cache'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_key(video, model, prompt, count, shape):
    stat = model.stat()
    fields = ['sam3-raw-v1', digest(video), str(model.resolve()), stat.st_size,
              stat.st_mtime_ns, prompt, count, list(shape)]
    return hashlib.sha256(json.dumps(fields).encode()).hexdigest()


def sources_key(paths):
    h=hashlib.sha256()
    for p in paths:
        st=p.stat()
        h.update(f'{st.st_dev}:{st.st_ino}:{st.st_size}:{st.st_mtime_ns}\n'.encode())
    return h.hexdigest()


def cached_video(work, camera, paths, dest):
    folder=cache_dir(work)
    if not folder.is_dir():return False
    src=folder/(camera+'-'+sources_key(paths)+'.mp4')
    if not src.is_file():return False
    os.link(src,dest)
    return True


def pack(outputs, shape):
    import numpy as np
    ids = np.asarray(outputs['out_obj_ids'])
    scores = np.asarray(outputs['out_probs'])
    masks = np.asarray(outputs['out_binary_masks'], dtype=bool).reshape((-1,*shape))
    if len(ids)!=len(masks) or len(scores)!=len(ids):
        raise ValueError('SAM3 instance counts differ')
    stream = io.BytesIO()
    np.savez(stream, ids=ids, scores=scores, bits=np.packbits(masks,axis=-1))
    return stream.getvalue()


def unpack(raw, shape):
    import numpy as np
    with np.load(io.BytesIO(raw),allow_pickle=False) as f:
        masks = np.unpackbits(f['bits'],axis=-1,count=shape[1]).view(bool)
        if masks.shape!=(len(f['ids']),*shape):
            raise ValueError('SAM3 cache shape mismatch')
        return dict(out_obj_ids=f['ids'],out_probs=f['scores'],out_binary_masks=masks)


def replay(video, model, prompt, count, shape, consume):
    video=Path(video).resolve()
    if video.parent.parts[-3:]!=('geometry','mesh_input','videos'):return False
    work=video.parents[3]
    if work.parent.parent!=WORK_ROOT:return False
    cache=cache_dir(work)
    if not cache.is_dir():return False
    path=cache/(cache_key(video,Path(model),prompt,count,shape)+'.zip')
    if not path.is_file():return False
    with zipfile.ZipFile(path) as z:
        order=z.namelist()
        if sorted(map(int,order))!=list(range(count)):raise ValueError('Incomplete SAM3 cache')
        for name in order:consume(int(name),unpack(z.read(name),shape))
    print('REUSED IMAGE-ONLY SAM3',video.name,prompt,count,flush=True)
    return True


def precompute(work,tmp):
    import torch
    sys.path.insert(0,str(UTILS))
    from sam3_video import Sam3VideoTracker,checkpoint
    from motion_mask_utils import PROMPTS
    torch.set_num_threads(2)
    if torch.cuda.mem_get_info()[0]<32*1024**3:
        print('Skipping speculative inference: GPU headroom below 32 GiB',flush=True);return
    tracker=Sam3VideoTracker()
    audit=json.loads((work/'pose/input_audit.json').read_text())
    by_cam={}
    for row in audit['mapping']:
        by_cam.setdefault(row['camera'],{})[row['frame']]=Path(row['image'])
    cache=cache_dir(work)
    cache.mkdir(parents=True,exist_ok=True)
    try:
        for prompt in dict.fromkeys([os.environ.get('SAM3_VEHICLE_PROMPT','car'),*PROMPTS]):
            for cam,frames in by_cam.items():
                if set(frames)!=set(range(101)):raise ValueError('Incomplete camera input')
                if torch.cuda.mem_get_info()[0] < 18*1024**3 or shutil.disk_usage(cache).free < 500*1024**3:
                    print('Yielding resources to production',flush=True);return
                key=sources_key(frames[i] for i in range(101))
                video=cache/(cam+'-'+key+'.mp4')
                if not video.exists():
                    frames_dir=tmp/cam;frames_dir.mkdir()
                    for i,path in frames.items():(frames_dir/f'{i:06d}.jpg').symlink_to(path)
                    subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-framerate','10','-i',str(frames_dir/'%06d.jpg'),'-frames:v','101','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-threads','2',str(tmp/'encode.mp4')],check=True)
                    (tmp/'encode.mp4').replace(video)
                shape=(900,1600);model=checkpoint()
                path=cache/(cache_key(video,model,prompt,101,shape)+'.zip')
                if path.exists():continue
                # Publish only whole sequences. A stopped speculative task leaves no usable partial.
                with zipfile.ZipFile(tmp/'prediction.zip','w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
                    tracker.infer(video,model,prompt,101,shape,frame_consumer=lambda i,o:z.writestr(str(i),pack(o,shape)))
                    if sorted(map(int,z.namelist()))!=list(range(101)):raise ValueError('Missing frames')
                (tmp/'prediction.zip').replace(path)
                print('READY',cam,prompt,101,path.name,flush=True)
    finally:tracker.close()


@contextmanager
def during_cpu_ba(config):
    import yaml
    work=config.resolve().parent.parent
    if (config.name!='delivery.yaml' or work.parent.parent!=WORK_ROOT
        or not (work/'canvas_input_check.json').is_file()
        or json.loads((work/'canvas_input_check.json').read_text()).get('version')!='l3-wys-full-frame-v1'
        or yaml.safe_load(config.read_text())['colmap'].get('use_gpu',True)):
        yield;return
    # The optional process shares only this worker's GPU and process group.
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    cgroup=Path('/sys/fs/cgroup/memory')
    if (cgroup/'memory.limit_in_bytes').is_file():
        limit=int((cgroup/'memory.limit_in_bytes').read_text())
        used=int((cgroup/'memory.usage_in_bytes').read_text())
        file_cache=int(next((s.split()[1] for s in (cgroup/'memory.stat').read_text().splitlines() if s.startswith('total_inactive_file ')),0))
        available=min(available,limit-used+file_cache)
    if available<40*1024**3 or shutil.disk_usage(work).free<500*1024**3:
        print('Skipping pre-BA speculation: insufficient memory or disk headroom',flush=True)
        yield;return
    folder=cache_dir(work);folder.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder,prefix='.building-') as tmp, (work/'logs/mesh_pose_overlap.log').open('a') as log:
        cmd=[os.environ.get('MESH_PYTHON',sys.executable),'-u',str(Path(__file__).resolve()),str(work),tmp]
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        children=[child]
        try:
            dvp=Path(os.environ.get('DVP_ROOT',PROJECT/'dvp-openmvs'))/'l3'
            runtime=Path(os.environ.get('DVP_RUNTIME',dvp.parent/'runtime'))
            sys.path.insert(0,str(dvp))
            from dvp_driver import environment
            env=environment(os.environ['CUDA_VISIBLE_DEVICES'])
            env.update(PYTHONPATH=os.environ.get('DVP_MOGE_ROOT',str(dvp.parent/'vendor/MoGe')),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
            with (work/'logs/dvp_pose_overlap.log').open('a') as dvplog:
                try:children.append(subprocess.Popen([os.environ.get('DVP_PYTHON',str(runtime/'venv/bin/python')),'-u',str(dvp/'moge_overlap.py'),str(work)],env=env,stdout=dvplog,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL))
                except OSError as exc:print('DVP prefetch unavailable:',exc,file=dvplog)
            yield
        finally:
            for proc in children:
                if proc.poll() is None:proc.terminate()
            for proc in children:
                try:proc.wait(timeout=8)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            print('CPU BA ended; completed image predictions remain available. Exit codes',[p.returncode for p in children],file=log)



if __name__=='__main__':precompute(Path(sys.argv[1]),Path(sys.argv[2]))
