from pathlib import Path
import os,json,time,subprocess,shutil,traceback
base=Path('/mnt/nuplan/l3data-reconstruction-bingxing/tem-test/colmap-ba-dvp-final/2026_04_25_14_32_57_dlp_pilotGtParser/after')
root=base/'dvp_sparse_moge_20260924';repo=Path('/mnt/zhoukaixuan_workspace/code/DVP-MVS');prepared=base/'dvp_openmvs_rerun_2x/prepared'
fusion=root/'10_openmvs_fusion';fusion.mkdir(exist_ok=True);(root/'logs').mkdir(exist_ok=True)
if not (root/'03_converted').exists():(root/'03_converted').symlink_to(prepared/'03_converted',target_is_directory=True)
ranges=root/'sparse_depth_ranges_openmvs.json'
scene=prepared/'08_openmvs_input/global/scene.mvs';neighbors=prepared/'08_openmvs_input/global/neighbors_diverse_pm20_top20.txt'
env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
runtime='/mnt/nuplan/gsapro-main-test/.deps/colmap-runtime'
env['LD_LIBRARY_PATH']=':'.join(['/mnt/nuplan/l3data-reconstruction-bingxing/colmap+openmvs/openmvs_runtime']+[runtime+p for p in ['/usr/lib','/usr/lib/x86_64-linux-gnu','/usr/lib/x86_64-linux-gnu/atlas','/lib/x86_64-linux-gnu']])+':'+env.get('LD_LIBRARY_PATH','')
commands=[];status={};started=time.monotonic()
def run(cmd,name):
 commands.append(cmd);(root/'openmvs_masked_commands.json').write_text(json.dumps(commands,indent=2));t=time.monotonic()
 print('START',name,flush=True)
 with (root/'logs'/name).open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 status[name+'_seconds']=time.monotonic()-t;print('DONE',name,status[name+'_seconds'],flush=True)
try:
 run(['python3','-c','import cv2,runpy,sys;cv2.setNumThreads(1);p=sys.argv.pop(1);sys.path.insert(0,str(__import__("pathlib").Path(p).parent));runpy.run_path(p,run_name="__main__")',str(repo/'mvs_process/convert_dvp_to_openmvs_dmap.py'),'--run',str(root/'09_dvp'),'--source',str(prepared/'03_converted'),'--scene',str(scene),'--images-txt',str(prepared/'07_sparse/triangulated_text/images.txt'),'--neighbors',str(neighbors),'--output',str(fusion/'dmaps'),'--depth-ranges-json',str(ranges)],'masked-dmap-convert.log')
 assert len(list((fusion/'dmaps').glob('depth*.dmap')))==707
 cmd=['/mnt/nuplan/open-source-projects/openMVS/install/bin/OpenMVS/DensifyPointCloud','-w',str(fusion/'dmaps'),'-i',str(scene),'-o',str(fusion/'openmvs_filtered_fused.mvs'),'--dense-config-file',str(repo/'mvs_process/openmvs_dvp_depth_filter.cfg'),'--mask-path',str(root/'06_masks/effective_masks'),'--ignore-mask-label','255','--view-neighbors-file',str(neighbors),'--cuda-device','-2','--max-threads','8','--resolution-level','1','--max-resolution','960','--min-resolution','640','--sub-resolution-levels','2','--number-views','20','--number-views-fuse','2','--iters','4','--geometric-iters','0','--fusion-mode','0','--fusion-filter','2','--fusion-depth-diff-threshold','0.010','--fusion-reprojection-threshold','2.5','--postprocess-dmaps','1','--estimate-colors','2','--estimate-normals','2','--tower-mode','0','--estimate-roi','0','--crop-to-roi','0','--filter-point-cloud','0','--remove-dmaps','0','--archive-type','2','-v','3']
 run(cmd,'masked-openmvs-fusion.log')
 assert (fusion/'openmvs_filtered_fused.ply').stat().st_size>0;status['returncode']=0
except Exception:
 status['returncode']=1;status['error']=traceback.format_exc();print(status['error'],flush=True)
finally:
 status['total_seconds']=time.monotonic()-started;(root/'masked_fusion_status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
