import json,tempfile,unittest
from pathlib import Path
import numpy as np
from motion_mask_utils import (PROMPTS,annotation_world_tracks,camera_matrices,category_group,compatible,
    consistent_annotation,depth_world_centroid,motion_decision,pack_mask,unpack_mask)

class MotionTests(unittest.TestCase):
    def test_threshold_is_strict_and_not_path_length(self):
        self.assertEqual(motion_decision([[0,0,0],[1,0,0]])['state'],'static')
        self.assertEqual(motion_decision([[0,0,0],[1.001,0,0]])['state'],'moving')
        self.assertEqual(motion_decision([[0,0,0],[.8,0,0],[0,0,0],[.8,0,0]])['state'],'static')
        self.assertEqual(motion_decision([[0,0,0],[2,0,0],[0,0,0]])['state'],'moving')
        self.assertEqual(motion_decision([[0,0,0],[0,0,1.1]])['state'],'moving')

    def test_ego_motion_removed_and_unselected_classes_included(self):
        infos=[]
        for step in range(3):
            T=np.eye(4);T[0,3]=step*3
            objects=[]
            for identity,name,world_x in [(1,'Car',10),(2,'Pedestrian',10+step*.6),(3,'Non_motor_rider',20+step)]:
                objects.append(dict(id=identity,type=name,location=[world_x-T[0,3],0,0],size=[1,1,1],rotation=[0,0,0]))
            infos.append(dict(ego2global_transformation_matrix_camera_reference_offset_optimized=T.tolist(),objects=objects))
        tracks=annotation_world_tracks(infos)
        self.assertEqual(set(tracks),{1,2,3})
        self.assertEqual(motion_decision(tracks[1]['positions'])['state'],'static')
        self.assertEqual(motion_decision(tracks[2]['positions'])['state'],'moving')
        self.assertEqual(motion_decision(tracks[3]['positions'])['state'],'moving')

    def test_no_metric_evidence_is_not_moving(self):
        self.assertEqual(motion_decision([])['state'],'uncertain')
        self.assertEqual(motion_decision([[0,0,0]])['state'],'uncertain')
        self.assertEqual(motion_decision([[0,0,0],[.6,0,0],[1.2,0,0]],source='lidar_surface_centroid',radii=[.3]*3)['state'],'uncertain')
        self.assertEqual(motion_decision([[0,0,0],[2,0,0],[4,0,0]],source='lidar_surface_centroid',radii=[.3]*3)['state'],'moving')

    def test_category_coverage_and_identity_switch(self):
        for c in ['Pedestrian','Person']:self.assertEqual(category_group(c),'person')
        for c in ['Non_motor_rider','Bicycle','Motorcycle','electric scooter','electric bicycle','Tricycle']:self.assertEqual(category_group(c),'two_wheeler')
        self.assertIn('tricycle',PROMPTS)
        for c in ['Huge_vehicle','Vehicle_else','Firetruck']:self.assertEqual(category_group(c),'vehicle')
        self.assertTrue(compatible('person','Non_motor_rider'));self.assertFalse(compatible('person','Suv'))
        self.assertEqual(category_group('Road_barrier'),'other')
        self.assertIsNone(consistent_annotation([dict(annotation_id=i) for i in [1,1,2,2]]))
        self.assertEqual(consistent_annotation([dict(annotation_id=7)]*3),7)

    def test_mixed_foreground_background_lidar_is_uncertain(self):
        result=motion_decision([[0,0,0],[35,0,0],[73,0,0]],source='lidar_surface_centroid',radii=[33,34,35])
        self.assertEqual(result['state'],'uncertain')
        self.assertEqual(result['reason'],'incoherent_lidar_surface_extent')
        self.assertEqual(motion_decision([[0,0,0],[2,0,0],[4,0,0]],source='lidar_surface_centroid')['state'],'uncertain')

    def test_lossless_mask_preserves_holes_and_bounds(self):
        mask=np.zeros((13,19),bool);mask[2:10,3:16]=True;mask[4:8,6:10]=False
        bbox,bits=pack_mask(mask);np.testing.assert_array_equal(unpack_mask(bbox,bits,mask.shape),mask)
        with self.assertRaises(ValueError):unpack_mask([0,0,20,13],bits,mask.shape)

    def test_lidar_world_coordinates(self):
        mask=np.ones((30,30),bool);depth=np.full((30,30),10.,np.float32)
        K=np.array([[100,0,15],[0,100,15],[0,0,1.]])
        C=np.eye(4);C[0,3]=5
        a=depth_world_centroid(mask,depth,K,np.eye(4));b=depth_world_centroid(mask,depth,K,C)
        np.testing.assert_allclose(np.array(b['world_position'])-a['world_position'],[5,0,0])
        self.assertIsNone(depth_world_centroid(mask,np.full(mask.shape,np.nan),K,C))

    def test_exports_all_movers_without_selected_id_filter(self):
        import cv2
        from segment_moving_objects import export_camera
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'input';source.mkdir();n=3;h,w=24,32
            video=cv2.VideoWriter(str(source/'input.avi'),cv2.VideoWriter_fourcc(*'MJPG'),10,(w,h))
            for _ in range(n):video.write(np.zeros((h,w,3),np.uint8))
            video.release()
            for folder in ['logs','review']:(root/folder).mkdir()
            masks={};tracks={};expected=np.zeros((h,w),bool)
            for pi,sid,state,y,x in [(0,1,'moving',2,2),(1,7,'moving',2,12),(5,9,'moving',12,2),(0,4,'static',12,12),(3,97,'uncertain',12,22)]:
                key=f'p{pi}_obj_{sid}';mask=np.zeros((h,w),bool);mask[y:y+5,x:x+5]=True;mask[y+2,x+2]=False
                masks[key]=mask;tracks[key]=dict(prompt=PROMPTS[pi],motion=dict(state=state))
                if state=='moving':expected|=mask
            for pi in range(len(PROMPTS)):
                p=root/'candidates/test'/f'p{pi}';p.mkdir(parents=True)
                arrays={}
                for key,mask in masks.items():
                    if key.startswith(f'p{pi}_'):arrays[key+'_bbox'],arrays[key+'_bits']=pack_mask(mask)
                for fi in range(n):np.savez_compressed(p/f'{fi:06d}.npz',**arrays)
            result=export_camera(root,source,dict(camera='test',height=h,width=w,file='input.avi'),dict(frames=n,fps=10,displacement_m=1.),tracks,PROMPTS)
            self.assertEqual(result['instance_frames'],9)
            for fi in range(n):
                np.testing.assert_array_equal(cv2.imread(str(root/'union-masks/test'/f'{fi:06d}.png'),0)>0,expected)
                with np.load(root/'masks/test'/f'{fi:06d}.npz') as a:
                    self.assertEqual({x[:-5] for x in a.files if x.endswith('_bbox')},{'p0_obj_1','p1_obj_7','p5_obj_9'})
            cap=cv2.VideoCapture(str(root/'videos/test.mp4'));self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),n);cap.release()

if __name__=='__main__':unittest.main()
