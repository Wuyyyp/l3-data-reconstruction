import unittest
import numpy as np
from assemble_moving_objects import group_tracks,merge_frame,remove_component_duplicates,select_mask,reject_occluded_riders

class WholeEntitiesTests(unittest.TestCase):
    def test_nearer_static_occluder_rejects_hidden_track_not_foreground_mover(self):
        mask=np.zeros((20,20),bool);mask[5:15,5:15]=True
        entities={k:dict(entity_type='rider_vehicle') for k in ['far','near']}
        instances={'far':mask.copy(),'near':mask.copy()}
        removed=reject_occluded_riders(instances,entities,{'static':[4,4,12,12]},
                                       {'static':(10.,12.),'far':(15.,17.),'near':(6.,8.)},{'static'})
        self.assertEqual(removed,{'far':'static'});self.assertEqual(set(instances),{'near'})
    def test_rider_person_and_cycle_are_one_instance(self):
        world={7:dict(category='Non_motor_rider',positions=[[0,0,0],[2,0,0]]),20:dict(category='Pedestrian',positions=[[0,0,0],[3,0,0]])}
        tracks={k:dict(prompt=p,motion=dict(annotation_id=i,state='moving'),observations=[]) for k,p,i in [('human7','person',7),('bike7','motorcycle',7),('walker20','person',20)]}
        entities,mapping=group_tracks(tracks,world,1.)
        body=np.zeros((12,16),bool);body[1:6,3:7]=True
        bike=np.zeros_like(body);bike[6:10,1:10]=True
        walker=np.zeros_like(body);walker[1:10,13:15]=True
        result=merge_frame(dict(human7=body,bike7=bike,walker20=walker),entities,mapping)
        self.assertEqual(set(result),{'obj_7','obj_20'});self.assertEqual(entities['obj_7']['entity_type'],'rider_vehicle');self.assertEqual(entities['obj_20']['entity_type'],'person')
        np.testing.assert_array_equal(result['obj_7'],body|bike)

    def test_static_rider_parts_do_not_become_a_moving_person(self):
        world={7:dict(category='Non_motor_rider',positions=[[0,0,0],[.5,0,0]])}
        tracks={'body':dict(prompt='person',motion=dict(annotation_id=7,state='moving'),observations=[])}
        entities,mapping=group_tracks(tracks,world,1.)
        self.assertEqual(merge_frame({'body':np.ones((5,5),bool)},entities,mapping),{})

    def test_full_entity_consumes_only_unassociated_components(self):
        full=np.zeros((12,16),bool);full[1:10,1:10]=True
        body=np.zeros_like(full);body[2:5,3:7]=True
        instances={'obj_7':full,'raw_person':body,'obj_20':body.copy()}
        entities={'obj_7':dict(entity_type='rider_vehicle',motion=dict(annotation_id=7)),'raw_person':dict(entity_type='person',motion={}), 'obj_20':dict(entity_type='person',motion=dict(annotation_id=20))}
        consumed=remove_component_duplicates(instances,entities)
        self.assertEqual(consumed,{'raw_person':'obj_7'});self.assertIn('obj_20',instances)

    def test_joint_mask_keeps_accessory_and_rejects_neighbor(self):
        whole=np.zeros((12,20),bool);whole[1:10,1:9]=True;whole[4,5]=False
        neighbor=np.zeros_like(whole);neighbor[1:10,11:19]=True
        seed=np.zeros_like(whole);seed[1:4,2:5]=True
        best=select_mask([neighbor,whole],[.99,.9],[1,1,8,9],[[11,1,8,9]],seed)
        self.assertIsNotNone(best);np.testing.assert_array_equal(best[1],whole);self.assertFalse(best[1][4,5])

if __name__=='__main__':unittest.main()
