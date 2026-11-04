import pickle
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CAMERAS={'left_front_camera':0,'right_front_camera':1,'rear_camera':2,'left_rear_camera':3,'right_rear_camera':4,'front_camera_fov30':9,'front_camera_fov120':10}
class U(pickle.Unpickler):
    def find_class(self,m,n):return super().find_class(m.replace('numpy._core','numpy.core'),n)
