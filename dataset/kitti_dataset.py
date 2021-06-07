import numpy as np
import re
from .kitti_data_base import *
import os

class KittiDetectionDataset:
    def __init__(self,root_path):
        self.root_path = root_path
        self.velo_path = os.path.join(self.root_path,"velodyne")
        self.image_path = os.path.join(self.root_path,"image_2")
        self.calib_path = os.path.join(self.root_path,"calib")
        self.label_path = os.path.join(self.root_path,"label_2")

        self.all_ids = os.listdir(self.velo_path)

    def __len__(self):
        return len(self.all_ids)
    def __getitem__(self, item):

        name = str(item).zfill(6)

        velo_path = os.path.join(self.velo_path,name+'.bin')
        image_path = os.path.join(self.image_path, name+'.png')
        calib_path = os.path.join(self.calib_path, name+'.txt')
        label_path = os.path.join(self.label_path, name+".txt")

        P2,V2C = read_calib(calib_path)
        points = read_velodyne(velo_path,P2,V2C)
        image = read_image(image_path)
        labels,label_names = read_detection_label(label_path)
        labels[:,3:6] = cam_to_velo(labels[:,3:6],V2C)[:,:3]

        return P2,V2C,points,image,labels,label_names

class KittiTrackingDataset:
    def __init__(self,root_path,seq_id,ob_path = None,load_image=False,load_points=False,type=["Car"]):
        self.seq_name = str(seq_id).zfill(4)
        self.root_path = root_path
        self.velo_path = os.path.join(self.root_path,"velodyne",self.seq_name)
        self.image_path = os.path.join(self.root_path,"image_02",self.seq_name)
        self.calib_path = os.path.join(self.root_path,"calib",self.seq_name)
        self.pose_path = os.path.join(self.root_path, "pose", self.seq_name,'pose.txt')
        self.type = type

        self.all_ids = os.listdir(self.velo_path)
        calib_path = self.calib_path + '.txt'

        self.P2, self.V2C = read_calib(calib_path)
        self.poses = read_pose(self.pose_path)
        self.load_image = load_image
        self.load_points = load_points

        self.ob_path = ob_path

    def __len__(self):
        return len(self.all_ids)-1
    def __getitem__(self, item):

        name = str(item).zfill(6)

        velo_path = os.path.join(self.velo_path,name+'.bin')
        image_path = os.path.join(self.image_path, name+'.png')

        if self.load_points:
            points = read_velodyne(velo_path,self.P2,self.V2C)
        else:
            points = None
        if self.load_image:
            image = read_image(image_path)
        else:
            image = None

        if item in self.poses.keys():
            pose = self.poses[item]
        else:
            pose = None

        if self.ob_path is not None:
            ob_path = os.path.join(self.ob_path, name + '.txt')
            if not os.path.exists(ob_path):
                objects = np.zeros(shape=(0, 7))
                det_scores = np.zeros(shape=(0,))
            else:
                objects_list = []
                det_scores = []
                with open(ob_path) as f:
                    for each_ob in f.readlines():
                        infos = re.split(' ', each_ob)
                        if infos[0] in self.type:
                            objects_list.append(infos[8:15])
                            det_scores.append(infos[15])
                if len(objects_list)!=0:
                    objects = np.array(objects_list,np.float32)
                    objects[:, 3:6] = cam_to_velo(objects[:, 3:6], self.V2C)[:, :3]
                    det_scores = np.array(det_scores,np.float32)
                else:
                    objects = np.zeros(shape=(0, 7))
                    det_scores = np.zeros(shape=(0,))
        else:
            objects = np.zeros(shape=(0,7))
            det_scores = np.zeros(shape=(0,))

        return self.P2,self.V2C,points,image,objects,det_scores,pose