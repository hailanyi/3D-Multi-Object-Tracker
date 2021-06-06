import os
import pickle
import numpy as np

class WaymoDataset:
    def __init__(self,root_path,gt_info_path=None,pred_info_path=None,types=['Vehicle','Pedestrian','Cyclist']):
        self.root_path =root_path
        self.gt_info_path = gt_info_path
        self.pred_info_path = pred_info_path

        self.types=types

        with open(self.gt_info_path, 'rb') as f:
            self.gt_pickle_file = pickle.load(f)
        with open(self.pred_info_path, 'rb') as f:
            self.pred_pickle_file = pickle.load(f)

        self.len_gt = len(self.gt_pickle_file)
        self.len_pred = len(self.pred_pickle_file)

        self.step = self.len_gt//self.len_pred

    def __len__(self):
        return self.len_pred

    def __getitem__(self, item):

        item_pred = item

        item_gt = item_pred*self.step

        frame_gt = self.gt_pickle_file[item_gt]
        frame_pred = self.pred_pickle_file[item_pred]

        pc_info = frame_gt['point_cloud']

        sequence_name = pc_info['lidar_sequence']

        frame_id = int(frame_gt['frame_id'][-3:])


        lidar_path = os.path.join(self.root_path,sequence_name,('%04d.npy' % frame_id))

        lidar_points =  np.load(lidar_path)

        gt_boxes = np.array(frame_gt['annos']['gt_boxes_lidar'])
        gt_names = np.array(frame_gt['annos']['name'])

        mask_gt = np.zeros(shape=gt_names.shape)

        for type in self.types:
            mask_gt += (gt_names==type)

        mask_gt=mask_gt.astype(np.bool)

        gt_boxes=gt_boxes[mask_gt]
        gt_names=gt_names[mask_gt]


        pred_boxes = np.array(frame_pred['boxes_lidar'])
        pred_scores = np.array(frame_pred['score'])
        pred_names = np.array(frame_pred['name'])

        mask_pred = np.zeros(shape=pred_names.shape)
        for type in self.types:
            mask_pred += (pred_names==type)

        mask_pred = mask_pred.astype(np.bool)

        pred_boxes = pred_boxes[mask_pred]
        pred_scores = pred_scores[mask_pred]
        pred_names = pred_names[mask_pred]

        infos = {'points':lidar_points,
                 'gt_boxes':gt_boxes,
                 'gt_names':gt_names,
                 'pred_boxes':pred_boxes,
                 'pred_scores':pred_scores,
                 'pred_names':pred_names}

        return infos


if __name__ == '__main__':

    root = "/media/wuhai/My Passport/dataset/waymo/3D-Detection-Tracking/dataset-npy/waymo_processed_data_train_val_test"
    gt_info = "/media/wuhai/My Passport/dataset/waymo/3D-Detection-Tracking/dataset-npy/waymo_infos_val.pkl"
    pred_info = "/media/wuhai/My Passport/project_local/OpenPCDet/output/one_frame/waymo_models/voxel_rcnn/default/eval/epoch_3/val/default/result.pkl"
    data = WaymoDataset(root,gt_info,pred_info)

    for i in range(len(data)):
        infos = data[i]
