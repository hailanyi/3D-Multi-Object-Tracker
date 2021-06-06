from .trajectory import Trajectory
from .config import config
import numpy as np

class Tracker3D:
    def __init__(self,tracking_bb_size=True,
                    tracking_features=False,
                    bb_as_features=False,
                    box_type='Kitti'):

        self.current_timestamp = None
        self.current_pose = None
        self.current_bbs = None
        self.current_features = None
        self.tracking_bb_size = tracking_bb_size
        self.tracking_features = tracking_features
        self.bb_as_features = bb_as_features
        self.box_type = box_type

        self.label_seed = 0

        self.active_trajectories = {}
        self.dead_trajectories = {}

    def tracking(self,bbs = None,
                 features = None,
                 scores = None,
                 pose = None,
                 timestamp = None
                 ):


        self.current_bbs = bbs
        self.current_features = features
        self.current_scores = scores
        self.current_pose = pose
        self.current_timestamp = timestamp

        self.trajectores_prediction()

        if bbs is None:
            return np.zeros(shape=(0))
        else:
            if len(bbs) == 0:
                return np.zeros(shape=(0))

            else:
                self.current_bbs = self.convert_bbs_type(bbs,self.box_type)
                self.current_bbs = self.register_bbs(self.current_bbs,self.current_pose)
                ids = self.association()
                self.trajectories_update_init(ids)

                return ids

    def convert_bbs_type(self,boxes,input_box_type):

        boxes = np.array(boxes)

        assert input_box_type in ["Kitti", "OpenPCDet", "Waymo"], 'unsupported input box type!'

        if input_box_type in ["OpenPCDet", "Waymo"]:
            return boxes

        if input_box_type == "Kitti":  # (h,w,l,x,y,z,yaw) -> (x,y,z,l,w,h,yaw)

            t_id = boxes.shape[1]//7
            new_boxes = np.zeros(shape=boxes.shape)
            new_boxes[:, :] = boxes[:, :]
            for i in range(t_id):
                b_id = i*7
                new_boxes[:, b_id+0:b_id+3] = boxes[:, b_id+3:b_id+6]
                new_boxes[:, b_id+3] = boxes[:, b_id+2]
                new_boxes[:, b_id+4] = boxes[:, b_id+1]
                new_boxes[:, b_id+5] = boxes[:, b_id+0]
                new_boxes[:, b_id+6] = (np.pi - boxes[:, b_id+6]) + np.pi / 2
                new_boxes[:, b_id+2] += boxes[:, b_id+0] / 2
            return new_boxes

    def get_registration_angle(self,mat):

        cos_theta=mat[0,0]
        sin_theta=mat[1,0]

        if  cos_theta < -1:
            cos_theta = -1
        if cos_theta > 1:
            cos_theta = 1

        theta_cos = np.arccos(cos_theta)

        if sin_theta >= 0:
            return theta_cos
        else:
            return 2 * np.pi - theta_cos

    def register_bbs(self,boxes,pose):

        ang = self.get_registration_angle(pose)

        t_id = boxes.shape[1] // 7
        ones = np.ones(shape=(boxes.shape[0],1))
        for i in range(t_id):
            b_id = i * 7
            box_xyz = boxes[:,b_id:b_id+3]
            box_xyz1 = np.concatenate([box_xyz,ones],-1)
            box_world = np.matmul(box_xyz1,pose.T)
            boxes[:,b_id:b_id+3] = box_world[:,0:3]
            boxes[:, b_id+6] += ang
        return boxes

    def trajectores_prediction(self):

        if len(self.active_trajectories) == 0 :
            return
        else:
            dead_track_id = []

            for key in self.active_trajectories.keys():
                if self.active_trajectories[key].consecutive_missed_num>=config.max_prediction_num:
                    dead_track_id.append(key)
                    continue
                if len(self.active_trajectories[key])-self.active_trajectories[key].consecutive_missed_num == 1 \
                    and len(self.active_trajectories[key])>= config.max_prediction_num_for_new_object :
                    dead_track_id.append(key)
                self.active_trajectories[key].state_prediction(self.current_timestamp)

            for id in dead_track_id:
                tra = self.active_trajectories.pop(id)
                self.dead_trajectories[id]=tra

    def compute_cost_map(self):
        all_ids = []

        all_predictions = []
        all_detections = []

        for key in self.active_trajectories.keys():
            all_ids.append(key)
            state = np.array(self.active_trajectories[key].trajectory[self.current_timestamp].predicted_state)
            state = state.reshape(-1)

            pred_score = np.array([self.active_trajectories[key].trajectory[self.current_timestamp].prediction_score])

            state = np.concatenate([state,pred_score])
            all_predictions.append(state)

        for i in range(len(self.current_bbs)):
            box = self.current_bbs[i]
            features = None
            if self.current_features is not None:
                features = self.current_features[i]
            score = self.current_scores[i]
            label=1
            new_tra = Trajectory(init_bb=box,
                 init_features=features,
                 init_score=score,
                 init_timestamp=self.current_timestamp,
                 label=label,
                 tracking_bb_size=self.tracking_bb_size,
                 tracking_features=self.tracking_features,
                 bb_as_features=self.bb_as_features)

            state = new_tra.trajectory[self.current_timestamp].predicted_state
            state = state.reshape(-1)
            all_detections.append(state)

        all_detections = np.array(all_detections)
        all_predictions = np.array(all_predictions)

        det_len = len(all_detections)
        pred_len = len(all_predictions)

        all_detections = all_detections.reshape((det_len,1,-1))
        all_predictions = all_predictions.reshape((1,pred_len,-1))

        all_detections = np.tile(all_detections,(1,pred_len,1))
        all_predictions = np.tile(all_predictions,(det_len,1,1))

        dis = (all_detections[...,0:3]-all_predictions[...,0:3])**2
        dis = np.sqrt(dis.sum(-1))

        cost = dis*all_predictions[...,-1]

        return cost,all_ids

    def association(self):

        if len(self.active_trajectories) == 0:
            ids = []
            for i in range(len(self.current_bbs)):
                ids.append(self.label_seed)
                self.label_seed+=1
            return ids
        else:
            ids = []
            cost_map, all_ids = self.compute_cost_map()
            for i in range(len(self.current_bbs)):
                min = np.min(cost_map[i])
                arg_min = np.argmin(cost_map[i])

                if min<config.assign_threshold:
                    ids.append(all_ids[arg_min])
                    cost_map[:,arg_min] = 100000
                else:
                    ids.append(self.label_seed)
                    self.label_seed+=1
            return ids


    def trajectories_update_init(self,ids):
        assert len(ids) == len(self.current_bbs)

        for i in range(len(self.current_bbs)):
            label = ids[i]
            box = self.current_bbs[i]
            features = None
            if self.current_features is not None:
                features = self.current_features[i]
            score = self.current_scores[i]

            if label in self.active_trajectories.keys():
                track = self.active_trajectories[label]
                track.state_update(
                     bb=box,
                     features=features,
                     score=score,
                     timestamp=self.current_timestamp)
            else:
                new_tra = Trajectory(init_bb=box,
                                     init_features=features,
                                     init_score=score,
                                     init_timestamp=self.current_timestamp,
                                     label=label,
                                     tracking_bb_size=self.tracking_bb_size,
                                     tracking_features=self.tracking_features,
                                     bb_as_features=self.bb_as_features)
                self.active_trajectories[label] = new_tra


    def post_processing(self):

        tra = {}
        for key in self.dead_trajectories.keys():
            track = self.dead_trajectories[key]
            track.filtering()
            tra[key] = track
        for key in self.active_trajectories.keys():
            track = self.active_trajectories[key]
            track.filtering()
            tra[key] = track

        return tra




