from dataset.kitti_dataset import KittiTrackingDataset
from dataset.kitti_data_base import velo_to_cam
from tracker.tracker import Tracker3D
import time
import tqdm
import os
from tracker.config import config
from tracker.box_op import *
import numpy as np

from evaluation_HOTA.scripts.run_kitti import eval_kitti

def track_one_seq(seq_id,
                  dataset_path,
                  detections_path,
                  tracking_type):
    """
    tracking one sequence
    Args:
        seq_id: int, the sequence id
        dataset_path: str, the tracking data set path
        detections_path: str, the detection results path
        tracking_type: str, object type, "Car", "Pedestrian", "Cyclists"

    Returns: dataset: KittiTrackingDataset
             tracker: Tracker3D
             all_time: float, all tracking time
             frame_num: int, num frames
    """
    detections_path += "/" + str(seq_id).zfill(4)

    tracker = Tracker3D(box_type="Kitti", tracking_features=False)
    dataset = KittiTrackingDataset(dataset_path, seq_id=seq_id, ob_path=detections_path,type=[tracking_type])

    all_time = 0
    frame_num = 0

    for i in range(len(dataset)):
        P2, V2C, points, image, objects, det_scores, pose = dataset[i]

        mask = det_scores>config.input_score
        objects = objects[mask]
        det_scores = det_scores[mask]

        start = time.time()

        tracker.tracking(objects[:,:7],
                             features=None,
                             scores=det_scores,
                             pose=pose,
                             timestamp=i)
        end = time.time()
        all_time+=end-start
        frame_num+=1

    return dataset, tracker, all_time, frame_num

def save_one_seq(dataset,
                 seq_id,
                 tracker,
                 save_path,
                 tracking_type):
    """
    saving tracking results
    Args:
        dataset: KittiTrackingDataset, Iterable dataset object
        seq_id: int, sequence id
        tracker: Tracker3D
        save_path: str,
        tracking_type: str,
    """
    tracks = tracker.post_processing(config.globally)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_name = os.path.join(save_path,str(seq_id).zfill(4)+'.txt')

    frame_first_dict = {}
    for ob_id in tracks.keys():
        track = tracks[ob_id]
        for frame_id in track.trajectory.keys():
            ob = track.trajectory[frame_id]
            if ob.updated_state is None:
                continue
            if ob.score<config.post_score:
                continue
            if frame_id in frame_first_dict.keys():
                frame_first_dict[frame_id][ob_id]=(np.array(ob.updated_state.T),ob.score)
            else:
                frame_first_dict[frame_id]={ob_id:(np.array(ob.updated_state.T),ob.score)}

    with open(save_name,'w+') as f:
        for i in range(len(dataset)):
            P2, V2C, points, image, _, _, pose = dataset[i]
            new_pose = np.mat(pose).I
            if i in frame_first_dict.keys():
                objects = frame_first_dict[i]

                for ob_id in objects.keys():
                    updated_state,score = objects[ob_id]

                    box_template = np.zeros(shape=(1,7))
                    box_template[0,0:3]=updated_state[0,0:3]
                    box_template[0,3:7]=updated_state[0,9:13]

                    box = register_bbs(box_template,new_pose)

                    box[:, 6] = -box[:, 6] - np.pi / 2
                    box[:, 2] -= box[:, 5] / 2
                    box[:,0:3] = velo_to_cam(box[:,0:3],V2C)[:,0:3]

                    box = box[0]

                    box2d = bb3d_2_bb2d(box,P2)

                    print('%d %d %s -1 -1 -10 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                          % (i,ob_id,tracking_type,box2d[0][0],box2d[0][1],box2d[0][2],
                             box2d[0][3],box[5],box[4],box[3],box[0],box[1],box[2],box[6],score),file = f)


def tracking_val_seq():
    dataset_path = "H:/data/tracking/training"            # the kitti tracking dataset root path
    detections_path = "data/point-rcnn/training"           # the detection results path
    save_path = 'evaluation/results/sha_key/data' # the results saving path

    tracking_type = "Car"

    os.makedirs(save_path,exist_ok=True)

    seq_list = [1,6,8,10,12,13,14,15,16,18,19]    # the tracking sequences

    all_time,frame_num = 0,0

    for id in tqdm.trange(len(seq_list)):
        seq_id = seq_list[id]
        dataset,tracker, this_time, this_num = track_one_seq(seq_id,dataset_path,detections_path,tracking_type)
        save_one_seq(dataset,seq_id,tracker,save_path,tracking_type)

        all_time+=this_time
        frame_num+=this_num

    print("Tracking time: ",all_time)
    print("Tracking frames: ", frame_num)
    print("Tracking FPS:", frame_num/all_time)
    print("Tracking ms:", all_time/frame_num)

    eval_kitti()

if __name__ == '__main__':
    tracking_val_seq()

