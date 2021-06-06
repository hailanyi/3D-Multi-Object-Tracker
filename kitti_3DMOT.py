from dataset.kitti_dataset import KittiTrackingDataset
from tracker.tracker import Tracker3D
import time
import tqdm
import os

def track_one_seq(seq_id,dataset_path,detections_path):

    detections_path += "/" + str(seq_id).zfill(4)

    tracker = Tracker3D(box_type="Kitti", tracking_features=False, tracking_bb_size=True)
    dataset = KittiTrackingDataset(dataset_path, seq_id=seq_id, ob_path=detections_path,type=["Car"])

    all_time = 0
    frame_num = 0

    for i in range(len(dataset)):
        P2, V2C, points, image, objects, det_scores, pose = dataset[i]

        mask = det_scores>0
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

    tracker.post_processing()

    return dataset,tracker, all_time, frame_num
        
def save_one_seq(dataset,seq_id,tracker,save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_name = os.path.join(save_path,str(seq_id).zfill(4)+'.txt')



def tracking_all_seq():
    dataset_path = "H:/数据集/traking/testing"

    detections_path = r"I:\projects\Experiments\2020\past-best\output-76\multi_frame\kitti_models\pv_rcnn\default\eval\epoch_76\test\default\final_result\data"

    save_path = 'results'

    seq_list = range(29)

    all_time,frame_num = 0,0

    for id in tqdm.trange(len(seq_list)):
        seq_id = seq_list[id]
        dataset,tracker, this_time, this_num = track_one_seq(seq_id,dataset_path,detections_path)
        save_one_seq(dataset,seq_id,tracker,save_path)

        all_time+=this_time
        frame_num+=this_num

    print("Tracking time: ",all_time)
    print("Tracking frames: ", frame_num)
    print("Tracking FPS:", frame_num/all_time)
    print("Tracking ms:", all_time/frame_num)

if __name__ == '__main__':
    tracking_all_seq()

