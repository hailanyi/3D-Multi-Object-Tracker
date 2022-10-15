## A guide to submit results to KITTI test set

Take the CasTrack as an example:

1. You need create a KITTI account.
2. You need prepare the test set detections of [CasA](https://drive.google.com/file/d/1LaousWNTldOV1IhdcGDRM_UGi5BFWDoN/view?usp=sharing), and organize the files as follows.
```
data
└── casa
       ├── testing
       |      ├──0000
       |      |    ├──000001.txt
       |      |    ├──....txt
       |      |    └──000153.txt
       |      ├──...
       |      └──0020
```
4. Modify the [casa config file](https://github.com/hailanyi/3D-Multi-Object-Tracker/blob/master/config/global/casa_mot.yaml) to adapt the test set:

```
dataset_path: "L:/data/kitti/tracking/testing" # training -> testing
detections_path: "data/casa/testing" # training -> testing

tracking_seqs:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] # test tracking seq

```
5. Modify the config file path in [kitti_3DMOT.py](https://github.com/hailanyi/3D-Multi-Object-Tracker/blob/master/kitti_3DMOT.py) line 164, and run the kitti_3DMOT.py:

```
    parser.add_argument('--cfg_file', type=str, default="config/global/casa_mot.yaml",
```
6. The tracking results are saved into evaluation/results/sha_key/data/, you need zip all the .txt files into a single .zip file. The file can be submitted to KITTI website based on your KITTI account.
