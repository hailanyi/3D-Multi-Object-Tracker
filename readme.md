# 3D Multi-Object Tracker
This project is developed for tracking multiple objects in 3D scene, and is a simplified
implementation of [paper](https://ieeexplore.ieee.org/abstract/document/9352500). The visualization code is from
[here](https://github.com/hailanyi/3D-Detection-Tracking-Viewer).
![](./doc/demo.gif)
## Features

* Fast: currently, the codes can achieve 700 FPS using only CPU (not include data op), can perform tracking 
on all kitti sequence in several seconds. 
* Support both online and global implementation. 
The overall framework of design is shown below:
![](./doc/framework.jpg)

## Kitti Results
Results on the Kitti tracking val seq [1,6,8,10,12,13,14,15,16,18,19] 
using second-iou and point-rcnn detections. We followed the HOTA metric, and tuned the parameters
 in this code by firstly considering the HOTA performance.
 
|Detector|HOTA  | DetA  |    AssA  |    DetRe  |   DetPr   |  AssRe  |   AssPr   |  LocA  |   MOTA  |
|---|---|---|---|---|---|---|---|---|---|
|second-iou	|78.783	|74.465	|83.618	|80.646	|84.742	|89.017	|88.595	|88.651	|85.082|
|point-rcnn	|78.885   | 75.764  |  82.41   |  83.567 |   82.04  |  87.2    |  87.59  |   87.305| 88.388|


## Prepare data 
You can download the Kitti tracking pose data from [here](https://drive.google.com/drive/folders/1Vw_Mlfy_fJY6u0JiCD-RMb6_m37QAXPQ?usp=sharing), and
you can find the point-rcnn and second-iou detections from [here](https://drive.google.com/file/d/1DQ1goFvfHRTYdfy5UqWpKRMxU_dsFi8i/view?usp=sharing).

To run this code, you should organize Kitti tracking dataset as below:
```
# Kitti Tracking Dataset       
└── kitti_tracking
       ├── testing 
       |      ├──calib
       |      |    ├──0000.txt
       |      |    ├──....txt
       |      |    └──0028.txt
       |      ├──image_02
       |      |    ├──0000
       |      |    ├──....
       |      |    └──0028
       |      ├──pose
       |      |    ├──0000
       |      |    |    └──pose.txt
       |      |    ├──....
       |      |    └──0028
       |      |         └──pose.txt
       |      ├──label_02
       |      |    ├──0000.txt
       |      |    ├──....txt
       |      |    └──0028.txt
       |      └──velodyne
       |           ├──0000
       |           ├──....
       |           └──0028      
       └── training # the structure is same as testing set
              ├──calib
              ├──image_02
              ├──pose
              ├──label_02
              └──velodyne 
```
Detections
```
└── point-rcnn
       ├── training
       |      ├──0000
       |      |    ├──000001.txt
       |      |    ├──....txt
       |      |    └──000153.txt
       |      ├──...
       |      └──0020
       └──testing 
```

## Requirements
```
python3
numpy
opencv
```

## Quick start
* Please modify the dataset path and detections path in the [yaml file](./config/point_rcnn_mot.yaml) 
to your own path.
* Then run ``` python3 kitti_3DMOT.py config/point_rcnn_mot.yaml``` 
* The results are automatically saved to ```evaluation\results\sha_key\data```, and 
evaluated by HOTA metrics.

## Notes
The evaluation codes are copied from [Kitti](https://github.com/JonathonLuiten/TrackEval).
