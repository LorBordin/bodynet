import os

movenet_thunder_path = os.sep.join(["models", "movenet_thunder_f16.tflite"])
movenet_lightning_path = os.sep.join(["models", "movenet_lightning_f16.tflite"])

COCO_KEYPOINT_DICT = {
    'nose': 0,            
    'left_eye': 1, 
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

MPII_KEYPOINT_DICT = {
    'right_ankle': 0, 
    'right_knee': 1, 
    'right_hip': 2, 
    'left_hip': 3, 
    'left_knee':4, 
    'left_ankle':5, 
    'pelvis': 6, 
    'thorax': 7, 
    'upper_neck': 8, 
    'head_top': 9, 
    'right_wrist':10, 
    'right_elbow': 11, 
    'right_shoulder': 12,
    'left_shoulder': 13, 
    'left_elbow': 14, 
    'left_wrist': 15
    }

MODELS_DICT = {
    "movenet_thunder"  : {"path": movenet_thunder_path, 
                          "image_size": 256,
                          "keypoints": COCO_KEYPOINT_DICT },
    "movenet_lightning": {"path": movenet_lightning_path,  
                          "image_size":192,
                          "keypoints": COCO_KEYPOINT_DICT} 
}

COCO_KEYPOINT_IDXS = [
    COCO_KEYPOINT_DICT['nose'], 
    COCO_KEYPOINT_DICT['right_eye'],
    COCO_KEYPOINT_DICT['left_eye'],
    COCO_KEYPOINT_DICT['right_ear'],
    COCO_KEYPOINT_DICT['left_ear'],
    COCO_KEYPOINT_DICT['right_shoulder'],
    COCO_KEYPOINT_DICT['left_shoulder'],
    COCO_KEYPOINT_DICT['right_elbow'],
    COCO_KEYPOINT_DICT['left_elbow'],
    COCO_KEYPOINT_DICT['right_wrist'],
    COCO_KEYPOINT_DICT['left_wrist'],
    COCO_KEYPOINT_DICT['right_hip'],
    COCO_KEYPOINT_DICT['left_hip'],
    COCO_KEYPOINT_DICT['right_knee'],
    COCO_KEYPOINT_DICT['left_knee'],
    COCO_KEYPOINT_DICT['right_ankle'],
    COCO_KEYPOINT_DICT['left_ankle']
    ]


MPII_KEYPOINT_IDXS = [
    MPII_KEYPOINT_DICT['left_ankle'], 
    MPII_KEYPOINT_DICT['left_knee'],
    MPII_KEYPOINT_DICT['left_hip'], 
    MPII_KEYPOINT_DICT['right_hip'],
    MPII_KEYPOINT_DICT['right_knee'],
    MPII_KEYPOINT_DICT['right_ankle'], 
    MPII_KEYPOINT_DICT['pelvis'], 
    MPII_KEYPOINT_DICT['thorax'], 
    MPII_KEYPOINT_DICT['upper_neck'], 
    MPII_KEYPOINT_DICT['head_top'], 
    MPII_KEYPOINT_DICT['left_wrist'], 
    MPII_KEYPOINT_DICT['left_elbow'], 
    MPII_KEYPOINT_DICT['left_shoulder'],
    MPII_KEYPOINT_DICT['right_shoulder'], 
    MPII_KEYPOINT_DICT['right_elbow'],
    MPII_KEYPOINT_DICT['right_wrist'],
]