import os

movenet_thunder_path = os.sep.join(["models", "movenet_thunder_f16.tflite"])
movenet_lightning_path = os.sep.join(["models", "movenet_lightning_f16.tflite"])

KEYPOINT_DICT = {
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

MODELS_DICT = {
    "movenet_thunder"  : {"path": movenet_thunder_path, 
                          "image_size": 256,
                          "keypoints": KEYPOINT_DICT },
    "movenet_lightning": {"path": movenet_lightning_path,  
                          "image_size":192,
                          "keypoints": KEYPOINT_DICT} 
}

N_KPTS = len(KEYPOINT_DICT)