import os

movenet_thunder_path = os.sep.join(["models", "movenet_thunder_f16.tflite"])
movenet_lightning_path = os.sep.joint(["models", "movenet_lightning_f16.tflite"])

MODELS_DICT = {
    "movenet_thunder"  : {"path": movenet_thunder_path, 
                          "img_size": 256 },
    "movenet_lightning": {"path": movenet_lightning_path,  
                          "img_size":192} 
}

