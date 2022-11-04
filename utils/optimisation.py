from ..postprocessing import get_joint_coords
from tensorflow.keras import layers as L
from tensorflow.keras.models import Model
import tensorflow as tf
import pathlib
import sys

def convert_to_tflite(model, 
                      out_model_name,
                      input_size, 
                      precision="int8", 
                      add_postprocessing=True, 
                      num_joints=17, 
                      tflite_models_dir="./saved_models/tflite_models/"):

    if add_postprocessing:
        img_input = L.Input((input_size, input_size, 3), dtype=tf.float32)
        depth_input = L.Input((input_size, input_size, 1), dtype=tf.float32)
        pred = model((img_input, depth_input))
        post = L.Lambda(lambda x: get_joint_coords(x, num_joints))(pred)
        model = Model(inputs=(img_input, depth_input), outputs=post)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    if precision=="int8":
        print("[INFO] Setting precision to INT8.")
    elif precision=="fp16":
        print("[INFO] Setting precision to FLOAT16.")
        converter.target_spec.supported_types = [tf.float16]
    else:
        print("[INFO] Unknown precision.")
        sys.exit()
    
    tflite_model = converter.convert()
    print("[INFO] Converting model...")

    tflite_models_dir = pathlib.Path(tflite_models_dir)
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_file = tflite_models_dir/out_model_name
    
    tflite_model_file.write_bytes(tflite_model)
    print(f"[INFO] Converted model saved @ {tflite_model_file}.")
