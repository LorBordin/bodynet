import tensorflow as tf
import numpy as np
import time

class BodyPoseNet():
    def __init__(self, model):
        
        self.interpreter = tf.lite.Interpreter(model_path=model["path"])
        self.interpreter.allocate_tensors()
        self.img_size = model["image_size"]
        self.n_keypoints = len(model["keypoints"].keys())

        
    def predict(self, img):
        """
        Returns the predicted keypoints coordinates (normalised) with scores

        parameters
        ----------
        img : np.array 
            input image (three channels)

        returns
        -------
        keypoints_with_scores : np.array
            17 keypoints coordinates (normalised to one) with scores

        time: float
            inference time in secs
        """
        
        # compute pct of padding
        H, W = img.shape[:2]
        
        if H>W:
            pad = .5 * (1 - W/H)
            x,  y = np.ones(self.n_keypoints), np.zeros(self.n_keypoints)
        else: 
            pad = .5 * (1 - H/W)
            x, y = np.zeros(self.n_keypoints), np.ones(self.n_keypoints)
        
        padding_bias = np.stack([x, y], axis=-1)
        padding_bias *= pad

        img = tf.expand_dims(img, axis=0)
        img = tf.image.resize_with_pad(img, self.img_size, self.img_size)
        img = tf.cast(img, tf.uint8)

        start = time.time()

        # TF Lite format expects tensor type of uint8 
        input_details = self.interpreter.get_input_details() 
        output_details = self.interpreter.get_output_details()
        self.interpreter.set_tensor(input_details[0]["index"], img)

        # Invoke inference
        self.interpreter.invoke()

        # Get the model predictions and subtract the bias
        keypoints_with_scores = self.interpreter.get_tensor(output_details[0]["index"])
        keypoints_with_scores[:,:2] -= padding_bias

        end = time.time()

        return keypoints_with_scores[0, 0], (end-start)