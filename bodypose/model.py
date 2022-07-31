import tensorflow as tf
import numpy as np
import time
import cv2

class BodyPoseNet():
    def __init__(self, model):
        
        self.interpreter = tf.lite.Interpreter(model_path=model["path"])
        self.interpreter.allocate_tensors()
        self.img_size = model["image_size"]

        self.keypoints = model["keypoints"]
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
        
        H, W = img.shape[:2]

        # image preprocessing
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
        keypoints_with_scores = self.interpreter.get_tensor(output_details[0]["index"])[0, 0]
        keypoints_with_scores = unpad_predictions(keypoints_with_scores, (H,W), self.n_keypoints)
        end = time.time()

        return keypoints_with_scores, (end-start)


def unpad_predictions(keypoints_with_scores, img_size, n_keypoints):
    """
    Converts the keypoints coordinates from padded image to the original image.

    Parameters
    ----------
    keypoints_with_scores : np.array (dtype: float)
        Normalised keypoints coordinates with scores in format (y, x, p).
    img_size : tuple (dtype: int)
        Image  height and width.
    n_keypoints : int
        Number of predicted keypoints.

    Returns
    -------
    keypoints_with_scores:
        Normalised keypoints coordinates with scores as pct of the 
        original height and width, in format (y, x, p).
    """
    H, W = img_size
    
    if H>W:
        pad = .5 * (1 - W/H)
        x,  y = np.ones(n_keypoints), np.zeros(n_keypoints)
        padding_bias = np.stack([y, x], axis=-1)
        padding_bias *= pad
        keypoints_with_scores[:,:2] -= padding_bias
        keypoints_with_scores[:, 1] *= H/W

    else: 
        pad = .5 * (1 - H/W) 
        x, y = np.zeros(n_keypoints), np.ones(n_keypoints)
        padding_bias = np.stack([y, x], axis=-1)
        padding_bias *= pad
        keypoints_with_scores[:,:2] -= padding_bias
        keypoints_with_scores[:, 0] *= W/H
    
    return keypoints_with_scores
