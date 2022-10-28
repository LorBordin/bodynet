from tensorflow.python.ops.image_ops import _ImageDimensions
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
import tensorflow.keras.layers as L
import tensorflow as tf
from config import KEYPOINT_DICT 

keypoints_idxs = [
    KEYPOINT_DICT['nose'], 
    KEYPOINT_DICT['right_eye'],
    KEYPOINT_DICT['left_eye'],
    KEYPOINT_DICT['right_ear'],
    KEYPOINT_DICT['left_ear'],
    KEYPOINT_DICT['right_shoulder'],
    KEYPOINT_DICT['left_shoulder'],
    KEYPOINT_DICT['right_elbow'],
    KEYPOINT_DICT['left_elbow'],
    KEYPOINT_DICT['right_wrist'],
    KEYPOINT_DICT['left_wrist'],
    KEYPOINT_DICT['right_hip'],
    KEYPOINT_DICT['left_hip'],
    KEYPOINT_DICT['right_knee'],
    KEYPOINT_DICT['left_knee'],
    KEYPOINT_DICT['right_ankle'],
    KEYPOINT_DICT['left_ankle']
    ]


class HorizontalFlip():
    """
        HorizontalFlip augmentation class. 
        Flips the image horizontaly with random uniform probability.

        Parameters
        ----------
        probability: float
            Flip probability.
        keypoints_idxs: list
            List of int that take into account the flip of the right/left keypoints.

    """
    def __init__(self, probability, keypoints_idxs=keypoints_idxs):
        self.probability = probability
        self.keypoints_idxs = keypoints_idxs

    def augment(self, rgb, c_kpts, c_cntrs):
        p = tf.random.uniform(shape=(), maxval=1)
        if p > self.probability:
            return rgb, c_kpts, c_cntrs
        rgb = tf.reverse(rgb, (1,))

        kpts_x, kpts_y = c_kpts[:, 0], c_kpts[:, 1]
        kpts_x = 1 - kpts_x
        c_kpts = tf.stack([kpts_x, kpts_y], axis=1)
        c_kpts = tf.gather(c_kpts, self.keypoints_idxs)

        cntrs_x, cntrs_y = c_cntrs[:, 0], c_cntrs[:, 1]
        cntrs_x = 1 - cntrs_x
        c_cntrs = tf.stack([cntrs_x, cntrs_y], axis=1)
        
        return rgb, c_kpts, c_cntrs


class HorizontalShift():
    """
        HorizontalShift augmentation class. 
        Shifts the image horizontaly by a random shift chosen uniformally picked in 
        the  interval [- max_shift, max_shift].

        Parameters
        ----------
        max_shift_range: float
            Max shift value in pct of the image.

    """
    def __init__(self, max_shift_range):
        self.max_shift_range = max_shift_range

    def augment(self, rgb, c_kpts, c_cntrs):
        
        shift = tf.random.uniform(shape=(), minval=-self.max_shift_range, maxval=self.max_shift_range)
        _, width, _ = _ImageDimensions(rgb, 3)
        width = tf.cast(width, tf.float32)
        pixel_shift = tf.cast(shift * width, tf.int32)
        shift = tf.cast(pixel_shift, tf.float32) / width

        rgb = tf.roll(rgb, shift=pixel_shift, axis=1)
        
        kpts_x, kpts_y = c_kpts[:, 0], c_kpts[:, 1]
        kpts_x += shift
        c_kpts = tf.stack([kpts_x, kpts_y], axis=1)

        cntrs_x, cntrs_y = c_cntrs[:, 0], c_cntrs[:, 1]
        cntrs_x += shift
        c_cntrs = tf.stack([cntrs_x, cntrs_y], axis=1)

        return rgb, c_kpts, c_cntrs


class VerticalShift():
    """
        VerticalShift augmentation class. 
        Shifts the image vertically by a random shift chosen uniformally picked in 
        the  interval [- max_shift, max_shift].

        Parameters
        ----------
        max_shift_range: float
            Max shift value in pct of the image.

    """
    def __init__(self, max_shift_range):
        self.max_shift_range = max_shift_range

    def augment(self, rgb, c_kpts, c_cntrs):

        shift = tf.random.uniform(shape=(), minval=-self.max_shift_range, maxval=self.max_shift_range)
        height, _, _ = _ImageDimensions(rgb, 3)
        height = tf.cast(height, tf.float32)
        pixel_shift = tf.cast(shift * height, tf.int32)
        shift = tf.cast(pixel_shift, tf.float32) / height

        rgb = tf.roll(rgb, shift=pixel_shift, axis=0)
        
        kpts_x, kpts_y = c_kpts[:, 0], c_kpts[:, 1]
        kpts_y += shift
        c_kpts = tf.stack([kpts_x, kpts_y], axis=1)

        cntrs_x, cntrs_y = c_cntrs[:, 0], c_cntrs[:, 1]
        cntrs_y += shift
        c_cntrs = tf.stack([cntrs_x, cntrs_y], axis=1)
        
        return rgb, c_kpts, c_cntrs


class AugmentationRGB():
    """
        AugmentationRGB class. Applies the following color transformations to the image:
        - Contrast
        - Brightness    

        Parameters
        ----------
        max_contrast: float
            Maximum contrast value. The default value is 0.2.
        max_brightness: float
            Maximum brightness value. The defaultvalue is  0.2.
    """
    def __init__(self, max_contrast=.2, max_brightness=.2):
        self.model = Sequential([
            L.RandomContrast(factor=max_contrast),
            L.RandomBrightness(factor=max_brightness, value_range=(-1,1)),
            ])

    def augment(self, rgb, c_kpts, c_cntrs):
        rgb = self.model(rgb)
        return rgb, c_kpts, c_cntrs
