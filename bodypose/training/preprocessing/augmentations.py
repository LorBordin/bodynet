from tensorflow.python.ops.image_ops import _ImageDimensions
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
import tensorflow.keras.layers as L
import tensorflow as tf


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
    def __init__(self, probability, keypoints_idxs):
        self.probability = probability
        self.keypoints_idxs = keypoints_idxs

    def augment(self, rgb, coords):
        p = tf.random.uniform(shape=(), maxval=1)
        if p > self.probability:
            return rgb, coords
        rgb = tf.reverse(rgb, (1,))
        coords_x, coords_y = coords[:, 0], coords[:, 1]
        coords_x = 1 - coords_x
        coords = tf.stack([coords_x, coords_y], axis=1)
        return rgb, tf.gather(coords, self.keypoints_idxs)


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

    def augment(self, rgb, coords):
        
        shift = tf.random.uniform(shape=(), minval=-self.max_shift_range, maxval=self.max_shift_range)
        _, width, _ = _ImageDimensions(rgb, 3)
        width = tf.cast(width, tf.float32)
        pixel_shift = tf.cast(shift * width, tf.int32)
        shift = tf.cast(pixel_shift, tf.float32) / width

        rgb = tf.roll(rgb, shift=pixel_shift, axis=1)
        
        coords_x, coords_y = coords[:, 0], coords[:, 1]
        coords_x += shift
        return rgb, tf.stack([coords_x, coords_y], axis=1)


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

    def augment(self, rgb, coords):

        shift = tf.random.uniform(shape=(), minval=-self.max_shift_range, maxval=self.max_shift_range)
        height, _, _ = _ImageDimensions(rgb, 3)
        height = tf.cast(height, tf.float32)
        pixel_shift = tf.cast(shift * height, tf.int32)
        shift = tf.cast(pixel_shift, tf.float32) / height

        rgb = tf.roll(rgb, shift=pixel_shift, axis=0)
        
        coords_x, coords_y = coords[:, 0], coords[:, 1]
        coords_y += shift
        return rgb, tf.stack([coords_x, coords_y], axis=1)


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

    def augment(self, rgb, depth, coords):
        rgb = self.model(rgb)
        return rgb, depth, coords


def create_dilation_model(img_size):
    """ 
        Returns a model to be used in preprocessing that perform an opening operation to 
        the Mask in order to include a bigger portion of the RGB image. 
    """
    img_input = L.Input(img_size + (3,))
    dep_input = L.Input(img_size + (1,))
    mask = L.Lambda(lambda x: 1. - tf.cast(tf.math.equal(x, 0), dtype=tf.float32))(dep_input)
    mask =   L.MaxPool2D(pool_size=(20, 20), strides=(1, 1), padding='same')(mask)
    masked = L.Multiply()([img_input, mask])

    return Model(inputs=[img_input, dep_input], outputs=masked)