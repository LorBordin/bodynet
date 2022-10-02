import tensorflow as tf

def load_RGB(imgPath):
    """ 
        Loads png image and returns a normalised tensor.

        Parameters
        ----------
        imgPath: str
            Path to image.
        
        Returns
        -------
        img: tf.tensor
            Normalised image tensor in format tf.float32.
    """

    img = tf.io.read_file(imgPath)
    img = tf.io.decode_png(img, channels=3)
    img = tf.cast(img, dtype=tf.float32) / 127.5 - 1
    return img