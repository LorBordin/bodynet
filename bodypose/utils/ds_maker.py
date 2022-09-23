from calendar import c
from ..dataset.preprocessing import load_depth, load_RGB
from .coordinates import read_joints_coords_opt

from tensorflow.python.ops.image_ops import _ImageDimensions
import tensorflow as tf
import numpy as np


def _bytes_feature(value):
    """ Returns a bytes_list from a string / byte. """
    if isinstance(value, type(tf.constant(0))):
      value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_sample(img, c_kpts, c_cntrs):
    """ Returns an example of a TFRecord sample. """
    image_string = img.numpy().astype("uint8").tobytes()
    kpts_string = c_kpts.numpy().tobytes()
    cntrs_string = c_cntrs.numpy().tobytes()

    feature = {
        'centres': _bytes_feature(c_cntrs),
        'keypoints': _bytes_feature(c_kpts),
        'image_raw': _bytes_feature(image_string),
              }

    return tf.train.Example(features=tf.train.Features(feature=feature))



def create_dataset(imgPaths, outDir, target_size, n_splits=1):
    """ 
        Creates the TFRecords from raw images and labels (keypoints and centres coordinates).

		Params
		------
		imgPaths: [str]
			List with the image paths.
		outDir: str
			Output directory.
		target_size: tuple
			Target image size (H, W).
		n_splits: int
			Number of different files in which the dataset has to be divided. Default to 1.
    """
    print(f"[INFO] Found {len(imgPath)} files.")
    np.random.shuffle(imgPath)
    idxs = np.random.randint(0, n_splits, len(imgPath))

    for idx in range(n_splits):
      
        print(f"[INFO] Creating Dataset {idx+1}/{n_splits}. ", end="")
        outSplitPath = outDir.replace(".tfrecords", f"_{idx}.tfrecords")
        with tf.io.TFRecordWriter(outSplitPath) as f:
            
            split = imgPath[idxs==idx]
            for i, imgPath in enumerate(split):

                print(f"\r Processing file {i+1}/{len(split)}...", end="")

                img = load_RGB(imgPath)
                img = (img + 1) * 127.5
                height, width, _ = _ImageDimensions(img, 3)
                img = tf.image.resize(img, target_size)

                # get joints and centres coords
                kptsTTxtPath = imgPath.replace(".png", "_kpts.txt")
                cntrsTxtPath = imgPath.replace(".png", "_cntrs.txt")
                c_kpts = np.loadtxt(kptsTTxtPath).astype(np.float32)
                c_cntrs = np.loadtxt(cntrsTxtPath).astype(np.float32)
        
                tf_example = create_sample(img, c_kpts, c_cntrs)
                f.write(tf_example.SerializeToString())

      print()


            




