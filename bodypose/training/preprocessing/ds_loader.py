#from .preprocessing import create_joint_mask
from .preprocessing import create_density_maps
from .preprocessing import sum_density_maps
#from .preprocessing import mask_img
from .preprocessing import crop_roi
from config import MPII_KEYPOINT_DICT

N_KPTS = len(MPII_KEYPOINT_DICT)

import random
import numpy as np
import tensorflow as tf
from imutils import paths
from tensorflow.python.ops.image_ops import _ImageDimensions

AUTOTUNE = tf.data.AUTOTUNE
IMAGE_FEATURE_DESCRIPTION = {
    'centres': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    'keypoints': tf.io.FixedLenFeature([], tf.string), 
    'image_raw': tf.io.FixedLenFeature([], tf.string),
    }

def parse_image_function(example_proto):
    """ Parses the input tf.train.Example proto using the dictionary 
    IMAGE_FEATURE_DESCRIPTION. 
    """
    return tf.io.parse_single_example(example_proto, IMAGE_FEATURE_DESCRIPTION)


def unpack_data(packed_dict, img_size):
    """ Unpack the raw data extracted from a TFRecord file.
    
    Parameters
    ----------
    packed_dict : dict
        Input dict containing raw data.
    img_size : tuple
        Raw image size.

    Returns:
    --------
    img : tf.tensor[float]
        Decoded image in format uint8.
    c_kpts : tf.tensor[float]
        Decoded keypoints (normalised) coordinates.
    c_cntrs : tf.tensor[float]
        Decoded centers (normalised) coordinates. 
    """
    img = tf.reshape(
        tf.io.decode_raw(packed_dict["image_raw"], tf.uint8),
        img_size + (3,)
        )
    c_kpts = tf.reshape(
        tf.io.decode_raw(packed_dict["keypoints"], tf.float32),
        (N_KPTS, 3)
        )
    c_cntrs = tf.reshape(
        tf.io.decode_raw(packed_dict["centres"], tf.float32), (-1, 2)
        )

    return img, c_kpts, c_cntrs


@tf.function
def pad_and_augment(
        img, 
        c_kpts, 
        c_cntrs,
        target_size, 
        roi_prob=-1,
        augmentations=[]
        ):
    """ Performs image augmetation and padding.Returns the preprocessed image 
    and the adjusted keypoints and centres coordinates.

    Parameters
    ----------
    img : tf.tensor[tf.uint8]
        Decoded image.
    c_kpts : tf.tensor[tf.float32]
        Normalised keypooints coordinates.
    c_cntrs : tf.tensor[tf.float32]
        Normalised person centres.
    target_size : tuple[int]
        Target image heigth and width
    roi_prob : float
        Probability to apply the 'focus_on_roi' augmentation, by default to -1.
    augmentations : list[function]
        List of image augmetations, by default to '[]'.

    Returns
    -------
    img : tf.tensor[tf.float32]
        Augmented and normalised image.
    c_kpts : tf.tensor[tf.float32]
        Augmented keypoints coordinates.
    c_cntrs : tf.tensor[tf.float32]
        Augmented person centres coordinates.
    """
    img = tf.cast(img, tf.float32) / 127.5 - 1.
    
    # visible coordinates
    vis_kpts = tf.where(c_kpts[:,-1]==1., 1., 0.)
    c_kpts = c_kpts[:, :2] * tf.expand_dims(vis_kpts, axis=1)

    roi_proba = random.uniform(0, 1) #tf.random.uniform(minval=0, maxval=1, shape=())
    if roi_proba<roi_prob:
        img, c_kpts, c_cntrs = crop_roi(
            img,
            c_kpts,
            vis_kpts,
            c_cntrs, 
            c_kpts, 
            use_random_margin = True,
            min_margin=.05, 
            mean_margin=0.15, 
            confidence_thres=0.05
            )
    
    # augment the images
    for op in augmentations:
        img, c_kpts, c_cntrs = op.augment(img, c_kpts, c_cntrs)
    
    vis_kpts = tf.where(tf.math.reduce_sum(c_kpts, axis=-1)==0, 0., 1.)

    # pad
    height, width, _ = _ImageDimensions(img, 3)
    ratio = tf.cast(width/height, dtype=tf.float32)
    img = tf.image.resize_with_pad(
        img,
        target_height=target_size[0],
        target_width=target_size[1]
        )
    
    # pad images and adjust coords accordingly
    kpts_x, ktps_y = c_kpts[:, 0], c_kpts[:, 1]
    cntrs_x, cntrs_y = c_cntrs[:, 0], c_cntrs[:, 1]

    if ratio < 1:
        pad = (1 - tf.cast(ratio, dtype=tf.float32)) / 2
        kpts_x = pad + kpts_x * ratio
        cntrs_x = pad + cntrs_x * ratio
    elif ratio > 1:
        pad = (1 - tf.cast(1./ratio, dtype=tf.float32)) / 2 
        ktps_y = pad + ktps_y * 1./ratio
        cntrs_y = pad + cntrs_y * ratio

    c_kpts = tf.stack([kpts_x, ktps_y], axis=-1)
    c_cntrs= tf.stack([cntrs_x, cntrs_y], axis=-1)
    
    c_kpts *= tf.expand_dims(vis_kpts, axis=1)

    return img, c_kpts, c_cntrs


def create_labels(c_kpts, c_cntrs, grid_dim):
    """ Creates the labels use for training. Returns two types of labels:
    i.  keypoints normalised coordinates
    ii. Probability maps for the person centres and the keypoints

    Parameters
    ----------
    c_kpts : tf.tensor[float]
        Decoded keypoints (normalised) coordinates.
    c_cntrs : tf.tensor[float]
        Decoded centers (normalised) coordinates.
    grid_dim : int
        Size of the pdf. Must be equal to the FPN output.

    Returns
    -------
    y_coords : tf.tensor[float]
        Tensor of coordinates: [vis_coords, x_cord, y_coord, x_coord, y_coord].
    pdfs : tf.tensor[float]
        Probability density maps.
    """
    ### MAIN LABELS ###
    # visible coordinates
    vis_kpts = tf.where(tf.math.reduce_sum(c_kpts, axis=-1)==0., 0., 1.)
    vis_kpts = tf.expand_dims(vis_kpts, axis=-1)
    y_coords = tf.concat([vis_kpts, c_kpts[:, :2], c_kpts[:, :2]], axis=-1)

    ### AUXILIARY LABELS ###
    
    # create the body centres map
    centres_heatmap = sum_density_maps(
        create_density_maps(c_cntrs, grid_dim)
        )

    # create the joint heatmaps
    kpts_heatmaps = create_density_maps(c_kpts, grid_dim)
    
    y_pdfs = tf.concat([centres_heatmap, kpts_heatmaps], axis=-1)
    
    return y_pdfs


def decode_samples(
    filePaths,
    img_size,
    num_parallel_reads 
    ):
    """ Reads and parse the data from TFRecord files.
    
    Parameters
    ----------
    filePaths : list[str]
        List of TFRecords paths.
    img_size : tuple[int]
        Image height and width.
    num_parallel_reads : int
        Number of files to read in parallel.

    Returns
    -------
    raw_ds : tf.data.Dataset
        Prepared dataset.
    """
    unpacking = lambda x: unpack_data(x, img_size)
    
    raw_ds = tf.data.TFRecordDataset(
        filePaths, 
        num_parallel_reads=num_parallel_reads
        )
    raw_ds = raw_ds.map(parse_image_function)
    raw_ds = raw_ds.map(unpacking)
    
    return raw_ds


def load_TFRecords_dataset(
    filePaths = [],
    dirPath = "", 
    batch_size=8, 
    img_size=(416, 416), 
    target_size=(416,416),           
    grid_dim = 52,
    augmentations=[], 
    roi_thresh=.1,
    debug=False,
    ):
    """
        Datasetloader. Operations:
      i. loads the RGB image,
     ii. loads the keypoints  and person centres coords list,
    iii. performas image normalisazion and augmentation
     iv. generates the labels
      v. shuffles and batches

    Parameters
    ----------
    filePaths : list[str], optional
        Tfrecords list paths.
    dirPath : str, optional
        Path to the directory containing the TFRecords.
    batch_size : int
        Batch size, by default to 8.
    img_size : tuple[int]
        Image height and width as fetched from the TFRecords, by default to 
        (416, 416).
    target_size : tuple[int]
        Target image heigth and width, by default to (416, 416).
    grid_dim : int
        Size of the probability density maps. Must be equal to the FPN output.
    augmentations : list[function]
        List of image augmetations, by default to '[]'.
    roi_thresh : float
        Probability to apply the 'focus_on_roi' augmentation, by default to -1.
    
    Returns
    -------
    output_ds : tf.data.Dataset
        Final datset loader.
    """
    
    # list the files
    if len(filePaths) == 0:
        filePaths = list(paths.list_files(dirPath, contains="tfrecord"))
    n_readers = len(filePaths)
    
    preprocess = lambda img, kpts, cntrs : pad_and_augment(
        img, 
        kpts, 
        cntrs, 
        target_size=target_size, 
        augmentations=augmentations,
        roi_prob=roi_thresh, 
        )
    mk_labels = lambda img, kpts, cntrs: (img, create_labels(
        kpts,
        cntrs, 
        grid_dim)
        )

    # create the dataset
    raw_ds = decode_samples(filePaths, img_size, n_readers)
    output_ds = raw_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(mk_labels, num_parallel_calls=AUTOTUNE)
    if not debug:
        output_ds = output_ds.shuffle(buffer_size=1000).batch(batch_size)
    else: 
        output_ds = output_ds.batch(batch_size)
    output_ds = output_ds.prefetch(AUTOTUNE)
    
    return output_ds

