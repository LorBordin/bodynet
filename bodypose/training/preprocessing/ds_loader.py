#from .preprocessing import create_density_map
#from .preprocessing import create_joint_mask
#from .preprocessing import create_depth_map
#from .preprocessing import mask_img
from .preprocessing import crop_roi
from config import N_KPTS

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
        Probability to apply the 'focus_on_roi' augmentation, by default -1.
    augmentations : list[function]
        List of image augmetations.

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

    # augment the images
    for op in augmentations:
        img, c_kpts, c_cntrs = op.augment(img, c_kpts, c_cntrs)
    
    roi_proba = tf.random.uniform(minval=0, maxval=1, shape=())
    if roi_proba<roi_prob:
        img, c_kpts = crop_roi(
            img,
            c_kpts,
            vis_kpts, 
            c_kpts, 
            min_margin=.05, 
            max_margin=0.15, 
            thresh=0.05
            )
    
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
    c_cntrs= tf.stack([cntrs_x, cntrs_x], axis=-1)
    
    vis_mask = tf.stack([vis_kpts]*2, axis=-1)
    c_kpts *= vis_mask

    return img, c_kpts, c_cntrs


def create_labels(joints_coords, grid_dim, exclude_joints=[]):
    """ Given the joints_coords list it created the labels used for the training.
        - Outputs: y: tensor [grid_dim**2, num_joints*4+1]: 
                      ([1] object pdf, [n_joints] pdfs, [n_joints * 2] offsets x/y, [n_joints] z_coords)
    """
    # exclude some joints from detection
    if len(exclude_joints)!=0:
        mask = create_joint_mask(
            JOINTS_NAME, 
            tf.convert_to_tensor(exclude_joints)
            )
        joints_coords = tf.boolean_mask(joints_coords, mask)
    n_joints = 30 - len(exclude_joints)

    ### MAIN LABELS ###
    probas = tf.cast(
        tf.cast(tf.math.reduce_sum(joints_coords,axis=-1), dtype=tf.bool), 
        tf.float32)
    probas = tf.expand_dims(probas, axis=-1)
    y_coords = tf.concat(
        [probas, joints_coords, joints_coords[:, :2]],
        axis=-1
        )

    ### AUXILIARY LABELS ###
    # create the body center map
    mask = tf.concat([probas==1]*2, axis=-1)
    body_center = tf.reshape(
        tf.boolean_mask(joints_coords[:, :2], mask),
        (-1, 2)
        )
    body_center = tf.math.reduce_mean(body_center, axis=0)
    center_heatmap = create_density_map(
        tf.expand_dims(body_center, axis=0),
        grid_dim
        )

    # create the joint heatmaps
    joint_heatmaps = create_density_map(joints_coords, grid_dim)
    
    y_aux = tf.concat([center_heatmap, joint_heatmaps], axis=-1)

    y_fake = tf.ones((grid_dim, grid_dim, n_joints+1))
    
    return y_coords, y_aux, y_fake


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


def load_TFRecords_dataset(dirPath, 
                           batch_size=8, 
                           img_size=(416, 312),
                           target_size=(416, 416),            
                           grid_dim = 52,
                           augmentations=[], 
                           masked_img = False,
                           focus_on_roi = False,
                           roi_thresh=.1,
                           exclude_joints = []):
    """
        Datasetloader. Operations:
        - loads, normalise and resize with pad RGB and DEPTH,
        - loads the joints coords list (2D_coords, z_coords),
        - shuffles the imgs,
        - generates the labels from the coords list, 
        - batches the data
    """
    
    # list the files
    filePaths = list(paths.list_files(dirPath, contains="tfrecord"))
    n_readers = len(filePaths)
    
    preprocess = lambda x, y, z : pad_and_augment(
        x, 
        y, 
        z, 
        target_size, 
        focus_on_roi, 
        roi_prob=roi_thresh, 
        augmentations=augmentations
        )
    mk_labels = lambda x, y, z: ((x, y), create_labels(
        z, 
        grid_dim, 
        exclude_joints=exclude_joints)
        )

    # create the dataset
    raw_ds = decode_samples(filePaths, img_size, n_readers)
    output_ds = raw_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(mk_labels, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.shuffle(buffer_size=1000).batch(batch_size)

    # apply binary mask on img based on depth 
    if masked_img:
        output_ds = output_ds.map(lambda x, y: (mask_img(x), y))

    output_ds = output_ds.prefetch(AUTOTUNE)
    
    return output_ds

