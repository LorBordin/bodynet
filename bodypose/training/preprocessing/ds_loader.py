from ..utils import EXCLUDE_JOINTS_DEFAULT
from ..utils import JOINTS_NAME

from ...dataset.tfrecord_maker import IMAGE_FEATURE_DESCRIPTION
from .preprocessing import create_density_map
from .preprocessing import create_joint_mask
from .preprocessing import create_depth_map
from .preprocessing import mask_img
from .preprocessing import crop_roi

import numpy as np
import tensorflow as tf
from imutils import paths
from tensorflow.python.ops.image_ops import _ImageDimensions

AUTOTUNE = tf.data.AUTOTUNE


def parse_image_function(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, IMAGE_FEATURE_DESCRIPTION)


def unpack_data(packed_dict, img_size):
    img = tf.reshape(tf.io.decode_raw(packed_dict["image_raw"], tf.uint8), img_size + (3,))
    labels = tf.reshape(tf.io.decode_raw(packed_dict["joints"], tf.float32), (30, 3))

    return img, labels


def pad_and_augment(img, 
                    joints_coords, 
                    target_size,
                    focus_on_roi, 
                    roi_thresh,
                    augmentations=[]):
    """ Return the preprocessed image, depth and the normalised 3D joints coords. """
    img = tf.cast(img, tf.float32) / 127.5 - 1.
    
    # visible coordinates
    probas = tf.cast(tf.cast(tf.math.reduce_sum(joints_coords,axis=-1), dtype=tf.bool), tf.float32)

    # augment the images
    for op in augmentations:
        img, depth, joints_coords = op.augment(img, depth, joints_coords)
    
    roi_proba = tf.random.uniform(minval=0, maxval=1, shape=())
    if focus_on_roi and roi_proba<roi_thresh:
        img, joints_coords = crop_roi(img, joints_coords, probas, joints_coords, min_margin=.05, max_margin=0.15, thresh=0.05)
    
    # pad
    height, width, _ = _ImageDimensions(img, 3)
    ratio = tf.cast(width/height, dtype=tf.float32)
    img = tf.image.resize_with_pad(img, target_height=target_size[0], target_width=target_size[1])
    
    # pad images and adjust coords accordingly
    coords_x, coords_y = joints_coords[:, 0], joints_coords[:, 1]

    if ratio < 1:
        pad = (1 - tf.cast(ratio, dtype=tf.float32)) / 2
        coords_x = pad + coords_x * ratio
    elif ratio > 1:
        pad = (1 - tf.cast(1./ratio, dtype=tf.float32)) / 2 
        coords_y = pad + coords_y * 1./ratio

    joints_coords = tf.stack([coords_x, coords_y], axis=-1)
    
    vis_mask = tf.stack([probas]*2, axis=-1)
    joints_coords *= vis_mask

    return img, joints_coords


def create_labels(joints_coords, grid_dim, exclude_joints=[]):
    """ Given the joints_coords list it created the labels used for the training.
        - Outputs: y: tensor [grid_dim**2, num_joints*4+1]: 
                      ([1] object pdf, [n_joints] pdfs, [n_joints * 2] offsets x/y, [n_joints] z_coords)
    """
    # exclude some joints from detection
    if len(exclude_joints)!=0:
        mask = create_joint_mask(JOINTS_NAME, tf.convert_to_tensor(exclude_joints))
        joints_coords = tf.boolean_mask(joints_coords, mask)
    n_joints = 30 - len(exclude_joints)

    ### MAIN LABELS ###
    probas = tf.cast(tf.cast(tf.math.reduce_sum(joints_coords,axis=-1), dtype=tf.bool), tf.float32)
    probas = tf.expand_dims(probas, axis=-1)
    y_coords = tf.concat([probas, joints_coords, joints_coords[:, :2]], axis=-1)

    ### AUXILIARY LABELS ###
    # create the body center map
    mask = tf.concat([probas==1]*2, axis=-1)
    body_center = tf.reshape(tf.boolean_mask(joints_coords[:, :2], mask), (-1, 2))
    body_center = tf.math.reduce_mean(body_center, axis=0)
    center_heatmap = create_density_map(tf.expand_dims(body_center, axis=0), grid_dim)

    # create the joint heatmaps
    joint_heatmaps = create_density_map(joints_coords, grid_dim)
    
    y_aux = tf.concat([center_heatmap, joint_heatmaps], axis=-1)

    y_fake = tf.ones((grid_dim, grid_dim, n_joints+1))
    
    return y_coords, y_aux, y_fake


def single_image_preprocessing(filePath, img_size, target_size, grid_dim, augmentations):
    
    unpacking = lambda x: unpack_data(x, img_size)
    
    paths_ds = tf.data.TFRecordDataset(filePath)
    raw_ds = paths_ds.map(parse_image_function)
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
                           exclude_joints = EXCLUDE_JOINTS_DEFAULT):
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
    
    img_preproc = lambda x: single_image_preprocessing(x, img_size, target_size, grid_dim, augmentations)
    preprocess = lambda x, y, z : pad_and_augment(x, y, z, target_size, focus_on_roi, roi_thresh=roi_thresh, augmentations=augmentations)
    mk_labels = lambda x, y, z: ((x, y), create_labels(z, grid_dim, exclude_joints=exclude_joints))

    # create the dataset
    paths_ds = tf.data.Dataset.from_tensor_slices(filePaths)
    raw_ds = paths_ds.interleave(img_preproc, cycle_length=n_readers)
    output_ds = raw_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(mk_labels, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.shuffle(buffer_size=1000).batch(batch_size)

    # apply binary mask on img based on depth 
    if masked_img:
        output_ds = output_ds.map(lambda x, y: (mask_img(x), y))

    output_ds = output_ds.prefetch(AUTOTUNE)
    
    return output_ds

