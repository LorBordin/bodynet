import tensorflow as tf
from tensorflow import keras

@tf.function
def MPJDE(true_coords, pred_coords):
    """ Returns the Mean Per Joint Displacement Error. Works in all dimensions """
    disp = tf.math.abs(true_coords - pred_coords)
    mean_disp = tf.math.reduce_mean(disp, axis=-1)
    return mean_disp


@tf.function
def AvgMDE(true_coords, pred_coords):
    """ Works in any dimension. """
    per_joint_error = MPJDE(true_coords, pred_coords)
    return tf.math.reduce_mean(per_joint_error)


@tf.function
def avgMDE_2D(y_true, y_pred):

    num_joints = tf.cast(y_true.shape[-2], tf.int32)

    coords_2d_pred = tf.gather(y_pred, list(range(1, num_joints+1)), axis=-2)
    coords_2d_true = tf.gather(y_true, [1,2,3], axis=-1)
    coords_2d_pred = tf.gather(coords_2d_pred, [1,2,3], axis=-1)
    
    return AvgMDE(coords_2d_true, coords_2d_pred)


@tf.function
def avgMDE_2D_RAW(y_true, y_pred):

    coords_2d_true = y_true[:, :, 3:]
    coords_2d_pred = y_pred[:, :, 3:]
    
    return AvgMDE(coords_2d_true, coords_2d_pred)


@tf.function
def Accuracy(y_true, y_pred, threshold=.5):
    """ Evaluate the percentage of joints which are correctly predicted as (not) visible. """

    num_joints = tf.cast(y_true.shape[-2], tf.int32)

    pred_proba = tf.gather(y_pred, list(range(1, num_joints+1)), axis=-2)
    true_proba = tf.gather(y_true, [0], axis=-1)
    pred_proba = tf.gather(pred_proba, [0], axis=-1)

    true_thresh = tf.where(true_proba>threshold, 1., 0.)
    pred_thresh = tf.where(pred_proba>threshold, 1., 0.)

    accuracy = tf.cast(true_thresh == pred_thresh, tf.float32)

    return tf.math.reduce_sum(accuracy) / tf.math.reduce_sum(tf.ones_like(true_thresh))  