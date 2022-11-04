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

    coords_2d_true = y_true[:, :, 1:3]
    coords_2d_pred = y_pred[:, :, 1:3]
    
    return AvgMDE(coords_2d_true, coords_2d_pred)


@tf.function
def avgMDE_2D_RAW(y_true, y_pred):

    coords_2d_true = y_true[:, :, 3:]
    coords_2d_pred = y_pred[:, :, 3:]
    
    return AvgMDE(coords_2d_true, coords_2d_pred)


@tf.function
def Accuracy(y_true, y_pred, threshold=.5):
    """ Evaluate the percentage of joints which are correctly predicted as (not) visible. """

    num_joints = tf.cast(y_pred.shape[1], tf.int32)

    true_proba = y_true[:,:,0]
    pred_proba = y_pred[:,:,0]

    true_thresh = tf.reshape(tf.where(true_proba>threshold, 1., 0.), (-1, num_joints, 1))
    pred_thresh = tf.reshape(tf.where(pred_proba>threshold, 1., 0.), (-1, num_joints, 1))

    return tf.math.reduce_sum(pred_thresh) / tf.math.reduce_sum(tf.ones_like(true_thresh))  