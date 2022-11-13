import tensorflow as tf
from tensorflow import keras

NUM_JOINTS = 16

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
def avgMDE_2D(y_true, y_pred, threshold=.5):

    # 1. select the probability
    y_true_proba = tf.gather(y_true, [0], axis=-1)
    
    # 2. exclude non visible joints coords
    threshold_mask = y_true_proba > threshold
    threshold_mask = tf.concat([threshold_mask] * 2, axis=-1)

    #coords_2d_pred = tf.gather(y_pred, list(range(1, NUM_JOINTS+1)), axis=-2)
    coords_2d_true = tf.gather(y_true, [1,2], axis=-1)
    coords_2d_pred = tf.gather(y_pred, [1,2], axis=-1)

    coords_2d_true = tf.boolean_mask(coords_2d_true, threshold_mask)
    coords_2d_pred = tf.boolean_mask(coords_2d_pred, threshold_mask)
    
    return AvgMDE(coords_2d_true, coords_2d_pred)


@tf.function
def avgMDE_2D_Raw(y_true, y_pred, threshold=.5):

    # 1. select the probability
    y_true_proba = tf.gather(y_true, [0], axis=-1)
    
    # 2. exclude non visible joints coords
    threshold_mask = y_true_proba > threshold
    threshold_mask = tf.concat([threshold_mask] * 2, axis=-1)

    #coords_2d_pred = tf.gather(y_pred, list(range(1, NUM_JOINTS+1)), axis=-2)
    coords_2d_true = tf.gather(y_true, [1,2], axis=-1)
    coords_2d_pred = tf.gather(y_pred, [3,4], axis=-1)

    coords_2d_true = tf.boolean_mask(coords_2d_true, threshold_mask)
    coords_2d_pred = tf.boolean_mask(coords_2d_pred, threshold_mask)
    
    return AvgMDE(coords_2d_true, coords_2d_pred)



@tf.function
def Accuracy(y_true, y_pred, threshold=.5):
    """ Evaluate the percentage of joints which are correctly predicted as (not) visible. """

    #pred_proba = tf.gather(y_pred, list(range(1, NUM_JOINTS+1)), axis=-2)
    true_proba = tf.gather(y_true, [0], axis=-1)
    pred_proba = tf.gather(y_pred, [0], axis=-1)

    true_thresh = tf.where(true_proba>threshold, 1., 0.)
    pred_thresh = tf.where(pred_proba>threshold, 1., 0.)

    accuracy = tf.cast(true_thresh == pred_thresh, tf.float32)

    return tf.math.reduce_sum(accuracy) / tf.math.reduce_sum(tf.ones_like(true_thresh))  