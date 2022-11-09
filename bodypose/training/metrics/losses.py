from tensorflow import keras
import tensorflow as tf


ALPHA = 2
BETA = 4
GAMMA = 1e1
LAMBDA = 1 #1e2
NUM_JOINTS = 16

def FocalLoss(y_true, y_pred, num_joints):
    """ Focal Loss used to estimate the probability of the joint being in the cell """
    bin_cross_entropy = tf.keras.losses.binary_crossentropy(
        tf.expand_dims(y_true, axis=-1), 
        tf.expand_dims(y_pred, axis=-1)
        )
    f_1 = tf.math.pow((1-y_pred), ALPHA) * bin_cross_entropy
    f_0 = tf.math.pow((1-y_true), BETA) \
        * tf.math.pow(y_pred, ALPHA) * bin_cross_entropy
    
    loss = tf.where(y_true==1, f_1, f_0)
    loss = 1. / tf.cast(num_joints, tf.float32) * tf.reduce_sum(loss)
    
    return loss


# *** DOES NOT WORRK RIGHT NOW ***
#def DeltaJointsLoss(y_true, y_pred, j_list):
#    """ 
#    Returns the squared displacement error, normalised w.r.t. the true pred,
#    between the left and the right joints. The purpose is to ease the joint 
#    swithching problem.
#    """
#    left_true, right_true = split_LR_joints(y_true, j_list)
#    left_pred, right_pred = split_LR_joints(y_pred, j_list)
#    delta_true = left_true - right_true
#    delta_pred = left_pred - right_pred
#    return tf.math.pow(delta_true - delta_pred, 2)
#
#
#def split_LR_joints(y, j_list):
#    """ 
#    Splits the joints into left and right parts. Assumes the list contains 
#    only the symmetric joints, firt the left joints, then the right ones. 
#    j_list is hardcoded and can be found in utils.joints.
#    """
#    n = len(j_list) // 2
#    left = tf.gather(y, j_list[:n], axis=1)
#    right = tf.gather(y, j_list[n:], axis=1)
#    return left, right


@tf.function
def ClassificationLoss(y_true, y_pred, ):

    # 1. select the probability
    y_pred_proba = tf.gather(y_pred, list(range(1, NUM_JOINTS+1)), axis=-2)
    y_true_proba = tf.gather(y_true, [0], axis=-1)
    y_pred_proba = tf.gather(y_pred_proba, [0], axis=-1)

    # BinaryCrossEntropy Loss on probabilities
    binary_cross_entropy = keras.losses.binary_crossentropy(y_true_proba, y_pred_proba)
    
    return binary_cross_entropy
    

@tf.function
def RegrCoordsLoss(y_true, y_pred, threshold=0.5):

    # 1. select the probability
    y_true_proba = tf.gather(y_true, [0], axis=-1)
    
    # 2. Deal with coordinates
    # exclude non visible joints coords
    threshold_mask = y_true_proba > threshold
    threshold_mask = tf.concat([threshold_mask] * 2, axis=-1)
    
    # select coords
    y_pred_c = tf.gather(y_pred, list(range(1, NUM_JOINTS+1)), axis=-2)
    y_true_c = tf.gather(y_true, [1,2], axis=-1)
    y_pred_c = tf.gather(y_pred_c, [1,2], axis=-1)

    y_true_c = tf.boolean_mask(y_true_c, threshold_mask)
    y_pred_c = tf.boolean_mask(y_pred_c, threshold_mask)

    # MSE Loss on coords 2D
    mse = keras.losses.mse(y_true_c, y_pred_c)

    return mse

