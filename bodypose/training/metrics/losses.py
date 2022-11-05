from tensorflow import keras
import tensorflow as tf


ALPHA = 2
BETA = 4
GAMMA = 1e1
LAMBDA = 1 #1e2

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
def TotalLoss_2D(y_true, y_pred, threshold=0.5, j_list=[]):
    
    y_true_reg, y_true_aux = y_true[0], y_true[1]
    y_pred_reg, y_pred_aux = y_pred[0], y_pred[1]

    num_joints = tf.cast(y_pred.shape[1], tf.int32)

    ### PRIMARY LOSS ###
    # 1. select the probability
    print("*** labels ",  y_true_reg.shape, "***")
    y_true_proba = tf.gather(y_true_reg, [0], axis=-1)
    y_pred_proba = tf.gather(y_pred_reg, [0], axis=-1)

    # BinaryCrossEntropy Loss on probabilities
    binary_cross_entropy = keras.losses.binary_crossentropy(y_true_proba, y_pred_proba)
    binary_cross_entropy = tf.reduce_sum(binary_cross_entropy)
    
    # 2. Deal with coordinates
    # exclude non visible joints coords
    threshold_mask = tf.reshape(tf.where(y_true_proba>threshold, 1., 0.), (-1, num_joints, 1))
    y_true_reg *= threshold_mask
    y_pred_reg *= threshold_mask

    # select coords
    y_true_c = tf.gather(y_true_reg, [1,2], axis=-1)
    y_pred_c = tf.gather(y_pred_reg, [1,2], axis=-1)

    # MSE Loss on coords 3D
    mse = keras.losses.mse(y_true_c, y_pred_c)
    mse_loss = tf.math.reduce_sum(mse)

    # select RAW coords
    y_true_raw_c = tf.gather(y_true_reg, [3,4], axis=-1)
    y_pred_raw_c = tf.gather(y_pred_reg, [3,4], axis=-1)
    
    # MSE Loss on RAW coords 2D
    mse_raw = keras.losses.mse(y_true_raw_c, y_pred_raw_c)
    mse_loss_raw = tf.math.reduce_sum(mse_raw)

    # Displacement Loss on 2D coords
    #delta_loss = DeltaJointsLoss(y_true_c[:,:,:2], y_pred_c[:,:,:2], j_list)
    #delta_loss = tf.math.reduce_sum(delta_loss)
    
    ### AUXILIARY LOSS ###
    focal_loss = FocalLoss(y_true_aux, y_pred_aux, num_joints)
    
    #return LAMBDA * (mse_loss + delta_loss) + focal_loss + binary_cross_entropy + mse_loss_raw
    return LAMBDA * (mse_loss) + focal_loss + binary_cross_entropy + mse_loss_raw


@tf.function
def RegressionLoss2D(y_true, y_pred, threshold=0.5, j_list=[]):
    
    num_joints = tf.cast(y_pred.shape[1], tf.int32)

    # 1. select the probability
    y_true_proba = tf.gather(y_true, [0], axis=-1)
    y_pred_proba = tf.gather(y_pred, [0], axis=-1)

    # BinaryCrossEntropy Loss on probabilities
    binary_cross_entropy = keras.losses.binary_crossentropy(y_true_proba, y_pred_proba)
    binary_cross_entropy = tf.reduce_sum(binary_cross_entropy)
    
    # 2. Deal with coordinates
    # exclude non visible joints coords
    threshold_mask = y_true_proba > threshold
    threshold_mask = tf.concat([threshold_mask] * 2, axis=-1)
    
    # select coords
    y_true_c = tf.gather(y_true, [1,2], axis=-1)
    y_pred_c = tf.gather(y_pred, [1,2], axis=-1)

    y_true_c = tf.boolean_mask(y_true_c, threshold_mask)
    y_pred_c = tf.boolean_mask(y_pred_c, threshold_mask)

    # MSE Loss on coords 2D
    mse = keras.losses.mse(y_true_c, y_pred_c)
    mse_loss = tf.math.reduce_sum(mse)
    
    # select RAW coords
    y_true_raw_c = tf.gather(y_true, [3,4], axis=-1)
    y_pred_raw_c = tf.gather(y_pred, [3,4], axis=-1)

    y_true_raw_c = tf.boolean_mask(y_true_raw_c, threshold_mask)
    y_pred_raw_c = tf.boolean_mask(y_pred_raw_c, threshold_mask)

    
    # MSE Loss on RAW coords 2D
    mse_raw = keras.losses.mse(y_true_raw_c, y_pred_raw_c)
    mse_loss_raw = tf.math.reduce_sum(mse_raw)
    
    # Displacement Loss on 2D coords
    #delta_loss = DeltaJointsLoss(y_true_c[:,:,:2], y_pred_c[:,:,:2], j_list)
    #delta_loss = tf.math.reduce_sum(delta_loss)
    
    #return LAMBDA * (mse_loss + delta_loss + mse_loss_raw) + binary_cross_entropy + mse_loss_raw
    return LAMBDA * (mse_loss + binary_cross_entropy) + mse_loss_raw


@tf.function
def AuxiliaryLoss(y_true, y_pred):
    
    num_joints = tf.cast(tf.math.sqrt(tf.cast(y_pred.shape[-2], tf.float32)), tf.int32)
    focal_loss = FocalLoss(y_true, y_pred, num_joints)
    
    return LAMBDA * focal_loss 