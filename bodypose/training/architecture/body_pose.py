from tensorflow.keras.models import Model
import tensorflow.keras.layers as L  
import tensorflow as tf

import os

from ..postprocessing import get_joint_coords
from .backbone import create_backbone
from .feature_block import create_FB
from .head import create_Head
from .fpn import create_FPN

EPSILON = 1e-2
    
def create_model(input_shapes, 
                 strides, 
                 num_joints, 
                 alpha=1, 
                 backbone_arch="mobilenetV2", 
                 use_depthwise=False,
                 use_FB=True):
    """
    input_shapes:       list of tuples
    strides:            stride factor for the backbone
    num_joints:         number of body keypoints
    alpha:              depth factor of the network
    backbone_arch:      mobileNetV2 or mobileNetV3
    use_depthwise:      if True uses depthwise convolutions
    """
    img_shape = input_shapes[0]
    depth_shape = input_shapes[1]
    
    # backbone 
    in_img, backbone = create_backbone(img_shape, strides, alpha, arch=backbone_arch)
    
    # FPN
    FPN = create_FPN(inputs=backbone, in_channels=int(128*alpha))
    
    # Depth Feature Block
    in_depth, FB = create_FB(depth_shape, alpha=alpha, use_depthwise=use_depthwise, stride=strides[-1]) 

    # Head
    xy_input = L.Concatenate(axis=-1, name="head_2d_input")([FPN, FB])  
    centermap, k_offsets, k_heatmaps, c_offsets, depth_coords = create_Head(inputs=xy_input, 
                                            num_joints=num_joints, 
                                            out_channels=int(256*alpha),
                                            use_depthwise=use_depthwise)
    s = centermap.shape[1]
    aux_outputs = L.Concatenate(axis=-1)([centermap, k_heatmaps])
    aux_outputs = L.Reshape((s*s, num_joints+1), name="aux_output")(aux_outputs)

    ## POST PROCESSING ##

    # Get the position of the center of the body
    center_offset = tf.convert_to_tensor(0.5)
    inv_dist = L.Lambda(lambda x: get_inverse_dist_grid(x))([centermap, center_offset])
    w_centermap = L.Multiply()([centermap, inv_dist])
    centermask = L.Lambda(lambda x: get_max_mask(x))(w_centermap)

    # get the coords of the center
    xx = L.Lambda(lambda x: grid_coords(x, axis=1))(centermask)
    x_c = L.Multiply()([xx, centermask])
    x_c = L.GlobalMaxPooling2D()(x_c)
    x_c = L.Concatenate()([x_c]*num_joints)

    yy = L.Lambda(lambda x: grid_coords(x, axis=0))(centermask)
    y_c = L.Multiply()([yy, centermask])
    y_c = L.GlobalMaxPooling2D()(y_c)    
    y_c = L.Concatenate()([y_c]*num_joints)

    center = L.Concatenate()([x_c, y_c])
    
    # select the joint offsets from the body center
    rep_mask = L.Concatenate()([centermask]*num_joints*2)
    
    k_offsets = L.Multiply()([k_offsets, rep_mask]) 
    k_offsets = L.GlobalAveragePooling2D()(k_offsets)
    k_offsets = L.Lambda(lambda x: x * s * s)(k_offsets)
    
    raw_joint_coords = L.Add()([k_offsets, center])

    # reshape properly
    k_offsets = L.Reshape((2, num_joints))(raw_joint_coords)
    k_offsets = L.Permute((2,1))(k_offsets)

    # Get Weighted Heatmaps
    inv_dist = L.Lambda(lambda x: get_inverse_dist_grid(x))([k_heatmaps, k_offsets])
    k_heatmaps = L.Multiply()([k_heatmaps, inv_dist])
    
    # apply softmax to weigthed Heatmaps
    k_heatmaps = L.Reshape((s*s, num_joints))(k_heatmaps)
    k_heatmaps = L.Softmax()(k_heatmaps)
    k_heatmaps = L.Reshape((s, s, num_joints), name="weighted_heatmaps")(k_heatmaps)

    aux_outputs_2 = L.Concatenate(axis=-1)([w_centermap, k_heatmaps])
    aux_outputs_2 = L.Reshape((s*s, num_joints+1), name="aux_output_2")(aux_outputs_2)

    jointmasks = L.Lambda(lambda x: get_max_mask(x))(k_heatmaps)

    # Get the joints coordinates
    xx = L.Lambda(lambda x: grid_coords(x, axis=1))(jointmasks)
    x_j = L.Multiply()([xx, jointmasks])
    x_j = L.GlobalMaxPooling2D()(x_j)

    yy = L.Lambda(lambda x: grid_coords(x, axis=0))(jointmasks)
    y_j = L.Multiply()([yy, jointmasks])
    y_j = L.GlobalMaxPooling2D()(y_j) 

    # Get depth coords
    depth_coords = L.Multiply()([depth_coords, jointmasks])
    depth_coords = L.GlobalMaxPooling2D()(depth_coords)

    # Get the joint offsets
    jointmasks = L.Concatenate()([jointmasks, jointmasks])
    c_offsets = L.Multiply()([c_offsets, jointmasks])
    c_offsets = L.GlobalMaxPooling2D()(c_offsets)  
    c_offsets = L.Lambda(lambda x: x / tf.cast(s, tf.float32))(c_offsets)

    joint_coords = L.Concatenate(name="coarse_coords_concat")([x_j, y_j])
    joint_coords = L.Add(name="final_coords")([joint_coords, c_offsets])

    person_probas = L.Multiply()([centermap, centermask])
    person_probas = L.GlobalMaxPooling2D()(person_probas)
    person_probas = L.Activation("sigmoid")(person_probas)

    keypoints_probas = L.GlobalMaxPooling2D()(k_heatmaps)
    probas = L.Multiply()([person_probas, keypoints_probas])
    
    # Reshape before concatenate the outputs
    probas = L.Reshape((-1, 1))(probas)
    depth_coords = L.Reshape((-1, 1))(depth_coords)

    joint_coords = L.Reshape((2, -1))(joint_coords)
    joint_coords = L.Permute((2,1))(joint_coords)

    raw_joint_coords = L.Reshape((2, -1))(raw_joint_coords)
    raw_joint_coords = L.Permute((2,1))(raw_joint_coords)

    outputs = L.Concatenate(axis=-1)([probas, joint_coords, depth_coords, raw_joint_coords])
    outputs = L.Reshape((num_joints, 6), name="output")(outputs)

    return Model(inputs=[in_img, in_depth], outputs=[outputs, aux_outputs, aux_outputs_2])


def get_max_coords(X):
    pass

def get_inverse_dist_grid(item):
    """
        Given a tensor X and an offset (either scalar of vector), return a map grid with 
        the inverse distance from the offset.
    """

    X, offsets = item
    X_ones = tf.ones_like(X)
    if len(offsets.shape)>1:
        offset_x = offsets[:, :, 0]
        offset_y = offsets[:, :, 1]
        offset_x = tf.transpose(tf.transpose(X_ones, (1, 2, 0, 3)) * offset_x, (2, 0, 1, 3))
        offset_y = tf.transpose(tf.transpose(X_ones, (1, 2, 0, 3)) * offset_y, (2, 0, 1, 3))
    else:
        offset_x = offset_y = offsets

    xx = grid_coords(X, axis=1)
    xx = tf.math.pow(xx - offset_x, 2)

    yy = grid_coords(X, axis=0)
    yy = tf.pow(yy - offset_y, 2)

    dist = tf.math.sqrt(xx + yy + EPSILON)

    return 1.0 / dist

def get_max_mask(x):
    """ Returns a mask with all zeros except in the position of the max. """
    size = x.shape[1]
    n_joints = x.shape[-1]
    x = tf.reshape(x, (-1, size*size, n_joints))
    y = tf.cast(tf.equal(x, tf.expand_dims(tf.math.reduce_max(x, axis=1), axis=1)), tf.float32)
    return tf.reshape(y, (-1, size, size, n_joints))

def grid_coords(x, axis):
    """ Returns a tensor with same shape of input with the coords along the x/y dimension """
    y = tf.ones_like(x, dtype=tf.float32)
    grid_dim = x.shape[1]
    num_joints = x.shape[-1]
    coords_grid = tf.range(tf.cast(grid_dim, tf.float32)) / tf.cast(grid_dim, tf.float32)                      # vector that ranges from 0 to 1 with step 1/grid_dim
    coords_grid = tf.stack([coords_grid] * grid_dim)                # 2d tensor
    if axis==0:                                                     # x --> y coords
        coords_grid = tf.transpose(coords_grid)
    coords_grid = tf.stack([coords_grid] * num_joints, axis=-1) # tile num_joints tensors together
    return y * coords_grid
 


if __name__=="__main__":

    model = create_model([(416, 416, 3), (416, 416, 1)], (32, 16, 8), 17)
    print(model.summary())
    print()
    
    import numpy as np
    import time

    img = (np.random.randint(0, 255, (1, 416, 416, 3)) / 255).astype(np.float32)
    depth = (np.random.randint(0, 3000, (1, 416, 416, 1)) / 255).astype(np.float32)
    
    start = time.time()
    for i in range(100):
        model([img, depth])
    end = time.time()
    print(f"[INFO] Average inference time: {(end-start)/100 * 1000:.1f} ms.")