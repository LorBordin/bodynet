from tensorflow.keras.models import Model
import tensorflow.keras.layers as  L
import tensorflow as tf

from .custom_layers import get_inverse_dist_grid
from .custom_layers import get_max_mask
from .custom_layers import grid_coords 

def create_postproc_model(inputs, name="post_processing", debug=False):
    """
        Returns a model to decode the movenet predictions.
        Postprocessing steps:
        i.   weight the body center heatmap with the inverse distance from the image center
             and choose the position with the higher rate,
        ii.  get a raw estimate of the joints coordinates by adding the keypoints offset coords 
             to the center coords,
        iii. weight the keypoint heatmap with the inverse distance from the raw  keypoints 
             coordinates and choose the position with the highest rate to get a coarse estimate
             of the joints coordinates,
        iv.  add the coordinates offset to the coarse keypoints coordinates to refine the estimate.

    Parameters
    ----------
    inputs: keras.layer
        Input layers (centermap, keypoint offsets, keypoints heatmaps, coordinates offsets).
    name: str
        Model name. The default value is "post_processing".
    """
    
    centermap, k_offsets, k_heatmaps, c_offsets = inputs

    grid_dim, num_joints = k_heatmaps.shape[2:]

    aux_outputs = L.Concatenate(axis=-1)([centermap, k_heatmaps])
    aux_outputs = L.Reshape((grid_dim * grid_dim, num_joints+1), name="aux_output")(aux_outputs)

    # Select the center position of the most centered person
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
    k_offsets = L.GlobalAveragePooling2D()(k_offsets) # use Average inst of Max since offsets can be negative
    k_offsets = L.Lambda(lambda x: x * grid_dim * grid_dim)(k_offsets)
    
    raw_joint_coords = L.Add()([k_offsets, center])

    # reshape properly
    k_offsets = L.Reshape((2, num_joints))(raw_joint_coords)
    k_offsets = L.Permute((2,1))(k_offsets)

    # Get Weighted Heatmaps
    inv_dist = L.Lambda(lambda x: get_inverse_dist_grid(x))([k_heatmaps, k_offsets])
    k_heatmaps = L.Multiply()([k_heatmaps, inv_dist])
    
    # apply softmax to weigthed Heatmaps
    k_heatmaps = L.Reshape((grid_dim * grid_dim, num_joints))(k_heatmaps)
    k_heatmaps = L.Softmax()(k_heatmaps)
    k_heatmaps = L.Reshape((grid_dim, grid_dim, num_joints), name="weighted_heatmaps")(k_heatmaps)

    aux_outputs_2 = L.Concatenate(axis=-1)([w_centermap, k_heatmaps])
    aux_outputs_2 = L.Reshape((grid_dim * grid_dim, num_joints+1), name="aux_output_2")(aux_outputs_2)

    jointmasks = L.Lambda(lambda x: get_max_mask(x))(k_heatmaps)

    # Get the joints coordinates
    xx = L.Lambda(lambda x: grid_coords(x, axis=1))(jointmasks)
    x_j = L.Multiply()([xx, jointmasks])
    x_j = L.GlobalMaxPooling2D()(x_j)

    yy = L.Lambda(lambda x: grid_coords(x, axis=0))(jointmasks)
    y_j = L.Multiply()([yy, jointmasks])
    y_j = L.GlobalMaxPooling2D()(y_j) 

    # Get the joint offsets
    jointmasks = L.Concatenate()([jointmasks, jointmasks])
    c_offsets = L.Multiply()([c_offsets, jointmasks])
    c_offsets = L.GlobalMaxPooling2D()(c_offsets)  
    c_offsets = L.Lambda(lambda x: x / tf.cast(grid_dim, tf.float32))(c_offsets)

    joint_coords = L.Concatenate(name="coarse_coords_concat")([x_j, y_j])
    joint_coords = L.Add(name="final_coords")([joint_coords, c_offsets])

    person_probas = L.Multiply()([centermap, centermask])
    person_probas = L.GlobalMaxPooling2D()(person_probas)
    person_probas = L.Activation("sigmoid")(person_probas)

    keypoints_probas = L.GlobalMaxPooling2D()(k_heatmaps)
    probas = L.Multiply()([person_probas, keypoints_probas])
    
    # Reshape before concatenate the outputs
    probas = L.Reshape((-1, 1))(probas)

    joint_coords = L.Reshape((2, -1))(joint_coords)
    joint_coords = L.Permute((2,1))(joint_coords)

    raw_joint_coords = L.Reshape((2, -1))(raw_joint_coords)
    raw_joint_coords = L.Permute((2,1))(raw_joint_coords)

    outputs = L.Concatenate(axis=-1)([probas, joint_coords, raw_joint_coords])
    outputs = L.Reshape((num_joints, 5), name="output")(outputs)
    
    if debug:
        post_proc = Model(inputs, [outputs, aux_outputs, aux_outputs_2], name=name)
    else:
        post_proc = Model(inputs, [outputs, aux_outputs], name=name)
    
    return post_proc
