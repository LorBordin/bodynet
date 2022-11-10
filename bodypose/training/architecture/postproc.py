from tensorflow.keras.models import Model
import tensorflow.keras.layers as  L
import tensorflow as tf

from .custom_layers import get_inverse_dist_grid
from .custom_layers import get_max_mask
from .custom_layers import grid_coords 

def create_postproc_model(inputs, name="post_processing"):
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
    
    centermap, k_heatmaps, c_offsets = inputs

    grid_dim, num_joints = k_heatmaps.shape[-2:]
    
    # apply softmax to weigthed Heatmaps
    #w_k_heatmaps = L.Reshape((grid_dim * grid_dim, num_joints))(w_k_heatmaps)
    #w_k_heatmaps = L.Softmax()(w_k_heatmaps)
    #w_k_heatmaps = L.Reshape((grid_dim, grid_dim, num_joints), name="weighted_heatmaps")(w_k_heatmaps)

    jointmask = L.Lambda(lambda x: get_max_mask(x))(k_heatmaps)
    #centermask = L.Lambda(lambda x: get_max_mask(x))(centermap)

    # Get the joints coordinates
    xx = L.Lambda(lambda x: grid_coords(x, axis=1))(jointmask)
    x_j = L.Multiply()([xx, jointmask])
    x_j = L.GlobalMaxPooling2D()(x_j)

    yy = L.Lambda(lambda x: grid_coords(x, axis=0))(jointmask)
    y_j = L.Multiply()([yy, jointmask])
    y_j = L.GlobalMaxPooling2D()(y_j) 

    kpts_probas = L.Multiply()([k_heatmaps, jointmask])
    kpts_probas = L.GlobalMaxPooling2D()(kpts_probas)

    # Get the joint offsets
    jointmasks_offset = L.Concatenate()([jointmask, jointmask])
    c_offsets = L.Multiply()([c_offsets, jointmasks_offset])
    c_offsets = L.GlobalMaxPooling2D()(c_offsets)  
    c_offsets = L.Lambda(lambda x: x / tf.cast(grid_dim, tf.float32))(c_offsets)

    kpts_coords = L.Concatenate(name="coarse_coords_concat")([x_j, y_j])
    kpts_coords = L.Add(name="final_coords")([kpts_coords, c_offsets])
    
    # Reshape before concatenate the outputs
    kpts_probas = L.Reshape((-1, 1))(kpts_probas)
    kpts_coords = L.Reshape((2, -1))(kpts_coords)
    kpts_coords = L.Permute((2,1))(kpts_coords)

    kpts_coords = L.Concatenate()([kpts_probas, kpts_coords])
    heatmaps = L.Concatenate()([centermap, k_heatmaps])

    post_proc = Model(inputs, [kpts_coords, heatmaps, c_offsets], name=name)
    
    return post_proc
