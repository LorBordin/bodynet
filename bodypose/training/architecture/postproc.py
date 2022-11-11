from tensorflow.keras.models import Model
import tensorflow.keras.layers as  L
import tensorflow as tf

from .custom_layers import ExtractCoordinates
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
    
    centermap, k_offsets, k_heatmaps, h_offsets = inputs
    grid_dim, num_joints = k_heatmaps.shape[-2:]

    # 1. Select the position of the person center and joints
    centermask = L.Lambda(lambda x: get_max_mask(x))(centermap)
    jointmask = L.Lambda(lambda x: get_max_mask(x))(k_heatmaps)

    # 2. Extract the corresponding coordinates
    center = ExtractCoordinates(n_rep=num_joints)(centermask)
    kpts_coords = ExtractCoordinates(n_rep=1)(jointmask)
     
    # 3. Get the keypoints offsets from the body center
    # need to use Average instead of Max layer since offsets can be negative
    offset_mask = L.Concatenate()([centermask]*num_joints*2)
    
    k_offsets = L.Multiply()([k_offsets, offset_mask]) 
    k_offsets = L.GlobalAveragePooling2D()(k_offsets) 
    k_offsets = L.Lambda(lambda x: x * grid_dim * grid_dim)(k_offsets)
    
    raw_kpts_coords = L.Add()([k_offsets, center])

    # 4. Get the offsets from the keypoints heatmaps
    offset_mask = L.Concatenate()([jointmask]*2)
    h_offsets = L.Multiply()([h_offsets, offset_mask])
    h_offsets = L.GlobalMaxPooling2D()(h_offsets)  
    h_offsets = L.Lambda(lambda x: x / tf.cast(grid_dim, tf.float32))(h_offsets)

    kpts_coords = L.Add(name="final_coords")([kpts_coords, h_offsets])
    kpts_coords = L.Reshape((2, -1))(kpts_coords)
    kpts_coords = L.Permute((2,1))(kpts_coords)
    
    # 5. Get the keypoint probability
    kpts_probas = L.Multiply()([k_heatmaps, jointmask])
    kpts_probas = L.GlobalMaxPooling2D()(kpts_probas)
    kpts_probas = L.Reshape((-1, 1))(kpts_probas)
    
    # Concatenate the outputs together
    kpts_coords = L.Concatenate()([kpts_probas, kpts_coords, raw_kpts_coords])
    heatmaps = L.Concatenate()([centermap, k_heatmaps])

    post_proc = Model(inputs, [kpts_coords, heatmaps], name=name)
    
    return post_proc
