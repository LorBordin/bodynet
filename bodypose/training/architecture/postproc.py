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

    # 1. Get the coordinates of the person center 
    center_offset = tf.convert_to_tensor(0.5)
    inv_dist = L.Lambda(lambda x: get_inverse_dist_grid(x))([centermap, center_offset])
    w_centermap = L.Multiply()([centermap, inv_dist])
    centermask = L.Lambda(lambda x: get_max_mask(x))(w_centermap)
    
    center = ExtractCoordinates(n_rep=num_joints)(centermask)

    # 2. Get the keypoints offset from the person center 
    offset_mask = L.Concatenate()([centermask]*num_joints*2)
    
    k_offsets = L.Multiply()([k_offsets, offset_mask]) 
    k_offsets = L.GlobalAveragePooling2D()(k_offsets) 
    k_offsets = L.Lambda(lambda x: x * grid_dim * grid_dim)(k_offsets)
    
    # 3. Get the raw keypoints coordinates
    raw_kpts_coords = L.Add()([k_offsets, center])
    raw_kpts_coords = L.Reshape((2, -1))(raw_kpts_coords)
    raw_kpts_coords = L.Permute((2,1))(raw_kpts_coords)

    # 4. Get the coordinates of the keypoints 
    joint_offsets = L.Reshape((2, num_joints))(raw_kpts_coords)
    joint_offsets = L.Permute((2,1))(joint_offsets)
    inv_dist = L.Lambda(lambda x: get_inverse_dist_grid(x))([k_heatmaps, joint_offsets])
    w_k_heatmaps = L.Multiply()([k_heatmaps, inv_dist])
    jointmask = L.Lambda(lambda x: get_max_mask(x))(w_k_heatmaps)

    kpts_coords = ExtractCoordinates(n_rep=1)(jointmask)

    # 5. Get the offsets from the keypoints heatmaps
    offset_mask = L.Concatenate()([jointmask]*2)
    h_offsets = L.Multiply()([h_offsets, offset_mask])
    h_offsets = L.GlobalMaxPooling2D()(h_offsets)  
    h_offsets = L.Lambda(lambda x: x / tf.cast(grid_dim, tf.float32))(h_offsets)

    kpts_coords = L.Add(name="final_coords")([kpts_coords, h_offsets])
    kpts_coords = L.Reshape((2, -1))(kpts_coords)
    kpts_coords = L.Permute((2,1))(kpts_coords)
    
    # 6. Get the keypoint probability
    kpts_probas = L.Multiply()([k_heatmaps, jointmask])
    kpts_probas = L.GlobalMaxPooling2D()(kpts_probas)
    kpts_probas = L.Reshape((-1, 1))(kpts_probas)
    
    # 7. Concatenate the outputs together
    kpts_coords = L.Concatenate()([kpts_probas, kpts_coords, raw_kpts_coords])
    heatmaps = L.Concatenate()([centermap, k_heatmaps])

    post_proc = Model(inputs, [kpts_coords, heatmaps], name=name)
    
    return post_proc
