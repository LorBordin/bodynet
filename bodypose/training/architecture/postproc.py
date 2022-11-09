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
    
    heatmaps = inputs

    num_joints = heatmaps.shape[-1]
    
    # apply softmax to weigthed Heatmaps
    #w_k_heatmaps = L.Reshape((grid_dim * grid_dim, num_joints))(w_k_heatmaps)
    #w_k_heatmaps = L.Softmax()(w_k_heatmaps)
    #w_k_heatmaps = L.Reshape((grid_dim, grid_dim, num_joints), name="weighted_heatmaps")(w_k_heatmaps)

    jointmasks = L.Lambda(lambda x: get_max_mask(x))(inputs)

    # Get the joints coordinates
    xx = L.Lambda(lambda x: grid_coords(x, axis=1))(jointmasks)
    x_j = L.Multiply()([xx, jointmasks])
    x_j = L.GlobalMaxPooling2D()(x_j)

    yy = L.Lambda(lambda x: grid_coords(x, axis=0))(jointmasks)
    y_j = L.Multiply()([yy, jointmasks])
    y_j = L.GlobalMaxPooling2D()(y_j) 

    probas = L.Multiply()([heatmaps, jointmasks])
    probas = L.GlobalMaxPooling2D()(probas)
    
    # Reshape before concatenate the outputs
    probas = L.Reshape((-1, 1))(probas)
    x_j = L.Reshape((-1, 1))(x_j)
    y_j = L.Reshape((-1, 1))(y_j)

    coords = L.Concatenate()([probas, x_j, y_j])

    post_proc = Model(inputs, [coords, heatmaps], name=name)
    
    return post_proc
