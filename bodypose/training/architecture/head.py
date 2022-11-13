from tensorflow_addons.activations import mish
from tensorflow.keras.models import Model
import tensorflow.keras.layers as L

from .custom_layers import Conv3x3Module
from .custom_layers import SpatialAttentionModule

def create_Head(inputs, 
                out_channels, 
                num_joints, 
                activation=mish, 
                use_depthwise=False,
                name="head"):
    """ 
        Returns the head of the pose estimation model on top of the FPN.
        It is composed by 4 separate heads:
            - centermap that predicts the persons center,
            - keypoints offesets: keypoints offsets from the person center of the most centered person,
            - keypoints heatmaps: keypoints  heatmpas of the most centered person,
            - coordinate offsets: keypoints offsets from the peak of the respective heatmap.

        Parameters
        ----------
        inputs: keras.layer
            Input layer (from the FPN).
        out_channels: int
            Number filters of the internal convolutional layers.
        num_joints: int
            Number of predicted joints.
        activation: function
            Activation function of the internal convolutional layers
        use_depthwise: bool
            If True uses depthwise convolution instead of standard convolution for the internal layers.
        name: str
            Model name.

        Returns
        -------
        head: keras.model
            Head model.
    """
    
    n_dim = 2

    # CenterMap Head
    c_name = name+"_centermap"
    centermap = Conv3x3Module(out_channels, activation, c_name, use_depthwise)(inputs)
    attention  = SpatialAttentionModule(1)(centermap)
    centermap = L.Conv2D(1, (1, 1), name=c_name)(centermap)
    centermap = L.Multiply()([centermap, attention])
    centermap = L.Activation("sigmoid", name=c_name+"_act")(centermap)

    # Keypoints offset from center
    k_name = name+"_k_offsets"
    k_offsets = Conv3x3Module(out_channels, activation, k_name, use_depthwise)(inputs)
    k_offsets = L.Conv2D(num_joints*2, (1, 1), name=k_name)(k_offsets)
    k_offsets = L.Activation("tanh", name=k_name+"_act")(k_offsets)
    
    # Keypoints heatmaps 
    h_name = name+"_k_heatmaps"
    k_heatmaps = Conv3x3Module(out_channels, activation, h_name, use_depthwise)(inputs)
    attention = SpatialAttentionModule(num_joints)(k_heatmaps)
    k_heatmaps = L.Conv2D(num_joints, (1, 1), name=h_name)(k_heatmaps)
    k_heatmaps = L.Multiply()([k_heatmaps, attention])
    k_heatmaps = L.Activation("sigmoid", name=h_name+"_act")(k_heatmaps)

    # Coordinates offsets
    c_name = name+"_c_offsets"
    c_offsets = Conv3x3Module(out_channels, activation, c_name, use_depthwise)(inputs)
    c_offsets = L.Conv2D(num_joints * n_dim, (1, 1), name=c_name)(c_offsets)
    c_offsets = L.Activation("sigmoid", name=c_name+"_act")(c_offsets)
    
    outputs = [centermap, k_offsets, k_heatmaps, c_offsets]
    
    head  = Model(inputs, outputs, name=name)
    
    return head


if __name__=="__main__":
        
    inputs = L.Input((52, 52, 128))
    head_2d = create_Head(inputs, 256, 17)
    print(head_2d.summary())