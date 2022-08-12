from tensorflow_addons.activations import mish
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
import tensorflow.keras.layers as L

from .custom_layers import DepthWiseConv2D

def conv_3x3_module(inputs, out_channels, activation, name, use_depthwise):
    if use_depthwise:
        conv_3x3 = DepthWiseConv2D(inputs, out_channels, (3, 3), (1, 1), name=name+"_DWConv") 
    else:
        conv_3x3 = L.Conv2D(out_channels, (3, 3), padding="same", kernel_regularizer=l2(1e-4), name=name+"_Conv")(inputs)
    conv_3x3 = L.BatchNormalization(momentum=.9, epsilon=2e-5, name=name+"_bn_Conv")(conv_3x3)
    conv_3x3 = L.Activation(activation, name=name+"_Conv_act")(conv_3x3)
    return conv_3x3


def create_Head(inputs, 
                out_channels, 
                num_joints, 
                activation=mish, 
                use_depthwise=False,
                name="head"):
    """ Create Head for Classification and Regression on top of FPN that resembles CenterNet. """
    
    n_dim = 2

    # CenterMap Head
    c_name = name+"_centermap"
    centermap = conv_3x3_module(inputs, out_channels, activation, c_name, use_depthwise)
    centermap = L.Conv2D(1, (1, 1), name=c_name)(centermap)
    centermap = L.Activation("sigmoid", name=c_name+"_act")(centermap)
    
    # Keypoints offset from center
    k_name = name+"_k_offsets"
    k_offsets = conv_3x3_module(inputs, out_channels, activation, k_name, use_depthwise)
    k_offsets = L.Conv2D(num_joints*2, (1, 1), name=k_name)(k_offsets)
    #k_offsets = L.Activation("tanh", name=k_name+"_act")(k_offsets)

    # Keypoints heatmaps 
    h_name = name+"_k_heatmaps"
    k_heatmaps = conv_3x3_module(inputs, out_channels, activation, h_name, use_depthwise)
    k_heatmaps = L.Conv2D(num_joints, (1, 1), name=h_name)(k_heatmaps)
    k_heatmaps = L.Activation("sigmoid", name=h_name+"_act")(k_heatmaps)

    # Coordinates offsets
    c_name = name+"_c_offsets"
    c_offsets = conv_3x3_module(inputs, out_channels, activation, c_name, use_depthwise)
    c_offsets = L.Conv2D(num_joints * n_dim, (1, 1), name=c_name)(c_offsets)
    c_offsets = L.Activation("sigmoid", name=c_name+"_act")(c_offsets)
    
    # Depth coordinates
    depth_name = name+"_depth_coords"
    depth_input = L.Concatenate(axis=-1, name="depth_input")([inputs, k_heatmaps])
    depth_c = conv_3x3_module(depth_input, out_channels, activation, depth_name, use_depthwise)
    depth_c = L.Conv2D(num_joints, (1, 1), name=depth_name)(depth_c)
    depth_c = L.Activation("sigmoid", name=depth_name+"_act")(depth_c)


    outputs = [centermap, k_offsets, k_heatmaps, c_offsets, depth_c]
    
    return outputs


if __name__=="__main__":
    inputs = L.Input((52, 52, 128))
    outputs = create_Head_2D(inputs, 256, 17)
    head_2d = Model(inputs, outputs)
    print(head_2d.summary())