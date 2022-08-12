from .custom_layers import shortcut_block, residual_block, DepthWiseConv2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
import tensorflow.keras.layers as L
import numpy as np


def create_FB(input_shapes, alpha, use_depthwise=False, name="feature_block", stride=8):

    inputs = L.Input(input_shapes)

    if use_depthwise:
        x = DepthWiseConv2D(inputs, 64, (7,7), (1,1), use_bias=False, name=name+"_Conv1")
    else:
        x = L.Conv2D(64, (7,7), padding="same", use_bias=False, kernel_regularizer=l2(1e-4), name=name+"_Conv1")(inputs)
    x = L.BatchNormalization(momentum=.9, epsilon=2e-5, name=name+"bn_Conv1")(x)
    x = L.Activation("relu", name=name+"_Conv1_act")(x)

    y = shortcut_block(x, int(128*alpha), name=name+"_Short1")
    x = residual_block(x, int(64*alpha), depthwise_conv=use_depthwise, name=name+"_Res1")
    x = L.Add(name=name+"_Add1")([x, y])
    x = L.Activation("relu", name=name+"_Add1_act")(x)

    n_it = np.log2(stride).astype(int)
    for i in range(n_it):
        y = shortcut_block(x, int(128*alpha), strides=(2,2), name=name+f"_Short{i+2}")
        x = residual_block(x, int(64*alpha), strides=(2,2), depthwise_conv=use_depthwise, name=name+f"_Res{i+2}")
        x = L.Add(name=name+f"_Add{i+2}")([x, y])
        x = L.Activation("relu", name=name+f"_Add{i+2}_act")(x)
    
    return inputs, x  


if __name__=="__main__":
    inputs, outputs = create_FB((416,416,1), alpha=1)
    FB = Model(inputs, outputs)
    print(FB.summary())