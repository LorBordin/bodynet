from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers as L

def DepthWiseConv2D(inputs, n_filters, kernel, strides, name, use_bias=True):
    
    x = L.DepthwiseConv2D(kernel_size=kernel, strides=strides, padding="same",  
                          use_bias=use_bias, kernel_regularizer=l2(1e-4), name=name+"inter")(inputs)
    x = L.Conv2D(n_filters, kernel_size=(1,1), strides=(1,1), padding="same", 
                          use_bias=use_bias, name=name+"intra")(x)
    return x


def residual_block(inputs, filters, kernel=(3,3), strides=(1,1), depthwise_conv=False, name=None):
        
    x = L.Conv2D(filters, (1,1), padding="same", use_bias=False, kernel_regularizer=l2(1e-4), name=name+"_Conv1")(inputs)
    x = L.BatchNormalization(momentum=.9, epsilon=2e-5, name=name+"_bn_Conv1")(x)
    x = L.Activation("relu", name=name+"_Conv1_act")(x)
    
    if depthwise_conv:
        x = DepthWiseConv2D(x, filters, kernel, strides, use_bias=False, name=name+"_DWConv2")
    else:
        x = L.Conv2D(filters, kernel, strides, padding="same", use_bias=False, name=name+"_Conv2")(x)
        
    x = L.BatchNormalization(momentum=.9, epsilon=2e-5, name=name+"bn_Conv2")(x)
    x = L.Activation("relu", name=name+"_Conv2_act")(x)
    
    x = L.Conv2D(filters*2, (1,1), padding="same", use_bias=False, kernel_regularizer=l2(1e-4), name=name+"_Conv3")(x)
    x = L.BatchNormalization(momentum=.9, epsilon=2e-5, name=name+"_bn_Conv3")(x)
    x = L.Activation("relu", name=name+"_Conv3_act")(x)
    
    return x


def shortcut_block(inputs, filters, strides=(1,1), name=None):
    
    x = L.Conv2D(filters, (1,1), strides, padding="same", use_bias=False, kernel_regularizer=l2(1e-4), name=name+"_Conv1")(inputs)
    x = L.BatchNormalization(momentum=.9, epsilon=2e-5, name=name+"_bn_Conv1")(x)
    x = L.Activation("relu", name=name+"Conv1_acts")(x)
    
    return x
