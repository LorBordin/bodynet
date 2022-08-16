from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers as L


class DepthWiseConv2D(L.Layer):
    """ 
        Depthwise 2D Convolution class. It firstly forwards the input to a depthwise 
        convolutional layer, with respectively kernel and strides kernel_size  and strides, 
        then the output is forwarded to a Conv2D layer with n_filters and (1,1) kernel_size 
        and strides.

        Parameters
        ----------
        n_filters: int
            Number of output filters.
        kernel: tuple
            Size of the kernel of the depthwise convolutional layer.
        strides: tuple
            Strides of the depthwise convolutional layer.
        name: str
            name of the layer
        use_bias: bool
            If True each internal layer uses have use_biasseto to True. 
    """
    def __init__(self, n_filters, kernel, strides, name, use_bias=True):
        super().__init__()
        self.depthwise_conv = L.DepthwiseConv2D(kernel_size=kernel, 
                                                strides=strides, 
                                                padding="same",
                                                use_bias=use_bias,
                                                kernel_regularizer=l2(1e-4),
                                                name=name+"inter")
        self.conv = L.Conv2D(n_filters, 
                             kernel_size=(1,1), 
                             strides=(1,1),
                             padding="same",
                             use_bias=use_bias,
                             name=name+"intra")

    def call(self, inputs):
        x = self.depthwise_conv(inputs)
        return self.conv(x)          


class Conv3x3Module(L.Layer):
    """
        Convolutional block class, composed by a 3x3 convolution, batch_normalization and 
        activation function.

        Parameters
        ----------
        n_filters: int
            Number of output filters.
        activation: function
            Activation function. The default value if mish.
        name: str
            Layer name.
        use_deptwise: if True uses a DepthWiseConv2D layer instead of Conv2D
    """
    def __init__(self, n_filters, activation, name, use_depthwise):
        super().__init__()
        
        if use_depthwise:
            self.conv_3x3 = DepthWiseConv2D(n_filters, (3,3), (1,1), name=name+"_DWConv")
        else:
            self.conv_3x3 =  L.Conv2D(n_filters, (3,3), 
                                      padding="same", kernel_regularizer=l2(1e-4),  name=name+"_Conv")

        self.bn = L.BatchNormalization(momentum=.9, epsilon=2e-5, name=name+"_bn_Conv")
        self.act = L.Activation(activation, name=name+"_Conv_act")
    
    def call(self, inputs):
        x = self.conv_3x3(inputs)
        x = self.bn(x)
        return self.act(x)