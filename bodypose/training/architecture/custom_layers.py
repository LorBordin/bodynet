from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras import layers as L
import tensorflow as tf 

EPSILON = 1e-2

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


class ExtractCoordinates(L.Layer):
    def __init__(self, n_rep):
        super().__init__()

        self.n_rep = n_rep

        self.x_mesh = L.Lambda(lambda x: grid_coords(x, axis=1))
        self.y_mesh = L.Lambda(lambda x: grid_coords(x, axis=0))
        self.mult = L.Multiply()
        self.glob_max = L.GlobalMaxPooling2D()
        self.concat =  L.Concatenate()
        
    def call(self, inputs):

        x = self.x_mesh(inputs)
        x = self.mult([x, inputs])
        x = self.glob_max(x)
        x = self.concat([x] * self.n_rep)

        y = self.y_mesh(inputs)
        y = self.mult([y, inputs])
        y = self.glob_max(y)
        y = self.concat([y] * self.n_rep)
        
        coords = self.concat([x, y])

        return coords 


class SpatialAttentionModule(L.Layer):
    def __init__(self, n_filters):
        self.n_filters = n_filters
        self.kernel_sizev= 7

        self.avg_pool = L.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))
        self.max_pool = L.Lambda(lambda x: K.max(x, axis=3, keepdims=True))
        self.concat = L.Concatenate(axis=3)
        self.conv = L.Conv2D(
            filters = n_filters,
			kernel_size = self.kernel_size,
			strides=1,
			padding='same',
			activation='sigmoid',
			kernel_initializer='he_normal',
			use_bias=False
            )	
        self.multiply = L.Multiply()
        self.act = L.Activation("sigmoid")

    def call(self, inputs):
        avg_pool = self.avg_pool(inputs)
        max_pool = self.max_pool(inputs)
        x = self.concat([avg_pool, max_pool])
        x = self.conv(x)
        x = self.act(x)
        x = self.multiply([inputs, x])
        return x



		
	return multiply([input_feature, cbam_feature])

def grid_coords(x, axis):
    """
        Given a tensor of shape (batch_size, grid_dim, grid_dim, channels), returns a tensor with
        the same shape whose entries are  given by the x(y) coord of the grid normalised to one.

        Example:
        x = tf.ones((1,5,5,2))
        y = grid_coords(x, axis=0)
        y[0,:,:,0] = y[0,:,:,1] = <tf.Tensor: shape=(5, 5), dtype=float32, numpy=
                                  array([[0. , 0. , 0. , 0. , 0. ],
                                         [0.2, 0.2, 0.2, 0.2, 0.2],
                                         [0.4, 0.4, 0.4, 0.4, 0.4],
                                         [0.6, 0.6, 0.6, 0.6, 0.6],
                                         [0.8, 0.8, 0.8, 0.8, 0.8]], dtype=float32)>

        Parameters
        ----------
        x:  tf.Tensor
            Input tensor of shape (batch_size, grid_dim, grid_dim, channels).
        axis: int
            Coordinates axis, 0:y, 1:x.

        Returns
        -------
        coords_tensor: tf.Tensor
            Output tensor with the same size as input.

    """
    y = tf.ones_like(x, dtype=tf.float32)
    grid_dim = x.shape[1]
    num_joints = x.shape[-1]
    coords_grid = tf.range(tf.cast(grid_dim, tf.float32)) / tf.cast(grid_dim, tf.float32)                      # vector that ranges from 0 to 1 with step 1/grid_dim
    coords_grid = tf.stack([coords_grid] * grid_dim)                # 2d tensor
    if axis==0:                                                     # x --> y coords
        coords_grid = tf.transpose(coords_grid)
    coords_grid = tf.stack([coords_grid] * num_joints, axis=-1) # tile num_joints tensors together
    coords_tensor = y * coords_grid
    return coords_tensor


def get_max_mask(x):
    """ 
        Given a tensor of shape (batch_size, grid_dim, grid_dim, channels), returns a boolean mask 
        which is non-zero in the position of the channel maxima (along the dimensions grid_dim).

        Example    
        x = np.zeros((1, 5,  5, 2))
        x[0,2,3,0] = 4
        x[0,1,2,0] = 4.5
        y =  get_max_mask(x)
        
        y[0,:,:,0] = <tf.Tensor: shape=(5, 5), dtype=float32, numpy=
                     array([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]], dtype=float32)>
        
        y[0,:,:,1] = <tf.Tensor: shape=(5, 5), dtype=float32, numpy=
                     array([[0., 0., 0., 0., 0.],
                            [0., 0., 1., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]], dtype=float32)>
    
    Parameters
    ----------
    x: tf.Tensor
        Input tensor of shape (batch_size, grid_dim, grid_dim, channels).

    Returns
    -------
    mask : tf.Tensor
        Output tensor with the same size as input.

    """
    size = x.shape[1]
    channels = x.shape[-1]
    x = tf.reshape(x, (-1, size*size, channels))
    y = tf.cast(tf.equal(x, tf.expand_dims(tf.math.reduce_max(x, axis=1), axis=1)), tf.float32)
    mask = tf.reshape(y, (-1, size, size, channels))
    return mask


def get_inverse_dist_grid(item):
    """
        Given a tensor of  shape (batch_size, grid_dim, grid_dim, channels) and a (list of)
        offset(s), returns a tensor with the same shape, whose entries are inverse proportional 
        to the euclidean distance from the corresponding grid centre to the offset.

        Example:
        x = tf.zeros((1, 5,  5, 2))
        offsets = tf.convert_to_tensor([.5, .2])
        y = get_inverse_dist_grid((x, offsets))

        y[0,:,:,0] = <tf.Tensor: shape=(5, 5), dtype=float32, numpy=
                     array([[1.40028  , 1.6903085, 1.924501 , 1.924501 , 1.6903085],
                            [1.6903085, 2.2941573, 3.0151134, 3.0151134, 2.2941573],
                            [1.924501 , 3.0151134, 5.773503 , 5.7735023, 3.0151134],
                            [1.924501 , 3.0151134, 5.7735023, 5.773502 , 3.0151134],
                            [1.6903085, 2.2941573, 3.0151134, 3.0151134, 2.2941573]],
                           dtype=float32)>

        y[0,:,:,1] = <tf.Tensor: shape=(5, 5), dtype=float32, numpy=
                     array([[ 3.3333333,  4.4721355,  3.3333333,  2.1821787,  1.5617375],
                            [ 4.4721355, 10.       ,  4.4721355,  2.425356 ,  1.6439899],
                            [ 3.3333333,  4.4721355,  3.3333333,  2.1821787,  1.5617375],
                            [ 2.1821787,  2.425356 ,  2.1821787,  1.7407765,  1.3736056],
                            [ 1.5617375,  1.6439899,  1.5617375,  1.3736056,  1.1704115]],
                           dtype=float32)>

        Parameters
        ----------
        x: tf.Tensor
            Input tensor of shape (batch_size, grid_dim, grid_dim, channels).
        offsets:  tf.Tensor
            Tensor of the normalised offsets. It can be single valued (same offset for each map)
            or length=channels (different offset  for each channel).
        
        Returns
        -------
        inv_dist: tf.Tensor
            Output tensor with the same shape  as input.
    """

    x, offsets = item 

    X_ones = tf.ones_like(x)
    if len(offsets.shape)>1:
        offset_x = offsets[:, :, 0]
        offset_y = offsets[:, :, 1]
        offset_x = tf.transpose(tf.transpose(X_ones, (1, 2, 0, 3)) * offset_x, (2, 0, 1, 3))
        offset_y = tf.transpose(tf.transpose(X_ones, (1, 2, 0, 3)) * offset_y, (2, 0, 1, 3))
    else:
        offset_x = offset_y = offsets

    xx = grid_coords(x, axis=1)
    xx = tf.math.pow(xx - offset_x, 2)

    yy = grid_coords(x, axis=0)
    yy = tf.pow(yy - offset_y, 2)

    dist = tf.math.sqrt(xx + yy + EPSILON)
    inv_dist = 1.0 / dist

    return inv_dist