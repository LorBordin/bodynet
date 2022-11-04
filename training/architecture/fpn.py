from tensorflow.keras.regularizers import l2
from tensorflow_addons.activations import mish
from tensorflow.keras.models import Model
import tensorflow.keras.layers as L


def create_FPN(inputs, 
               in_channels, 
               activation=mish, 
               name="FPN"):
    """ 
        Returns a Feature Pyramid Network built on top of the backbone. 

        Parameters
        ----------
        inputs: list
            List of input layers from the backbone.
        in_channels: int
            Number of channels for each convolutional layer.
        activation: function
            Activation function applied after a convolutional layer.
            The default value is mish.
        name: str
            Model name.
        
        Returns
        -------
        fpn: keras.model 
            Feature pyramid network model.
    """
    first_head = True
    
    for i, in_layer in enumerate(inputs):
        
        if first_head:
            x = L.Conv2D(in_channels, (1,1), kernel_regularizer=l2(1e-4), name=name+f"_Conv{i+1}")(in_layer)
            x = L.BatchNormalization(momentum=.9, epsilon=2e-5, name=name+f"_bn_Conv{i+1}")(x)
            x = L.Activation(activation, name=name+f"_Conv{i+1}_act")(x)
            first_head = False
        
        else:
            y = L.Conv2D(in_channels, (1,1), kernel_regularizer=l2(1e-4), name=name+f"_Conv{i+1}")(in_layer)
            y = L.BatchNormalization(momentum=.9, epsilon=2e-5, name=name+f"_bn_Conv{i+1}")(y)
            y = L.Activation(activation, name=name+f"_Conv{i+1}_act")(y)
            x = L.UpSampling2D((2,2), interpolation="bilinear", name=name+f"_UpSamp{i+1}")(x)
            x = L.Add(name=name+f"_Add{i+1}")([x, y])
    
    fpn = Model(inputs, x,  name=name)
    
    return fpn


if __name__=="__main__":
    inputs = [L.Input((13, 13, 1280)), L.Input((26, 26, 576)), L.Input((52, 52, 192))]
    FPN = create_FPN(inputs, 128)
    print(FPN.summary())
