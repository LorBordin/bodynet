from tensorflow.keras.models import Model
from tensorflow import keras

def _add_prefix_layers_name(model, prefix):
    """
        Changes the layers name by adding a given prefix.\

        Parameters
        ----------
        model : keras.model
            Input model.
        prefix : str
            Prefix to be added at each layer's name.

        Returns
        -------
        None
    """
    for layer in model.layers:
        layer._name = prefix + layer.name


def get_features_layers(model, strides, img_shape):
    """ 
        Finds and returns a list of convolutional layers from a given backbone model
        that will be later passed to the FPN.
        Example: given an img_shape=(416, 416, 3) and a strides=[32, 16, 8] it will 
            return those layers such that the ouput images have dimensions 13, 26, 52.
        
        Parameters
        ----------
        backbone: keras.model
            Input convolutional model that acts as a backbone.
        strides: list 
            List of multipliers (int) that are usewd to  select the layers.
        img_shape : tuple
            Dimension of the img. The Width is expected to be equal to the Height.

        Returns
        -------
        layers : list
            List of layers that match the corresponding strides.
    """
    f_layers = []
    counter = 0

    for i, layer in enumerate(model.layers[::-1]):
    
        shape = img_shape[0] // strides[counter]
    
        if shape in layer.output_shape:
            l_index = len(model.layers) - 1 - i
            f_layers.append(l_index)
            counter+=1
        
        if counter == len(strides):
            break
    
    layers = [model.layers[i].output for i in f_layers]

    return layers


def create_backbone(input_shape, strides, alpha=1, arch="mobilenetV2", name="backbone"):
    """ 
        Build backbone on top given convolutional model.

        Parameters
        ----------
        input_shape: tuple
            Input image shape.
        strides: tuple
            List of multipliers (int) that are usewd to  select the layers.
        alpha: float
            Depth parameter that controls the number of filters of each convolutional layer.
            The default value is 1.
        arch: str
            Input model architecture. The current options are MobileNetV2 and MobileNetV3.
            The default value is MobileNetV2.
        name: str
            Name of the backbone.

        Returns
        -------
        backbone : keras.model
            Backbone model build from the given architecture.
    """
    
    if arch=="mobilenetV2":
        weights = "imagenet" if alpha in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4] else None
        model = keras.applications.MobileNetV2(include_top=False, 
                                               weights=weights, 
                                               input_shape=input_shape,
                                               alpha=alpha)
    elif arch=="mobilenetV3":
        weights = "imagenet" if alpha in [0.75] else None
        model = keras.applications.MobileNetV3Large(include_top=False, 
                                                    weights=weights, 
                                                    input_shape=input_shape,
                                                    alpha=alpha)
    else:
        print("[ERROR] Undefined backbone architecture.")
                                       
    _add_prefix_layers_name(model, prefix=name+"_")
    in_layers = model.layers[0].input
    out_layers = get_features_layers(model, strides, input_shape)
    
    backbone = Model(in_layers, out_layers)
    return backbone


if __name__=="__main__":

    backbone = create_backbone((416, 416, 3), strides=(32,16,8)) 
    print(backbone.summary())