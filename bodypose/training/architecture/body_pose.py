from tensorflow.keras.models import Model
import tensorflow.keras.layers as L  
import tensorflow as tf

import os

from .postproc import create_postproc_model
from .backbone import create_backbone
from .head import create_Head
from .fpn import create_FPN
    
class MoveNet(tf.keras.Model):
    """
        Unoffcial implemetation of the Google Movenet, built from the Centernet.
        
        Parameters
        ----------
        input_shapes: tuple
            Input image height, width and channels.
        strides: list           
            List of int corresponding to the desired strides.
        num_joints: int
            Number of body keypoints.
        alpha: float
            Depth factor that sets the number of cpnvolutional filters.  
            The default value is 1.
        backbone_arch: str
            Backbone architecture name. The current options are 
            "mobileNetV2" (default) or "mobileNetV3".
        use_depthwise: bool
            If True uses depthwise convolutions instead of normal convolutions.
    """
    def __init__(self, 
                 input_shape, 
                 strides, 
                 num_joints, 
                 alpha=1, 
                 backbone_arch="mobilenetV2", 
                 use_depthwise=False,
                 use_postproc=False):

        super().__init__()
        self.use_postproc = use_postproc
        self.grid_size = input_shape[0] // strides[-1]

        self.inputs = L.Input(input_shape)
        self.backbone = create_backbone(input_shape, strides, alpha, arch=backbone_arch)
        self.fpn = create_FPN(inputs=self.backbone.outputs, in_channels=int(128*alpha))
        self.head = create_Head(inputs=self.fpn.output, 
                           num_joints=num_joints, 
                           out_channels=int(256*alpha),
                           use_depthwise=use_depthwise)
        self.postproc = create_postproc_model(inputs=self.head.outputs)

    def call(self, inputs):
        x = self.backbone(inputs)
        x = self.fpn(x)
        x = self.head(x)
        if self.use_postproc:
            x = self.postproc(x)
        return x


if __name__=="__main__":

    model = MoveNet((416, 416, 3), (32, 16, 8), 17)
    print(model.summary())
    print()
    
    import numpy as np
    import time

    img = (np.random.randint(0, 255, (1, 416, 416, 3)) / 255).astype(np.float32)
    
    start = time.time()
    for i in range(100):
        model(img)
    end = time.time()
    print(f"[INFO] Average inference time: {(end-start)/100 * 1000:.1f} ms.")