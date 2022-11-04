import argparse
import time
import cv2
import av

from bodypose.demo.graphics import draw_keypoints

class VideoProcessor():
    
    def __init__(self):
        
        # model
        self.model_name = None 
        self.model = None 
        self.thresh = None

        # stats
        # self.n_frames = 0
        # self.inf_time = 0       
    

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")
            
        preds, pred_time = self.model.predict(img)
        img = draw_keypoints(img, preds, self.thresh, self.model.keypoints)

        #self.n_frames += 1
        #self.inf_time += pred_time

        return av.VideoFrame.from_ndarray(img, format="bgr24")
