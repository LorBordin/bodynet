import argparse
import cv2

from bodypose.graphics import draw_keypoints
from bodypose.model import BodyPoseNet
from config import MODELS_DICT

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lightning", 
        help="Model name. Options: thunder, lighting")
    ap.add_argument("--thresh", type=float, default=.5,
        help="Detection threshold (default 0.5).")
    args = vars(ap.parse_args())

    cam = cv2.VideoCapture(0)

    print("[INFO] Loading the model ...")
    model = MODELS_DICT["movenet_" + args["model"]]
    network = BodyPoseNet(model)

    print("[INFO] Starting the loop ...")

    while True:

        grabbed, frame = cam.read()
        if not grabbed:
            break
        
        preds, _ = network.predict(frame)
        frame = draw_keypoints(frame, preds, args["thresh"], network.keypoints)

        cv2.imshow("Canvas", frame)
        key =  cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            break

if __name__=="__main__":
    main()