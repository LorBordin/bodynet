import argparse
import time
import cv2

from bodypose.demo.graphics import draw_keypoints
from bodypose.demo.model import BodyPoseNet
from config import MODELS_DICT

def main(args):

    cam = cv2.VideoCapture(0)

    print("[INFO] Loading the model ...")
    model = MODELS_DICT["movenet_" + args["model"]]
    network = BodyPoseNet(model)
    
    # stats
    n_frames = 0
    inf_time = 0
    start = time.time()

    print("[INFO] Starting the loop ...")

    while True:

        grabbed, frame = cam.read()
        if not grabbed:
            break
        
        preds, pred_time = network.predict(frame)
        frame = draw_keypoints(frame, preds, args["thresh"], network.keypoints)

        if args["use_ui"]:
            args["ui_window"].image(frame)
        else:
            cv2.imshow("Canvas", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

        n_frames += 1
        inf_time += pred_time

        if (n_frames%100)==0:
            print(f"\r[INFO] Avearge fps: {n_frames/(time.time() - start):.2f}. Average inference time: {inf_time/n_frames:.3f} s.", end="")
    
    print()


if __name__=="__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lightning", 
        help="Model name. Options: thunder, lighting. The default key is ligthing.")
    ap.add_argument("--thresh", type=float, default=.2,
        help="Detection threshold. The default value is 0.2.")
    ap.add_argument("--use_ui", type=bool, default=False)
    args = vars(ap.parse_args())
    
    main(args)