import streamlit as st
import argparse
import time
import cv2

from bodypose.graphics import draw_keypoints
from bodypose.model import BodyPoseNet
from config import MODELS_DICT

def _max_width_():
        max_width_str = f"max-width: 1400px;"
        st.markdown(
            f"""
        <style>
        .reportview-container .main .block-container{{
            {max_width_str}
        }}
        </style>    
        """,
            unsafe_allow_html=True,
        )

def main(args):

    cam = cv2.VideoCapture(-1)

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


        args["ui_window"].image(frame, channels="BGR")

        n_frames += 1
        inf_time += pred_time

        if (n_frames%100)==0:
            print(f"\r[INFO] Avearge fps: {n_frames/(time.time() - start):.2f}. Average inference time: {inf_time/n_frames:.3f} s.", end="")
    
    cam.release()


if __name__=="__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lightning", 
        help="Model name. Options: thunder, lighting. The default key is ligthing.")
    ap.add_argument("--thresh", type=float, default=.2,
        help="Detection threshold. The default value is 0.2.")
    ap.add_argument("--use_ui", type=bool, default=True)
    args = vars(ap.parse_args())

    st.set_page_config(
    page_title="Bodypose live estimator",
    page_icon="🎈",
    layout="wide"
    )
    
    _max_width_()
    st.title("Bodypose live estimator")

    args["ui_window"] = st.image([])

    # sidebar
    with st.sidebar:

        st.sidebar.title("Features")
        
        args["model"] = st.radio(
            "Model",
            ["lightning", "thunder"],
        )

        args["thresh"]  = st.slider(
            "Threshold",
            min_value=0.0,
            max_value=1.0,
            step=.05,
            value=.2,
        )

    main(args)
    