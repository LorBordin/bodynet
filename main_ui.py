from streamlit_webrtc import webrtc_streamer
import streamlit as st
import argparse

from bodypose.videoprocessor import VideoProcessor
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

    ctx = webrtc_streamer(key="example", 
        video_processor_factory=VideoProcessor,
        rtc_configuration={ # Add this line
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

    if ctx.video_processor:
        ctx.video_processor.model_name = MODELS_DICT["movenet_" + args["model"]]
        ctx.video_processor.model = BodyPoseNet(ctx.video_processor.model_name)
        ctx.video_processor.thresh = args["thresh"]