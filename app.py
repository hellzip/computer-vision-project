import streamlit as st
import numpy as np
import cv2
import time
import random
import json

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st.error("Missing dependency: pip install streamlit-drawable-canvas")
    st.stop()

from inference_module import DrawPredictor

st.set_page_config(page_title="QuickDraw Game", layout="wide")

st.markdown("""
<style>
.target-text {font-size: 32px; font-weight: bold; color: #1f77b4; text-align: center; padding: 20px;}
.guess-box {padding: 12px; margin: 8px 0; background: #f0f4f8; border-radius: 6px; border-left: 4px solid #1f77b4;}
.guess-correct {background: #d4edda; border-left-color: #28a745;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    return DrawPredictor()

try:
    predictor = load_predictor()
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

if "target" not in st.session_state:
    st.session_state.target = None
if "active" not in st.session_state:
    st.session_state.active = False
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "attempts" not in st.session_state:
    st.session_state.attempts = 0
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None
if "win_time" not in st.session_state:
    st.session_state.win_time = None

st.title("Live Sketch")
st.caption("Draw an object and let the AI guess what you're drawing!")

with st.sidebar:
    st.header("Game Controls")
    
    if st.button("Random Target", use_container_width=True):
        st.session_state.target = random.choice(predictor.class_names)
        st.session_state.active = True
        st.session_state.start_time = time.time()
        st.session_state.attempts = 0
        st.session_state.canvas_key += 1
        st.session_state.last_pred = None
        st.session_state.win_time = None
    
    target = st.selectbox("Or pick a class:", predictor.class_names)
    if st.button("Start Game", use_container_width=True):
        st.session_state.target = target
        st.session_state.active = True
        st.session_state.start_time = time.time()
        st.session_state.attempts = 0
        st.session_state.canvas_key += 1
        st.session_state.last_pred = None
        st.session_state.win_time = None
    
    if st.button("Clear Canvas", use_container_width=True):
        st.session_state.canvas_key += 1

if st.session_state.get("active") and st.session_state.get("target"):
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown(f"<div class='target-text'>Draw: {st.session_state.get('target')}</div>", unsafe_allow_html=True)
        
        canvas = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=4,
            stroke_color="black",
            background_color="white",
            height=300,
            width=300,
            drawing_mode="freedraw",
            update_streamlit=True,
            key=f"canvas_{st.session_state.get('canvas_key', 0)}"
        )
    
    with col2:
        st.subheader("Predictions")
        
        elapsed = time.time() - st.session_state.get("start_time", time.time()) if st.session_state.get("start_time") else 0
        st.metric("Time", f"{elapsed:.1f}s")
        st.metric("Attempts", st.session_state.get("attempts", 0))
        
        st.write("**Top Guesses:**")
        
        if canvas is not None and canvas.image_data is not None:
            rgba = canvas.image_data.astype(np.uint8)
            gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
            img = gray  # Keep black strokes on white, consistent with training

            # Skip prediction if canvas is effectively empty
            if np.count_nonzero(img < 200) < 10:
                st.info("Draw a bit more for a confident prediction.")
                preds = []
            else:
                # Let errors bubble up to see the full traceback
                preds = predictor.predict_topk(img, k=3)

            if preds:
                top_label, top_conf = preds[0]
            else:
                st.info("No predictions available yet.")
                top_label, top_conf = None, None
            
            last_pred = st.session_state.get("last_pred")
            if top_label is not None and last_pred != top_label:
                st.session_state.last_pred = top_label
                st.session_state.attempts = st.session_state.get("attempts", 0) + 1
            
            for label, conf in preds:
                is_match = label == st.session_state.get("target")
                style = "guess-correct" if is_match else ""
                st.markdown(f"<div class='guess-box {style}'>{label}: {conf*100:.1f}%</div>", unsafe_allow_html=True)
            
            if top_label is not None and top_label == st.session_state.get("target"):
                if st.session_state.get("win_time") is None:
                    st.session_state.win_time = time.time()
                st.success(f"Correct! It's a {top_label}!")
                # Auto-change target after 5 seconds
                if time.time() - st.session_state.get("win_time", time.time()) >= 5:
                    st.session_state.target = random.choice(predictor.class_names)
                    st.session_state.start_time = time.time()
                    st.session_state.attempts = 0
                    st.session_state.canvas_key += 1
                    st.session_state.last_pred = None
                    st.session_state.win_time = None
                    st.rerun()
        else:
            st.info("Start drawing...")

else:
    st.info("Select a target from the sidebar to start playing!")
    
    with st.expander("View all 8 classes"):
        cols = st.columns(3)
        for i, cls in enumerate(predictor.class_names):
            with cols[i % 3]:
                st.write(f"• {cls}")