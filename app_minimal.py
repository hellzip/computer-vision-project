import streamlit as st

st.set_page_config(page_title="QuickDraw Test", layout="wide")

st.title("QuickDraw App - Deployment Test")
st.success("App is running successfully!")

st.write("If you see this, the basic Streamlit deployment works.")
st.write("Now we can debug the model loading issue.")

try:
    import torch
    st.success(f"PyTorch installed: {torch.__version__}")
except Exception as e:
    st.error(f"PyTorch error: {e}")

try:
    import cv2
    st.success("OpenCV installed")
except Exception as e:
    st.error(f"OpenCV error: {e}")

try:
    from streamlit_drawable_canvas import st_canvas
    st.success("Drawable canvas installed")
except Exception as e:
    st.error(f"Canvas error: {e}")

try:
    import os
    files = os.listdir(".")
    st.write("Files in directory:", files)
    
    if "quickdraw_mlp.pth" in files:
        st.success("Model file found!")
    else:
        st.error("Model file NOT found")
        
    if "class_map.json" in files:
        st.success("Class map found!")
    else:
        st.error("Class map NOT found")
        
except Exception as e:
    st.error(f"File system error: {e}")
