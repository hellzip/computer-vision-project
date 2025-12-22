import streamlit as st

st.set_page_config(page_title="QuickDraw Test", layout="wide")

st.title("Testing Model Loading")

try:
    st.write("Step 1: Importing inference_module...")
    from inference_module import DrawPredictor
    st.success("Module imported successfully!")
    
    st.write("Step 2: Creating predictor...")
    predictor = DrawPredictor()
    st.success("Predictor created!")
    
    st.write("Step 3: Checking class names...")
    st.write(f"Number of classes: {len(predictor.class_names)}")
    st.write(f"Classes: {predictor.class_names}")
    
    st.success("✅ Model loaded successfully! App should work now.")
    
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    import traceback
    st.code(traceback.format_exc())
