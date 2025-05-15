# app/app.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms as T
from PIL import Image
import os
import numpy as np
import pandas as pd
import altair as alt
from datetime import datetime

# Constants
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model/ResNet18.pth')
ASSETS_PATH = os.path.join(os.path.dirname(__file__), '../assets/logo.png')

# Streamlit Configuration
st.set_page_config(
    page_title="NeuroViT - AI Brain Stroke Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner=False)
def load_model():
    """Load and cache the trained ResNet18 model"""
    try:
        # Initialize model architecture
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at: {MODEL_PATH}")
            st.stop()
            
        # Load weights with error handling
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=device),
            strict=True
        )
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Initialize session state
if 'app_state' not in st.session_state:
    st.session_state.app_state = {
        'model': load_model(),
        'transform': T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'last_prediction': None,
        'prediction_history': []
    }

# Helper Functions
def save_prediction_to_history(prediction):
    """Save prediction to session history"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.app_state['prediction_history'].append({
        'timestamp': timestamp,
        'prediction': prediction['result'],
        'confidence': prediction['confidence'],
        'image_name': prediction.get('image_name', 'uploaded_image')
    })

def display_prediction_history():
    """Display prediction history in sidebar"""
    if st.session_state.app_state['prediction_history']:
        st.sidebar.subheader("Recent Predictions")
        for i, pred in enumerate(reversed(st.session_state.app_state['prediction_history'])):
            with st.sidebar.expander(f"{pred['timestamp']}: {pred['prediction']}"):
                st.write(f"**Confidence:** {pred['confidence']*100:.1f}%")
                st.write(f"**Image:** {pred['image_name']}")

# UI Components
def main_header():
    """Main header section"""
    col1, col2 = st.columns([1, 3])
    with col1:
        try:
            st.image(ASSETS_PATH, width=150)
        except:
            st.warning("Logo image not found")
    with col2:
        st.title("NeuroViT Brain Stroke Detection")
        st.caption("AI-powered analysis of brain scans for stroke detection")

def file_uploader_section():
    """File uploader and processing section"""
    uploaded_file = st.file_uploader(
        "Upload a brain scan image (CT/MRI)",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            image_name = uploaded_file.name
            
            with st.spinner("Analyzing image..."):
                # Transform and predict
                img_tensor = st.session_state.app_state['transform'](image).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = st.session_state.app_state['model'](img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Get results
                predicted_class = probabilities.argmax().item()
                confidence = probabilities[0][predicted_class].item()
                class_labels = {0: 'No Stroke Detected', 1: 'Stroke Detected'}
                
                # Store prediction
                prediction = {
                    'result': class_labels[predicted_class],
                    'confidence': confidence,
                    'probabilities': probabilities.cpu().numpy()[0],
                    'image': image,
                    'image_name': image_name
                }
                st.session_state.app_state['last_prediction'] = prediction
                save_prediction_to_history(prediction)
                
            display_results(prediction)
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

def display_results(prediction):
    """Display prediction results"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(
            prediction['image'], 
            caption=f"Uploaded Image: {prediction['image_name']}",
            use_container_width=True
        )
    
    with col2:
        # Prediction and confidence
        st.subheader("Analysis Results")
        
        if prediction['result'] == "Stroke Detected":
            st.error(f"**Prediction:** ⚠️ {prediction['result']}")
        else:
            st.success(f"**Prediction:** ✅ {prediction['result']}")
            
        st.metric(
            "Confidence Level", 
            f"{prediction['confidence']*100:.2f}%",
            delta="High confidence" if prediction['confidence'] > 0.85 else "Moderate confidence",
            delta_color="normal"
        )
        
        # Probability chart
        prob_df = pd.DataFrame({
            'Condition': ['No Stroke', 'Stroke'],
            'Probability': prediction['probabilities']
        })
        
        chart = alt.Chart(prob_df).mark_bar().encode(
            x='Condition',
            y='Probability',
            color=alt.condition(
                alt.datum.Condition == prediction['result'],
                alt.value('#FF4B4B' if prediction['result'] == "Stroke Detected" else '#00D154'),
                alt.value('lightgray')
            )
        ).properties(height=300)
        
        st.altair_chart(chart, use_container_width=True)
        
        # Clinical notes
        st.subheader("Clinical Notes")
        if prediction['result'] == "Stroke Detected":
            st.warning("""
            **Clinical Correlation Recommended**  
            This AI result suggests potential stroke findings.  
            Immediate review by a qualified radiologist is advised.
            """)
        else:
            st.info("""
            **Routine Follow-Up Recommended**  
            No signs of stroke detected in this analysis.  
            Continue with standard clinical protocols.
            """)

def sidebar_content():
    """Sidebar content"""
    st.sidebar.header("About NeuroViT")
    st.sidebar.info("""
    This application uses a fine-tuned ResNet18 model trained on the 
    BTX24 brain-stroke-dataset to detect potential stroke indicators 
    in medical images.
    """)
    
    st.sidebar.markdown("### How to Use")
    st.sidebar.write("""
    1. Upload a brain scan (CT/MRI)
    2. Wait for AI analysis
    3. Review results and clinical notes
    """)
    
    st.sidebar.markdown("### Model Information")
    st.sidebar.code(f"""
    Framework: PyTorch {torch.__version__}
    Model: ResNet18
    Device: {'GPU' if torch.cuda.is_available() else 'CPU'}
    Last loaded: {datetime.now().strftime("%Y-%m-%d %H:%M")}
    """)
    
    display_prediction_history()

def disclaimer_footer():
    """Application footer"""
    st.markdown("---")
    st.caption("""
    **Disclaimer:** This AI tool is for research and assistive purposes only.  
    Not intended for direct clinical diagnosis without human oversight.  
    NeuroViT v1.0 | [GitHub Repository](https://github.com/OcamaAslam/NeuroViT)  
    © 2025 NeuroViT Project | All rights reserved | [Engr. Muhammad Osama](https://linkedin.com/in/ocama-mohamed)  
    [Get connected with me on LinkedIn](https://linkedin.com/in/ocama-mohamed)
    """)

# Main App Flow
def main():
    main_header()
    file_uploader_section()
    sidebar_content()
    disclaimer_footer()

if __name__ == "__main__":
    main()