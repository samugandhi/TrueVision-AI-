import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageDraw
import pytesseract
from gtts import gTTS
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import os
import tempfile
import pygame

# Set API Key (Avoid exposing sensitive information in production)
genai.configure(api_key="Your API Key here")

# Streamlit App Configuration
st.set_page_config(
    page_title="VisionMate AI 🌟",
    layout="wide",
    page_icon="🌐"
)

# Utility Functions
def image_to_bytes(uploaded_file):
    try:
        bytes_data = uploaded_file.getvalue()
        return [{"mime_type": uploaded_file.type, "data": bytes_data}]
    except Exception as e:
        raise FileNotFoundError(f"Error processing image: {e}")

def extract_text_from_image(uploaded_file):
    try:
        img = Image.open(uploaded_file)
        extracted_text = pytesseract.image_to_string(img)
        return extracted_text if extracted_text.strip() else "No text found in the image."
    except Exception as e:
        raise ValueError(f"Error extracting text: {e}")

def text_to_speech(text):
    try:
        tts = gTTS(text, lang='en')
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            tts.save(temp_file.name)
            pygame.mixer.init()
            pygame.mixer.music.load(temp_file.name)
            pygame.mixer.music.play()
    except Exception as e:
        raise RuntimeError(f"Error converting text to speech: {e}")

@st.cache_resource
def load_object_detection_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

object_detection_model = load_object_detection_model()

def detect_objects(image, threshold=0.5):
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image)
    predictions = object_detection_model([img_tensor])[0]
    return predictions

# App Design and Features
st.markdown("""
    <style>
        body {
            background-color: #f4f5f7;
        }
        .stButton>button {
            border-radius: 12px;
            padding: 8px 16px;
            background-color: #0073e6;
            color: white;
            font-size: 14px;
        }
        .stButton>button:hover {
            background-color: #005bb5;
        }
    </style>
""", unsafe_allow_html=True)

st.title("VisionMate AI 🌐")
st.subheader("Empowering Visual Assistance with AI 💡")

# Sidebar
st.sidebar.header("✨ Features")
st.sidebar.markdown("""
- **Visual Insights**: Get detailed descriptions of uploaded images.
- **Object Detection**: Identify and highlight objects in your image.
- **Text Reader**: Extract and narrate text from images.
- **Accessibility Aid**: Convert extracted information to audio for ease of use.
""")

# File Uploader
uploaded_file = st.file_uploader("Upload an image to explore AI-powered insights:", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image Preview", use_column_width=True)

# Buttons with Grid Layout
col1, col2, col3, col4 = st.columns(4)
describe_button = col1.button("Analyze Image 📊")
extract_button = col2.button("Extract Text 📝")
tts_button = col3.button("Play Narration 🎙️")
stop_button = col4.button("Stop Audio ⏹️")

# Button Actions
if uploaded_file:
    if describe_button:
        with st.spinner("Analyzing the scene..."):
            st.success("Scene description will be displayed here.")
            # Add image analysis code or API call
    
    if extract_button:
        with st.spinner("Extracting text..."):
            text = extract_text_from_image(uploaded_file)
            st.info(f"Extracted Text: {text}")
    
    if tts_button:
        with st.spinner("Playing narration..."):
            text = extract_text_from_image(uploaded_file)
            text_to_speech(text)
    
    if stop_button:
        pygame.mixer.music.stop()
