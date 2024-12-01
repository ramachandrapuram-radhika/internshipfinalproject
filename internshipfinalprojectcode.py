import streamlit as st
from PIL import Image
import pyttsx3
import os
import pytesseract  
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
import tempfile
from gtts import gTTS

# Set Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize Google Generative AI with API Key
GEMINI_API_KEY = 'Google API Key' 
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

llm = GoogleGenerativeAI(model="gemini-1.5-pro", api_key=GEMINI_API_KEY)

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Streamlit Page Configurations
st.set_page_config(page_title="Visually Impaired Assistant", layout="wide", page_icon="🧠")

st.markdown(
    """
    <style>
     .main-title {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        color: #0662f6;
        margin-top: -20px;
     }
    .subtitle {
        font-size: 18px;
        color: #555;
        text-align: center;
        margin-bottom: 20px;
    }
    .feature-header {
        font-size: 24px;
        color: #333;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">Visually Impaired Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Solutions for Assisting Visually Impaired Individuals</div>', unsafe_allow_html=True)

# Sidebar Features
st.sidebar.image(
    r"C:\Users\Hp\Downloads\picture.jpg",
    width=250
)

st.sidebar.title("ℹ️ About")
st.sidebar.markdown(
    """
    ### 🌟 Key Features
    - 🖼️ **Scene Interpretation**  
      Transform images into insightful descriptions, highlighting key objects and actions.
    - 📄 **Text Extraction**  
      Seamlessly detect and extract text from any uploaded image.
    - 🎧 **Listen with Ease**  
      Convert text into voice and experience it through natural audio playback.

    ### 🚀 Why Choose This?
    - 👁️‍🗨️ **Enhancing Vision**: Helps users understand visual content with descriptive insights.  
    - ✍️ **Accessible Reading**: Enables reading text from images or signs without hassle.  
    - 🗣️ **Voice Assistance**: Provides instant audio feedback for hands-free interaction.

    ### 🤖 Powered By
    - 🌐 **Google Gemini API**: Cutting-edge AI for intelligent scene analysis.  
    - 🔍 **Tesseract OCR**: Accurate text detection and recognition.  
    - 🎙️ **Audio Tech**: Converts text into lifelike speech for smooth listening.  
    """
)

st.sidebar.text_area(
    "📜 Instructions",
    "Upload an image to start. Select features like Scene Description, Text Extraction, or Text-to-Speech."
)

# Functions for functionality
def extract_text_from_image(image):
    """Extracts text from the given image using OCR."""
    return pytesseract.image_to_string(image)

def text_to_speech(text):
    """Converts the given text to speech and plays it."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        tts = gTTS(text=text, lang="en")
        tts.save(temp_audio.name)
        return temp_audio.name

def generate_scene_description(input_prompt, image_data):
    """Generates a scene description using Google Generative AI."""
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content([input_prompt, image_data[0]])
    return response.text

def input_image_setup(uploaded_file):
    """Prepares the uploaded image for processing."""
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data,
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded.")

# Upload Image Section
st.markdown("<h3 class='feature-header'>Upload an Image</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drag and drop or browse an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Feature Selection Section
st.markdown("<h3 class='feature-header'>Choose a Feature</h3>", unsafe_allow_html=True)

with st.container():
    st.write("Select a feature to interact with the uploaded image:")
    
    # Feature selection checkboxes
    describe_scene = st.checkbox("Describe the Scene", key="describe")
    extract_text = st.checkbox("Extract Text", key="extract")
    convert_to_speech = st.checkbox("Convert Text to Speech", key="tts")

# Input Prompt for Scene Understanding
input_prompt = """
You are an AI assistant helping visually impaired individuals by describing the scene in the image. Provide:
1. List of items detected in the image with their purpose.
2. Overall description of the image.
3. Suggestions for actions or precautions for the visually impaired.
"""

# Process user interactions
if uploaded_file:
    image_data = input_image_setup(uploaded_file)

    if describe_scene:
        with st.spinner("Generating scene description..."):
            response = generate_scene_description(input_prompt, image_data)
            st.markdown("<h3 class='feature-header'>🔍 Scene Description</h3>", unsafe_allow_html=True)
            st.write(response)

    if extract_text:
        with st.spinner("Extracting text from the image..."):
            text = extract_text_from_image(image)
            st.markdown("<h3 class='feature-header'>📝 Extracted Text</h3>", unsafe_allow_html=True)
            st.text_area("Extracted Text", text, height=150)

    if convert_to_speech:
        with st.spinner("Converting text to speech..."):
            text = extract_text_from_image(image)
            if text.strip():
                audio_file_path = text_to_speech(text)
                st.success("Text converted to speech successfully!")
                st.audio(audio_file_path, format="audio/mp3", start_time=0)
            else:
                st.warning("No text found to convert.")
