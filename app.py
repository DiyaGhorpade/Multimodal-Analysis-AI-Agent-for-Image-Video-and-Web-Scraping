import streamlit as st
import os 
import tempfile
import subprocess
import cv2
import speech_recognition as sr
import google.generativeai as genai
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel("models/gemini-2.0-flash")
search_tool=DuckDuckGoSearchRun()

st.set_page_config(page_title="AI Multimodal Analyzer",layout="wide")
st.title("Multimodal Analysis AI Agent for Image,Video and Web Scraping")
st.markdown("Analyze media or perform web searches using Gemini AI")

def save_temp_file(uploaded_file):
    if uploaded_file:
        suffix=f".{uploaded_file.name.split('.')[-1]}"
        with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            return tmp.name
    return None

def extract_thumbnail(video_path):
    cap=cv2.VideoCapture(video_path)
    ret,frame=cap.read()
    cap.release()
    if ret:
        thumb_path=video_path+"_thumb.jpg"
        cv2.imwrite(thumb_path,frame)
        return thumb_path
    return None

def transcribe_audio(video_path):
    audio_path = video_path + "_audio.wav"
    subprocess.run(
        ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_path, '-y'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    recognizer = sr.Recognizer()
    transcription = ""

    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
            transcription = recognizer.recognize_google(audio)
    except Exception as e:
        transcription = f"Transcription failed: {e}"
    if os.path.exists(audio_path):
        os.remove(audio_path)

    return transcription

def analyze_content(prompt,image_path=None,transcription=None):
    content=prompt
    if transcription:
        content+=f"\n\nTranscription:\n{transcription}"
    
    try:
        if image_path:
            with open(image_path,"rb") as img:
              response=model.generate_content([content,{"mime_type":"image/jpeg","data":img.read()}])
        else:
            response=model.generate_content(content)
        return response.text
    except Exception as e:
        return f"Analysis failed:{e}"
    
def perform_web_search(query):
    results=search_tool.run(query)
    prompt=f"Web search results for '{query}':\n{results}\n\nProvide a comprehensive analysis."
    return analyze_content(prompt)

analysis_type=st.sidebar.radio("Choose analysis type:",["Image","Video","Web Search"])

if analysis_type=="Image":
   image_file=st.file_uploader("Upload an image:",type=["jpg","png","jpeg"])
   if st.button("Analyze Image") and image_file:
       image_path=save_temp_file(image_file)
       st.image(image_path,width=400)
       prompt="Provide a detailed analysis of this image"
       result=analyze_content(prompt,image_path=image_path)
       st.markdown(result)
       os.remove(image_path)

elif analysis_type == "Video":
    video_file = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov"]
    )

    if st.button("Analyze Video") and video_file:
        video_path = save_temp_file(video_file)
        thumbnail_path = extract_thumbnail(video_path)
        transcription = transcribe_audio(video_path)

        col1, col2 = st.columns(2)

        with col1:
            st.image(thumbnail_path, caption="Thumbnail", width=350)

        with col2:
            st.markdown("### Transcription")
            st.write(transcription)

        prompt = "Analyze this video based on the thumbnail and transcription."
        result = analyze_content(
            prompt,
            image_path=thumbnail_path,
            transcription=transcription
        )

        st.markdown(result)

        os.remove(video_path)
        os.remove(thumbnail_path)

elif analysis_type=="Web Search":
    query=st.text_input("Enter search query")
    if st.button("Search & Analyze") and query:
        result=perform_web_search(query)
        st.markdown(result)




