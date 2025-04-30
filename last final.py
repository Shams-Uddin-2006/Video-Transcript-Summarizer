import streamlit as st
import os
import wave
import json
from vosk import Model, KaldiRecognizer, SetLogLevel
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pytube import YouTube
import google.generativeai as genai

# Silence Vosk logging
SetLogLevel(-1)

# Constants
SAMPLE_RATE = 16000
CHUNK_SIZE = 8000
VOSK_MODEL_PATH = r"C:\Users\ghazi\Desktop\projects\rtrp final\vosk-model-small-en-us-0.15"
TEMP_VIDEO = "temp_video.mp4"
TEMP_AUDIO_MP3 = "temp_audio.mp3"
TEMP_AUDIO_WAV = "temp_audio.wav"

# Gemini API Key - use environment variable or secret manager
API_KEY = "AIzaSyCj5Ra-inxtbdBep4dkDbxM1X1jumW3u3M"
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Function: Download video from YouTube
def download_video(url):
    yt = YouTube(url)
    video_stream = yt.streams.filter(file_extension='mp4', only_video=False).first()
    video_stream.download(filename=TEMP_VIDEO)
    return TEMP_VIDEO

# Function: Extract audio
def extract_audio(video_path):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(TEMP_AUDIO_MP3)
    clip.close()

# Function: Convert to WAV for Vosk
def convert_to_wav(mp3_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
    audio.export(TEMP_AUDIO_WAV, format="wav")
    return TEMP_AUDIO_WAV

# Function: Transcribe
def transcribe(wav_path):
    if not os.path.exists(VOSK_MODEL_PATH):
        st.error(f"Vosk model not found in {VOSK_MODEL_PATH}. Download it from https://alphacephei.com/vosk/models")
        return ""

    wf = wave.open(wav_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
        st.error("WAV must be mono PCM.")
        return ""

    model = Model(VOSK_MODEL_PATH)
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    results = []
    while True:
        data = wf.readframes(CHUNK_SIZE)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            results.append(result.get("text", ""))

    final_result = json.loads(rec.FinalResult())
    results.append(final_result.get("text", ""))
    return " ".join(results)

# Function: Summarize
def summarize(text):
    prompt = (
        "The following is a raw, unstructured transcript of a motivational speech. "
        "Please summarize the main ideas clearly and concisely:\n\n" + text
    )
    response = gemini_model.generate_content(prompt)
    return response.text

# Streamlit App
st.title("🎥 Video Transcript Summarizer")

source = st.radio("Choose input type:", ["Upload video", "YouTube link"])

video_path = None

if source == "Upload video":
    uploaded_file = st.file_uploader("Upload MP4 video", type=["mp4"])
    if uploaded_file:
        with open(TEMP_VIDEO, "wb") as f:
            f.write(uploaded_file.read())
        video_path = TEMP_VIDEO

else:
    url = st.text_input("Enter YouTube URL")
    if url and st.button("Download Video"):
        with st.spinner("Downloading video..."):
            try:
                video_path = download_video(url)
                st.success("Video downloaded.")
            except Exception as e:
                st.error(f"Error downloading video: {str(e)}")

# Run processing
if video_path and st.button("Transcribe and Summarize"):
    try:
        with st.spinner("Extracting audio..."):
            extract_audio(video_path)

        with st.spinner("Converting to WAV..."):
            convert_to_wav(TEMP_AUDIO_MP3)

        with st.spinner("Transcribing..."):
            transcript = transcribe(TEMP_AUDIO_WAV)
            """st.subheader("📝 Transcript")
            st.text_area("Transcript", transcript, height=300)"""

        if transcript.strip():
            with st.spinner("Summarizing with Gemini..."):
                summary = summarize(transcript)
                st.subheader("📌 Summary")
                st.text_area("Summary", summary, height=200)
        else:
            st.warning("No transcript generated.")
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
