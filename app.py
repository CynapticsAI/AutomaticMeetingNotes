import streamlit as st
from process_audio import process_a
from process_youtube import process
from process_video import process_v
from summarizer import summarize
import os

def save_uploaded_file(uploaded_file, save_path):
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return save_path

def main():
    st.title("Automatic Meeting Notes Generatoree")

    input_source = st.radio("Select Input Source:", ("Video File", "YouTube URL", "Audio File"))
    video_file = None
    youtube_url = None
    summary = ""
    if input_source == "Video File" and video_file is not None:
        video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mkv"])
        file_name = video_file.name
        save_path = os.path.join("./", file_name)
        save_uploaded_file(video_file, save_path)
    elif input_source == "Audio File":
        audio_file = st.file_uploader("Upload Audio", type=["mp3", "wav"])
    else:
        youtube_url = st.text_input("Enter YouTube URL")


    target_lang = st.selectbox("Select Target Language:", ["en", "es", "fr", "de"])
    transcribed_text = ""
    if input_source == "Audio File" and audio_file is not None:
        transcribed_text = process_a(audio_file)
        summary = summarize(transcribed_text, target_lang)
    if input_source == "Video File" and video_file is not None:
        transcribed_text = process_v(f'{save_path}')
        summary = summarize(transcribed_text, target_lang)
    if input_source == "YouTube URL" and youtube_url is not None:
        transcribed_text = process(youtube_url)
        summary = summarize(transcribed_text, target_lang)

    st.subheader("Transcribed Text:")
    # st.text_area(label='Transcribed Text', value=transcribed_text, disabled=True)
    st.markdown(transcribed_text, unsafe_allow_html=True)
    st.subheader("Meeting Summary")
    # st.text_area(label='Summary', value=summary, disabled=True)
    st.markdown(summary, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
