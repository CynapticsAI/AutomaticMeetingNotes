from pytube import YouTube
import os
import subprocess
import torchaudio
import whisper
# from transformers import AutoProcessor, SeamlessM4TModel

# processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
# model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")

def download_video(url):
    yt = YouTube(url)
    audio = yt.streams.filter(only_audio=True).first()
    audio.download()
    os.rename(audio.default_filename, "audio.mp4")


def process(url):
    download_video(url)
    model = whisper.load_model("base")
    result = model.transcribe("audio.mp4")
    print(result['text'])
    return result['text']
