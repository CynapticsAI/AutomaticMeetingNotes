import os
from pydub import AudioSegment
import wave
import torchaudio
import os
import subprocess
import torchaudio
import whisper
from moviepy.editor import VideoFileClip
# from transformers import AutoProcessor, SeamlessM4TModel

# processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
# model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")

def process_a(audio):
    # audio.write_audiofile("audio.mp4", codec='aac')
    model = whisper.load_model("base")
    result = model.transcribe(audio)
    print(result['text'])
    return result['text']
