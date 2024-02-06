from pytube import YouTube
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import os
import subprocess
import torchaudio
import whisper
# from transformers import AutoProcessor, SeamlessM4TModel

# processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
# model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")

def convert_video(video):
    # video_clip = VideoFileClip(video)
    # audio_clip = video_clip.audio
    # audio_clip.write_audiofile("audio.mp4", codec='aac')
    # video_clip.close()
    # audio_clip.close()
    
    # temp_video_path = "temp_video.mp4"
    # with open(temp_video_path, 'wb') as f:
    #     f.write(video.read())
    audio = AudioSegment.from_file(video)
    output_audio_path = "audio.wav"
    audio.export(output_audio_path, format="wav")
    # os.remove(temp_video_path)

def process_v(video):
    print(video)
    convert_video(video)
    model = whisper.load_model("base")
    result = model.transcribe("audio.wav")
    print(result['text'])
    return result['text']
