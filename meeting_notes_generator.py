# Whisper model for transcription

!pip install pytube
from pytube import YouTube
link=''
try:
    # object creation using YouTube
    # which was imported in the beginning
    yt = YouTube(link)
except:
    print("Connection Error")
 yt.streams.filter(file_extension='mp4')
stream = yt.streams.get_by_itag(139)
stream.download('',"meeting.mp4")
!pip install git+https://github.com/openai/whisper.git
!pip install jiwer
import whisper

model = whisper.load_model("base")
result = model.transcribe("Workshop.mp4")
print(result['text'])

# Facebook M2M100 Model for summarization
!pip install sentencepiece
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# en_text = "Do not meddle in the affairs of wizards, for they are subtle and quick to anger."
# chinese_text = "不要插手巫師的事務, 因為他們是微妙的, 很快就會發怒."

french_text = "La ville de Paris est réputée pour sa beauté intemporelle, avec ses rues pavées, ses monuments emblématiques tels que la Tour Eiffel et le Louvre, et son ambiance artistique unique. Les cafés pittoresques en bord de Seine invitent les visiteurs à savourer un café tout en observant le monde qui passe. La richesse culturelle de la capitale française se reflète dans ses musées, ses galeries d'art et ses nombreux événements culturels. Paris, la Ville Lumière, continue de captiver le cœur de ceux qui la découvrent, offrant une expérience inoubliable."
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="fr")
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")

encoded_zh = tokenizer(french_text, return_tensors="pt")

generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("en"))
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

#Seamless M4T Model for Multilingual ip/op

import torchaudio
from transformers import AutoProcessor, SeamlessM4TModel
processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")
# Read an audio file and resample to 16kHz:
audio, orig_freq =  torchaudio.load("https://www2.cs.uic.edu/~i101/SoundFiles/preamble10.wav")
audio =  torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16_000) # must be a 16 kHz waveform array
audio_inputs = processor(audios=audio, return_tensors="pt")

# Process some input text as well:
# text_inputs = processor(text = "Hello, my dog is cute", src_lang="eng", return_tensors="pt")
# audio_array_from_text = model.generate(**text_inputs, tgt_lang="rus")[0].cpu().numpy().squeeze()
output_tokens = model.generate(**audio_inputs, tgt_lang="fra", generate_speech=False)
translated_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
print(f"Translated Text: {translated_text_from_audio}")