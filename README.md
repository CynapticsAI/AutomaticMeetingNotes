# Automatic Meeting Summarizer

## Overview
"Automatic Meeting Summarizer" model is a machine learning model that is used to summarize the input video file and provide the "minutes of meeting" to the client. The model is trained to provide summary in desired dialect. 

## Approach
The MDPI research paper provided us with an approach for speaker diarization and transcription of the input file. Use of SyncNet was a challenging task as the model takes a specific type of input that didn't match the requirements of our input files. Also the model was super slow and its training time was large. 
On surfing multiple models for trnascription, Whisper was the one that was most accurate and efficient. 
For Summarization and Translation, we used langchain, which is LLM of GPT model. Both summarization and translation can be achieved by this model.

## Dataset
1. https://www.kaggle.com/datasets/wiradkp/mini-speech-diarization
2. https://huggingface.co/datasets/knkarthick/AMI#dataset-creation
3. http://groups.inf.ed.ac.uk/ami/download/

## Model
**Transcription Model:** The model takes video file, audio file or YouTube video as an input and generates its transcript in English using Whisper Model.

**Summarization Model:** The summarization model generates concise and most relevant point from a given video file. Langchain model is used for summarization and M2M 100 model of Facebook for translation. 
The model can take a video or audio file, as well as YouTube video, essentially in english, as an input. The summary is generated in english and then according to user needs the text can be translated from one language to another. 

## Features
Our model is capable of taking audio, video and YouTube video files as input, transcribe them, and generate summary. The model has multilingual output which can be choosen by the user as per requirement. 

## Reference
1. https://github.com/Rehan-Ahmad/MultimodalDiarization
2. https://www.mdpi.com/1424-8220/19/23/5163
3. https://ieeexplore.ieee.org/document/9870007
4. https://platform.openai.com/docs/tutorials/meeting-minutes
5. https://dxiaochuan.medium.com/summarising-your-meeting-with-chatgpt-and-langchain-8eb646cfcdd1

## Contributors
- [@rishitmehrotra](https://github.com/mehrotrarishit)
- [@rainatathed](https://github.com/Raina-310304)
- [@ayushawasthi](https://github.com/ayushawasthi24)
- [@varadgaekwad](https://github.com/Varad-22)
- [@chanakyacherukumalli](https://github.com/Chanakya2456)
