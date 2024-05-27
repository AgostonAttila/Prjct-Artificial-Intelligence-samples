import streamlit as st
from audio_recorder_streamlit import audio_recorder
import openai
import base64


def setup_openai_client(api_key):    
      return openai.OpenAI(api_key=api_key)


def transcribe_audio(client,audio_path):
     
     with open(audio_path, "rb") as audio_file:
         transcript= client.audio.transcriptions.create(model="whisper-1",file = audio_file)
         return transcript.text

def featch_ai_response(client,input_text):
     messages = [{"role":"user","content":input_text}]
     response = client.chat.completions.create(model="gpt-3.5-turbo-1106",messages=messages)
     return response.choices[0].message.content

def text_to_audio(client,tect,audio_path):
     response = client.audio.speech.create(model="tts-1",voice="echo",input=text)
     response.stream_to_file(audio_path)


def main():

    st.sidebar.title("API KEY")
    api_key = st.sidebar._text_input("Enter your OpenAI API Key",type="password")

    st.title("Aurora Speak")
    st.write("Hi there! Click on the voice recorder.")
    if api_key:
        client = setup_openai_client(api_key)
        recorded_audio=audio_recorder()
        if recorded_audio:
             audio_file = "audio.mp3"
             with open(audio_file,"wb") as f:
                f.write(recorded_audio)
             transcribed_text = transcribe_audio(client, audio_file)
             st.write("Transcribed",transcribed_text)

             ai_response = featch_ai_response(client,transcribed_text)
             response_audio_file = "audio_response.mp3"
             text_to_audio(client,ai_response,response_audio_file)
             st.audio(response_audio_file)
             st.write("AI response", ai_response)


if __name__ == "__main__":
    main()


