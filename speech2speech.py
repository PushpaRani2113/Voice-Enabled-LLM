import streamlit as st
import speech_recognition as sr
import torch
import pyttsx3
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)
from transformers import BartForConditionalGeneration, BartTokenizer, pipeline
# Load your fine-tuned model on CPU
model = BartForConditionalGeneration.from_pretrained(r"D:\lab - 6sem\dllab\project\bart_model")
tokenizer = BartTokenizer.from_pretrained(r"D:\lab - 6sem\dllab\project\bart_model")
custom_pipeline = pipeline(task="summarization", model=model, tokenizer=tokenizer)
gen_kwargs = {'length_penalty': 0.5, 'num_beams': 5, "max_length": 200}
def vats():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Say something in English and pause; no need to press any key.")
        audio = recognizer.listen(source)
        st.write("Got it!")
    try:
        text = recognizer.recognize_google(audio)
        st.write("You spoke:",text)
    except sr.UnknownValueError:
        st.write("Sorry, I couldn't understand what you said.")
result = custom_pipeline(text, **gen_kwargs)
summary_output = result[0]["summary_text"]
st.write("Summary : ")
st.write(summary_output)
print(summary_output)
engine.say(summary_output)
engine.runAndWait()
def main():
    st.title("Spidex")
    button_clicked = st.button("Start Recording")
    if button_clicked:
        vats()
if __name__ == "__main__":
    main()