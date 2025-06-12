# import speech_recognition as sr
# import requests
# #from gtts import gTTS
# import os
# import pyttsx3
# from my_api import api_key
# # Function to capture voice input
# def capture_speech():
#     r = sr.Recognizer()
#     with sr.Microphone() as source:
#         print("Listening...")
#         audio = r.listen(source)
#         try:
#             text = r.recognize_google(audio)
#             print(f"You said: {text}")
#             return text
#         except sr.UnknownValueError:
#             print("Sorry, I could not understand the audio.")
#             return None
#         except sr.RequestError as e:
#             print(f"Could not request results; {e}")
#             return None

# # Function to send input text to Gemini API
# def send_to_gemini_api(text):
#     api_url = "https://console.cloud.google.com/apis/credentials?project=utility-lock-413913"  # Replace with actual endpoint
#     headers = {
#         "Authorization": api_key,  # Replace with your API key
#         "Content-Type": "application/json"
#     }
#     data = {
#         "prompt": text
#     }
#     response = requests.post(api_url, json=data, headers=headers)
#     if response.status_code == 200:
#         return response.json()["response"]  # Adjust based on actual API response format
#     else:
#         print(f"Error: {response.status_code}")
#         return None

# # Function to convert text to speech and play it
# def speak_text(response_text):
#     tts = gTTS(text=response_text, lang='en')
#     tts.save("response.mp3")
#     os.system("start response.mp3")  # For Windows; use "afplay" for macOS, "xdg-open" for Linux

# # Optional: Offline TTS
# def speak_text_offline(response_text):
#     engine = pyttsx3.init()
#     engine.say(response_text)
#     engine.runAndWait()

# # Main conversation loop with Sakhi
# def sakhi_conversation():
#     print("Hello! I am Sakhi, your AI assistant. How can I help you today?")
#     speak_text("Hello! I am Sakhi, your AI assistant. How can I help you today?")

#     while True:
#         # Step 1: Capture the user's speech
#         user_input = capture_speech()
#         if user_input is None:
#             continue
        
#         # Step 2: Send the transcribed speech to Gemini API
#         response_text = send_to_gemini_api(user_input)
#         if response_text is None:
#             speak_text("Sorry, I couldn't get a response.")
#             continue
        
#         # Include Sakhi in the response text
#         sakhi_response = f"Sakhi says: {response_text}"
#         print(sakhi_response)
        
#         # Step 3: Convert the AI's text response to speech
#         speak_text(sakhi_response)  # Or use speak_text_offline(sakhi_response)
        
#         # Optional: Exit the loop if the user says "bye"
#         if "bye" in user_input.lower():
#             speak_text("Goodbye! It was nice talking to you.")
#             print("Goodbye! It was nice talking to you.")
#             break

# # if __name__ == "__main__":
# #     sakhi_conversation()

import speech_recognition as sr
import pyttsx3
import requests
from textblob import TextBlob  # Using TextBlob for NLP
from my_api import api_key
import pyaudio

import nltk
nltk.download('punkt')       # For tokenization
nltk.download('punkt_tab')   # Download the punkt_tab resource
nltk.download('averaged_perceptron_tagger')  # For part-of-speech tagging

print(nltk.data.find('tokenizers/punkt'))
print(nltk.data.find('tokenizers/punkt_tab'))

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

# Gemini API Setup
GEMINI_API_URL = "https://console.cloud.google.com/apis/credentials?project=utility-lock-413913"
GEMINI_API_KEY = api_key  # Replace with your actual Gemini API key

# Function to convert speech to text
def capture_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand that.")
            return None
        except sr.RequestError as e:
            print(f"Request error: {e}")
            return None

# Function to perform basic NLP using TextBlob
def process_nlp(user_input):
    blob = TextBlob(user_input)
    print(f"NLP Processed: {blob.tags}")  # For visualization
    entities = {word for (word, pos) in blob.tags if pos == 'NNP'}  # Extract proper nouns as entities
    return entities

# Function to call Gemini AI for generating responses
def call_gemini_api(user_input):
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "input": user_input
    }
    
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            print(f"Gemini Response: {result['response']}")
            return result['response']
        else:
            print(f"Error with Gemini API: {response.status_code}")
            return "Sorry, I had trouble processing that."
    except Exception as e:
        print(f"Error: {e}")
        return "I am having technical difficulties."

# Function to convert text to speech
def speak_text(response_text):
    print(f"Sakhi: {response_text}")
    tts_engine.say(response_text)
    tts_engine.runAndWait()

# Function to combine both NLP and Gemini to respond intelligently
def sakhi_response(user_input):
    entities = process_nlp(user_input)

    # Check for specific intents or entities (like names)
    if entities:
        name = ', '.join(entities)  # Join entities to form a response
        return f"Hi {name}, how are you today?"

    # Fallback to Gemini API for complex conversations
    gemini_response = call_gemini_api(user_input)
    return gemini_response

# Main conversation loop
def sakhi_conversation():
    speak_text("Hello! I am Sakhi, your AI assistant. How can I help you today?")
    
    while True:
        user_input = capture_speech()
        if user_input is None:
            continue
        
        # Process and get the response (NLP + Gemini)
        response_text = sakhi_response(user_input)
        
        # Speak the response
        speak_text(response_text)

        # Exit if user says goodbye
        if "bye" in user_input.lower():
            speak_text("Goodbye! It was nice talking to you.")
            break

if __name__ == "__main__":
    sakhi_conversation()
