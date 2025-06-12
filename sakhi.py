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
