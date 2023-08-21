import numpy as np
import pyaudio
import tensorflow as tf
import tensorflow_hub as hub
import csv
import openai
import time
import os
from dotenv import load_dotenv
from flask import Flask, render_template_string

# Load YAMNet class names from CSV
def load_yamnet_classes(filepath='yamnet_class_map.csv'):
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        class_names = [row[2] for row in reader]
        return class_names[1:]

yamnet_classes = load_yamnet_classes()

# OpenAI API setup
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the YAMNet model
yamnet_model = hub.load('./yamnet_1/')

# Constants
SR = 16000
CHUNK_SIZE = SR * 2
CONVERSATION_DURATION = 20  # 20 seconds
last_sound_time = None
current_detected_sound = None
current_conversation = ""

app = Flask(__name__)

@app.route('/')
def index():
    global current_detected_sound, current_conversation
    sound = current_detected_sound if current_detected_sound else 'None'
    conversation = current_conversation if current_conversation else 'No conversation yet.'
    return render_template_string("""
        <html>
        <head>
            <meta http-equiv="refresh" content="5">
        </head>
        <body>
            <h1>Detected Sound: {{ sound }}</h1>
            <h2>AI Agents Conversation:</h2>
            <p>AI Agent 1: I just heard {{ sound }}. Have you ever encountered that?</p>
            <p>AI Agent 2: {{ conversation }}</p>
        </body>
        </html>
    """, sound=sound, conversation=conversation)

def get_ai_conversation_about_sound(sound):
    prompt = f"AI Agent 1: I just heard {sound}. Have you ever encountered that?\nAI Agent 2: "
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0].text.strip()

def audio_processing():
    global last_sound_time, current_detected_sound, current_conversation
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=SR, input=True, frames_per_buffer=CHUNK_SIZE)

    try:
        while True:
            audio_chunk = np.frombuffer(stream.read(CHUNK_SIZE, exception_on_overflow=False), dtype=np.int16)
            audio_chunk = audio_chunk / 32768.0
            scores, embeddings, _ = yamnet_model(np.reshape(audio_chunk, [-1]))
            prediction = np.mean(scores, axis=0)
            top_class = np.argmax(prediction)
            label = yamnet_classes[top_class]

            # This condition checks if either the label has changed OR the conversation duration is exceeded
            if label != current_detected_sound or (last_sound_time and (time.time() - last_sound_time) > CONVERSATION_DURATION):
                current_detected_sound = label
                current_conversation = get_ai_conversation_about_sound(label)
                last_sound_time = time.time()  # Update the last sound time.

    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    # Start the audio processing in a separate thread
    import threading
    thread = threading.Thread(target=audio_processing)
    thread.start()

    # Start the Flask app
    app.run(debug=True, port=5050)
