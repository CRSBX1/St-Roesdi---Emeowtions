import json
import numpy as np
from datetime import datetime
import joblib
import os
import uuid
import tempfile
import subprocess
import logging
import gc
from flask import Flask, jsonify, request, send_from_directory, render_template
from flask_cors import CORS
import google.generativeai as genai

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'cat_emotion_model.pkl')
app = Flask("Emeowtions_analyzer", template_folder=os.path.join(base_dir, "templates"), static_folder=os.path.join(base_dir, "static"))
CORS(app)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

class CatEmotionAnalyzer:
    def __init__(self):
        self.scaler = None
        self.label_encoder = None
        self.mlp = None
        self.emotion_labels = []

    def load_model(self, filepath):
        model_data = joblib.load(filepath)
        self.mlp = model_data['mlp']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.emotion_labels = model_data['emotion_labels']
        print(f"Model loaded from {filepath}")

    def predict_single_sample(self, sample, metadata=None):
        if self.mlp is None:
            raise ValueError("Model not loaded.")

        sample = sample.reshape(1, -1) if sample.ndim == 1 else sample
        sample_scaled = self.scaler.transform(sample)
        prediction = self.mlp.predict(sample_scaled)[0]
        probabilities = self.mlp.predict_proba(sample_scaled)[0]
        
        primary = self.label_encoder.inverse_transform([prediction])[0]
        confidence = float(np.max(probabilities))
        sorted_idx = np.argsort(probabilities)[::-1]
        secondary = self.label_encoder.inverse_transform([sorted_idx[1]])[0]
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "emotions": {
                "primary": primary,
                "secondary": secondary,
                "confidence": confidence,
                "secondary_confidence": float(probabilities[sorted_idx[1]]),
                "intensity": confidence,
                "all_probabilities": {
                    emotion: float(prob)
                    for emotion, prob in zip(self.emotion_labels, probabilities)
                }
            },
            "analysis_details": {
                "model_certainty": "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low",
                "top_emotions": self._get_top_emotions(probabilities, 3),
                "confidence_distribution": {
                    "mean": float(np.mean(probabilities)),
                    "std": float(np.std(probabilities)),
                    "entropy": float(-np.sum(probabilities * np.log(probabilities + 1e-10)))
                }
            }
        }
        if metadata:
            result.update(metadata)
        return result

    def _get_top_emotions(self, probs, top_n):
        sorted_idx = np.argsort(probs)[::-1]
        return [
            {
                "emotion": self.label_encoder.inverse_transform([idx])[0],
                "probability": float(probs[idx]),
                "rank": i + 1
            } for i, idx in enumerate(sorted_idx[:top_n])
        ]

analyzer = CatEmotionAnalyzer()
with app.app_context():
    analyzer.load_model(model_path)

def load_audio(file_path, target_sr=16000):
    import librosa
    waveform, sr = librosa.load(file_path, sr=target_sr)
    return waveform

def extract_yamnet_embeddings(waveform):
    import tensorflow as tf
    import tensorflow_hub as hub
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    scores, embeddings, _ = yamnet_model(waveform)
    return tf.reduce_mean(embeddings, axis=0).numpy()

# API Routes
@app.route('/api/analyze-cat-emotion', methods=['POST'])
def analyze_emotion():
    if 'audioOrVideoFile' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['audioOrVideoFile']
    file_type = request.form.get('fileType', 'unknown')
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    temp_path = os.path.join('/tmp', unique_filename)
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)

    try:
        file.save(temp_path)
        logging.info(f"Saved file to {temp_path}")

        if file_type == 'audio':
            waveform = load_audio(temp_path)
            embedding = extract_yamnet_embeddings(waveform)
        elif file_type == 'video':
            audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            subprocess.run([
                'ffmpeg', '-y', '-i', temp_path,
                '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path
            ], check=True)
            waveform = load_audio(audio_path)
            embedding = extract_yamnet_embeddings(waveform)
            os.remove(audio_path)
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        result = analyzer.predict_single_sample(np.array(embedding))

        pet_name = request.form.get('Pet.Name', 'your cat')
        pet_breed = request.form.get('Pet.Breed', 'a cat')
        pet_age = request.form.get('Pet.Age', 'unknown')
        pet_desc = request.form.get('Pet.description?', '')

        prompt = (
            f"You are an AI assistant for cats. {pet_name}, a {pet_breed}, age {pet_age}, is described as {pet_desc}. "
            f"The model detected the emotion '{result['emotions']['primary']}' with {result['emotions']['confidence']*100:.2f}%."
        )

        try:
            gemini_response = gemini_model.generate_content(prompt)
            chatbot_message = gemini_response.text
        except Exception as e:
            logging.error(f"Gemini API failed: {e}")
            chatbot_message = f"{pet_name} seems to feel {result['emotions']['primary'].lower()} ({result['emotions']['confidence']*100:.2f}%)."

        return jsonify({
            "analysisResult": result,
            "initialChatbotMessage": chatbot_message
        }), 200

    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        del file, waveform, embedding
        gc.collect()

@app.route('/api/chat', methods=['POST'])
def chat_with_gemini():
    try:
        data = request.get_json()
        user_message = data.get("message", "")
        pet_name = data.get("petName", "your cat")
        pet_breed = data.get("petBreed", "a cat")
        pet_emotion = data.get("emotion", "neutral")
        confidence = data.get("confidence", "100%")

        if not user_message.strip():
            return jsonify({"error": "Empty message"}), 400

        prompt = (
            f"User owns {pet_name} the {pet_breed}, who is {pet_emotion} with confidence {confidence}. "
            f"User asked: '{user_message}'. Respond with friendly, helpful advice."
        )

        gemini_response = gemini_model.generate_content(prompt)
        return jsonify({"reply": gemini_response.text})
    except Exception as e:
        logging.error(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
