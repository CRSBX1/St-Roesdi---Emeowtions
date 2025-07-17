import json
import numpy as np
from datetime import datetime
import joblib
import pickle
from flask import Flask, jsonify, request, send_from_directory, render_template
from flask_cors import CORS #Enables communication between react frontend and this backend
import os #for directory stuff
import uuid #to add unique names to added files
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import google.generativeai as genai
import subprocess
import tempfile

yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
app = Flask("Emeowtions_analyzer")
CORS(app)
Model_path = 'cat_emotion_model.pkl'

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")# <<< IMPORTANT: Replace with your actual key
genai.configure(api_key="AIzaSyAsza2Ba5lIlkZiA6Fc3BrIukQmh8rOYPU")
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

class CatEmotionAnalyzer:
    def __init__(self):
        self.scaler = None
        self.label_encoder = None
        self.mlp = None
        self.emotion_labels = []
        
    def load_model(self, filepath):
        """Load a pre-trained model"""
        model_data = joblib.load(filepath)
        self.mlp = model_data['mlp']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.emotion_labels = model_data['emotion_labels']
        print(f"Model loaded from {filepath}")
        print(f"Emotion labels: {self.emotion_labels}")
    
    def predict_single_sample(self, sample, metadata=None):
        """Predict single sample with JSON output"""
        if self.mlp is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Ensure sample is 2D array
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)
        
        # Scale the sample
        sample_scaled = self.scaler.transform(sample)
        
        # Get prediction
        prediction = self.mlp.predict(sample_scaled)[0]
        probabilities = self.mlp.predict_proba(sample_scaled)[0]
        
        # Get emotion details
        primary_emotion = self.label_encoder.inverse_transform([prediction])[0]
        confidence = np.max(probabilities)
        
        # Get secondary emotion
        sorted_indices = np.argsort(probabilities)[::-1]
        secondary_emotion = self.label_encoder.inverse_transform([sorted_indices[1]])[0]
        secondary_confidence = probabilities[sorted_indices[1]]
        
        # Build JSON output
        result = {
            "timestamp": datetime.now().isoformat(),
            "emotions": {
                "primary": primary_emotion,
                "secondary": secondary_emotion,
                "confidence": float(confidence),
                "secondary_confidence": float(secondary_confidence),
                "intensity": float(confidence),
                "all_probabilities": {
                    emotion: float(prob) 
                    for emotion, prob in zip(self.emotion_labels, probabilities)
                }
            },
            "analysis_details": {
                "model_certainty": "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low",
                "top_emotions": self._get_top_emotions(probabilities, top_n=3),
                "confidence_distribution": {
                    "mean": float(np.mean(probabilities)),
                    "std": float(np.std(probabilities)),
                    "entropy": float(-np.sum(probabilities * np.log(probabilities + 1e-10)))
                }
            }
        }
        
        # Add metadata if provided
        if metadata:
            result.update(metadata)
        
        return result
    
    def _get_top_emotions(self, probabilities, top_n=3):
        """Get top N emotions with their probabilities"""
        sorted_indices = np.argsort(probabilities)[::-1]
        top_emotions = []
        
        for i in range(min(top_n, len(sorted_indices))):
            idx = sorted_indices[i]
            emotion = self.label_encoder.inverse_transform([idx])[0]
            prob = float(probabilities[idx])
            top_emotions.append({
                "emotion": emotion,
                "probability": prob,
                "rank": i + 1
            })
        
        return top_emotions
    
    def predict_with_json_output(self, X_test, additional_metadata=None):
        """Generate predictions in JSON format for multiple samples"""
        if self.mlp is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Get predictions and probabilities
        predictions = self.mlp.predict(X_test)
        probabilities = self.mlp.predict_proba(X_test)
        
        results = []
        
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            # Get primary emotion
            primary_emotion = self.label_encoder.inverse_transform([pred])[0]
            confidence = np.max(probs)
            
            # Get secondary emotion
            sorted_indices = np.argsort(probs)[::-1]
            secondary_emotion = self.label_encoder.inverse_transform([sorted_indices[1]])[0]
            secondary_confidence = probs[sorted_indices[1]]
            
            # Create probability distribution
            emotion_probabilities = {
                emotion: float(prob) 
                for emotion, prob in zip(self.emotion_labels, probs)
            }
            
            # Build JSON structure
            result = {
                "timestamp": datetime.now().isoformat(),
                "sample_id": i,
                "emotions": {
                    "primary": primary_emotion,
                    "secondary": secondary_emotion,
                    "confidence": float(confidence),
                    "secondary_confidence": float(secondary_confidence),
                    "intensity": float(confidence),
                    "all_probabilities": emotion_probabilities
                },
                "prediction_details": {
                    "model_certainty": "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low",
                    "top_emotions": self._get_top_emotions(probs, top_n=3),
                    "prediction_strength": float(confidence - secondary_confidence)
                }
            }
            
            # Add metadata if provided
            if additional_metadata and i < len(additional_metadata):
                result.update(additional_metadata[i])
            
            results.append(result)
        
        return results

def save_predictions_to_json(predictions, filename):
    """Save predictions to JSON file"""
    with open(filename, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to {filename}")

def load_predictions_from_json(filename):
    """Load predictions from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)
    
def load_audio(file_path, target_sr=16000):
    waveform, sr = librosa.load(file_path, sr=target_sr)
    return waveform

def extract_yamnet_embeddings(waveform):
    # Run through YAMNet to get embeddings
    scores, embeddings, spectrogram = yamnet_model(waveform)
    # Average embeddings across time
    return tf.reduce_mean(embeddings, axis=0).numpy()

analyzer = CatEmotionAnalyzer()

with app.app_context():
    analyzer.load_model(Model_path)

# --- Define the API endpoint for emotion analysis ---
@app.route('/api/analyze-cat-emotion', methods=['POST'])
def analyze_emotion():
    # Check if a file was uploaded in the request
    if 'audioOrVideoFile' not in request.files:
        return jsonify({"error": "No 'audioOrVideoFile' part in the request"}), 400

    file = request.files['audioOrVideoFile']
    file_type = request.form.get('fileType', 'unknown') # Get 'audio' or 'video'

    # If the user submits an empty form without selecting a file
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # --- 1. Save the uploaded file temporarily ---
        # Create a unique filename to avoid conflicts if multiple users upload
        unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        temp_filepath = os.path.join('/tmp', unique_filename) # Use a suitable temp directory

        # Create /tmp directory if it doesn't exist (important for deployment)
        os.makedirs(os.path.dirname(temp_filepath), exist_ok=True)

        try:
            file.save(temp_filepath)
            print(f"File saved temporarily to: {temp_filepath}")
            if file_type == 'audio':
                audio = load_audio(temp_filepath)
                embedding = extract_yamnet_embeddings(audio)
                embedding = np.array(embedding)
            elif file_type == 'video':
                print("Extracting audio from video...")
                # Use tempfile for intermediate audio path
                audio_temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                audio_path = audio_temp.name
                audio_temp.close()

                # Run ffmpeg to extract audio as WAV
                try:
                    subprocess.run([
                        'ffmpeg', '-y', '-i', temp_filepath,
                        '-vn', '-acodec', 'pcm_s16le',
                        '-ar', '16000', '-ac', '1',
                        audio_path
                    ], check=True)

                    audio = load_audio(audio_path)
                    embedding = extract_yamnet_embeddings(audio)
                    embedding = np.array(embedding)

                    os.remove(audio_path)  # Clean up
                except subprocess.CalledProcessError as e:
                    return jsonify({"error": f"Failed to extract audio: {str(e)}"}), 500
            else:
                return jsonify({"error": "Unsupported file type specified."}), 400

            # --- 3. Make a prediction using your loaded .pkl model ---
            print("Making prediction with the model...")
            final_analysis_result = analyzer.predict_single_sample(embedding)
            print(f"Printing analysis results:\n{final_analysis_result}")
            
            # --- Generate initial Chatbot response using Gemini API ---
            print("Generating initial chatbot response using Gemini API...")
            pet_name = request.form.get('Pet.Name', 'your cat') # Assuming petName is sent from frontend
            pet_breed = request.form.get('Pet.Breed', 'a cat') # Assuming petBreed is sent from frontend
            pet_age = request.form.get('Pet.Age', 'age not known')
            pet_description = request.form.get('Pet.description?', 'no description given')



            # Craft the prompt for Gemini, including the analysis result
            chat_prompt = (
                f"You are an AI assistant specialized in cat behavior and emotions. "
                f"A user has just uploaded an audio/video for their cat, {pet_name} ({pet_breed}), age {pet_age}. {pet_name} is described as{pet_description}"
                f"The emotion analysis model has determined the primary emotion is '{final_analysis_result['emotions']['primary']}' "
                f"with an accuracy estimation of {final_analysis_result['emotions']['confidence']:.2f}%. "
                f"The recommendation is: '{final_analysis_result['analysis_details']['top_emotions'][0]['emotion']}'. " # Using primary emotion's recommendation
                f"Based on this analysis, provide a helpful and welcoming initial message to the user. "
                f"Encourage them to ask further questions about {pet_name}'s behavior or general cat care. "
                f"Keep it concise and friendly."
            )

            try:
                gemini_response = gemini_model.generate_content(chat_prompt)
                initial_chatbot_message = gemini_response.text
            except Exception as gemini_err:
                print(f"Error calling Gemini API: {gemini_err}")
                initial_chatbot_message = (
                    f"Hello! I've analyzed {pet_name}'s emotions. "
                    f"It seems {pet_name} is primarily feeling {final_analysis_result['emotions']['primary'].lower()} "
                    f"with a confidence of {final_analysis_result['emotions']['confidence']:.2f}%. "
                    f"My recommendation is to: '{final_analysis_result['analysis_details']['top_emotions'][0]['emotion']}'. " # Fallback if Gemini fails
                    f"I'm ready for your questions!"
                )

            # --- Combine and return the results ---
            # Send both the analysis result and the initial chatbot message back
            return jsonify({
                "analysisResult": final_analysis_result,
                "initialChatbotMessage": initial_chatbot_message
            }), 200

        except Exception as e:
            return jsonify({"error": f"Internal server error during analysis: {str(e)}"}), 500
        finally:
            # --- 5. Clean up the temporary file ---
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)

    return jsonify({"error": "Something went wrong with the file upload"}), 500

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
            return jsonify({"error": "Message is empty"}), 400

        prompt = (
            f"You are an AI assistant specialized in cat emotions and behavior. "
            f"A user is caring for a cat named {pet_name}, a {pet_breed}, currently showing {pet_emotion.lower()} emotion. "
            f"The user asked: \"{user_message}\". "
            f"{confidence}% is the confidence of the emotion analysis, use this information to your full advantage."
            f"If {confidence} level is medium (<70%) or low (<40%), ask the user for further inquiries regarding their cat situation."
            f"If user provides you with their current pet situation, use it to provide meaningful insights, advices, or helpful information."
            f"Please respond with helpful, warm advice or information."
        )

        gemini_response = gemini_model.generate_content(prompt)
        reply = gemini_response.text

        return jsonify({"reply": reply})
    
    except Exception as e:
        print(f"Gemini chat error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/')
def index():
    return render_template('index.html')  # Serves the frontend entry page

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory('static', path)  # Serve static assets

if __name__ == '__main__':
    # When running locally, Flask runs on http://127.0.0.1:5000 by default
    # debug=True allows for automatic reloading on code changes and provides debug info
    app.run(debug=True, port=5000, use_reloader=False)

