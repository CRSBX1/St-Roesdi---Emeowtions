
# Example usage script
import numpy as np
from cat_emotion_analyzer import CatEmotionAnalyzer

# Load model
analyzer = CatEmotionAnalyzer()
analyzer.load_model('cat_emotion_model.pkl')

# Example prediction (replace with your actual features)
sample_features = np.random.random(1024)  # Replace with real features
metadata = {
    "cat_id": "whiskers_123",
    "audio_file": "meow.wav",
    "duration": 2.5,
    "time_of_day": "morning"
}

prediction = analyzer.predict_single_sample(sample_features, metadata)
print(prediction)
