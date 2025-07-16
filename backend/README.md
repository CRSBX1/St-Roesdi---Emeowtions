
# Cat Emotion Analyzer Model Usage Instructions

## Loading the Model:
```python
from cat_emotion_analyzer import CatEmotionAnalyzer
analyzer = CatEmotionAnalyzer()
analyzer.load_model('cat_emotion_model.pkl')
```

## Making Predictions:
```python
# Single prediction
prediction = analyzer.predict_single_sample(features, metadata)

# Batch prediction
predictions = analyzer.predict_with_json_output(feature_matrix, metadata_list)
```

## JSON Output Format:
The model outputs structured JSON with emotions, confidence scores, and metadata.
Check model_metadata.json for emotion labels and feature requirements.
