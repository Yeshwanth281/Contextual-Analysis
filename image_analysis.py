from flask import Blueprint, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image

image_analysis_bp = Blueprint('image_analysis_bp', __name__)

# Load models
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

def determine_age_appropriateness(sentiment, textual_description):
    if sentiment == "POSITIVE":
        return "G (General)"
    elif sentiment == "NEGATIVE":
        if "violence" in textual_description.lower() or "explicit content" in textual_description.lower():
            return "R (Restricted) (18+)"
        else:
            return "PG-13 (Parental Guidance) (13+)"
    else:
        return "G (General)"

def analyze_image(image_path):
    image = Image.open(image_path)
    inputs = caption_processor(images=image, return_tensors="pt")
    outputs = caption_model.generate(**inputs, max_new_tokens=20)
    caption = caption_processor.decode(outputs[0], skip_special_tokens=True)
    sentiment_result = sentiment_analyzer(caption)[0]
    sentiment = sentiment_result['label']
    emotions = emotion_analyzer(caption)
    age_rating = determine_age_appropriateness(sentiment, caption)
    results = {
        "caption": caption,
        "sentiment": sentiment_result,
        "emotions": emotions[0],
        "age_rating": age_rating
    }
    return results

@image_analysis_bp.route('/analyze', methods=['POST'])
def analyze_image_route():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        file_path = f"uploads/{file.filename}"
        file.save(file_path)
        analysis_results = analyze_image(file_path)
        return jsonify(analysis_results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
