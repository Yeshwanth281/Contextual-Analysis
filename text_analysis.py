from flask import Blueprint, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

text_analysis_bp = Blueprint('text_analysis_bp', __name__)

# Load models for text analysis
sentiment_model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
sentiment_tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
emotion_model = AutoModelForSequenceClassification.from_pretrained('j-hartmann/emotion-english-distilroberta-base')
emotion_tokenizer = AutoTokenizer.from_pretrained('j-hartmann/emotion-english-distilroberta-base')

def analyze_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = sentiment_model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    confidence, sentiment = torch.max(probs, dim=1)
    sentiment_labels = ['negative', 'neutral', 'positive']
    return sentiment_labels[sentiment.item()], confidence.item()

def analyze_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = emotion_model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    confidence, emotion = torch.max(probs, dim=1)

    emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    custom_emotion_labels = {
        'anger': 'anger',
        'disgust': 'disgust',
        'fear': 'fear',
        'joy': 'happy',
        'neutral': 'neutral',
        'sadness': 'sadness',
        'surprise': 'surprise'
    }

    model_emotion = emotion_labels[emotion.item()]
    mapped_emotion = custom_emotion_labels.get(model_emotion, 'neutral')

    additional_emotions = {
        'rage': 'anger',
        'romance': 'joy',
        'humor': 'joy',
        'compassion': 'joy',
        'valor': 'joy',
        'wonder': 'surprise',
        'peace': 'joy'
    }

    if mapped_emotion == 'neutral':
        for key, value in additional_emotions.items():
            if key in text.lower():
                mapped_emotion = value
                break

    if 'sad' in text.lower() or 'melancholy' in text.lower():
        mapped_emotion = 'sadness'
    elif 'melodramatic' in text.lower():
        mapped_emotion = 'sadness'

    return mapped_emotion, confidence.item()

def age_appropriateness_rating(text):
    restricted_keywords = ["violence", "adult", "explicit", "drugs", "alcohol"]
    if any(word in text.lower() for word in restricted_keywords):
        return "R (Restricted) (18+)"
    else:
        return "G (General)"

def categorize_iab(text):
    iab_categories = {
        'arts': 'Arts & Entertainment',
        'business': 'Business',
        'education': 'Education',
        'family': 'Family & Parenting',
        'health': 'Health & Fitness',
        'food': 'Food & Drink',
        'news': 'News',
        'science': 'Science',
        'sports': 'Sports',
        'technology': 'Technology',
        'travel': 'Travel',
        'privacy': 'Law & Government'
    }
    text = text.lower()
    for keyword, category in iab_categories.items():
        if re.search(r'\b' + keyword + r'\b', text):
            return category
    return "Other"

def analyze_text(text):
    sentiment, sentiment_confidence = analyze_sentiment(text)
    emotion, emotion_confidence = analyze_emotion(text)
    age_rating = age_appropriateness_rating(text)
    iab_category = categorize_iab(text)
    results = {
        "text_excerpt": text[:200],  # Return only the first 200 characters for brevity
        "sentiment": {"label": sentiment, "confidence": sentiment_confidence},
        "emotion": {"label": emotion, "confidence": emotion_confidence},
        "age_rating": age_rating,
        "iab_category": iab_category
    }
    return results

@text_analysis_bp.route('/analyze', methods=['POST'])
def analyze_text_route():
    try:
        text = request.form.get('text')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        analysis_results = analyze_text(text)
        return jsonify(analysis_results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
