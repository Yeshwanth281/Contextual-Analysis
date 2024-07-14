from flask import Blueprint, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from bs4 import BeautifulSoup
import requests
import re

url_analysis_bp = Blueprint('url_analysis_bp', __name__)

# Load models for URL analysis
sentiment_model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
sentiment_tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
emotion_model = AutoModelForSequenceClassification.from_pretrained('j-hartmann/emotion-english-distilroberta-base')
emotion_tokenizer = AutoTokenizer.from_pretrained('j-hartmann/emotion-english-distilroberta-base')

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            text = ' '.join([para.get_text() for para in paragraphs])
            return text
        else:
            return "Error fetching the URL content."
    except Exception as e:
        return f"Error occurred: {str(e)}"

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
    return emotion_labels[emotion.item()], confidence.item()

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

def analyze_url(url):
    text = extract_text_from_url(url)
    if "Error" in text:
        return {"error": text}
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

@url_analysis_bp.route('/analyze', methods=['POST'])
def analyze_url_route():
    try:
        url = request.form.get('url')
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        analysis_results = analyze_url(url)
        return jsonify(analysis_results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
