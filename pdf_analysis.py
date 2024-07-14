from flask import Blueprint, request, jsonify
import PyPDF2
from transformers import pipeline

pdf_analysis_bp = Blueprint('pdf_analysis_bp', __name__)

def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()
        print("Extracted text:", text[:200])  # Debugging
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")  # Debugging
        return f"Error occurred: {str(e)}"

def analyze_sentiment(text):
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    result = sentiment_pipeline(text[:512])
    print("Sentiment analysis result:", result)  # Debugging
    return result[0]['label']

def analyze_emotion(text):
    emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    result = emotion_pipeline(text[:512])
    print("Emotion analysis result:", result)  # Debugging
    emotion_labels = ["love", "joy", "surprise", "anger", "sadness", "fear", "disgust"]
    filtered_emotions = [e for e in result if e['label'] in emotion_labels]
    if filtered_emotions:
        return filtered_emotions[0]['label']
    return "Neutral"

def classify_age_appropriateness(text):
    age_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    age_labels = ["G (General)", "R (Restricted) (18+)"]
    result = age_pipeline(text[:512], candidate_labels=age_labels)
    print("Age appropriateness result:", result)  # Debugging
    return result['labels'][0]

def categorize_iab(text):
    iab_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    iab_categories = [
        "Arts & Entertainment", "Automotive", "Business", "Careers",
        "Education", "Family & Parenting", "Health & Fitness",
        "Food & Drink", "Hobbies & Interests", "Home & Garden",
        "Law, Government, & Politics", "News", "Personal Finance",
        "Pets", "Science", "Sports", "Technology & Computing",
        "Travel", "Real Estate", "Shopping", "Society"
    ]
    result = iab_pipeline(text[:512], candidate_labels=iab_categories)
    print("IAB categorization result:", result)  # Debugging
    return result['labels'][0]

@pdf_analysis_bp.route('/analyze', methods=['POST'])
def analyze_pdf_route():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        file_path = f"uploads/{file.filename}"
        file.save(file_path)
        text = extract_text_from_pdf(file_path)
        if text.startswith("Error occurred"):
            return jsonify({'error': text}), 400
        sentiment = analyze_sentiment(text)
        emotion = analyze_emotion(text)
        age_appropriateness = classify_age_appropriateness(text)
        iab_category = categorize_iab(text)
        results = {
            "sentiment": sentiment,
            "emotion": emotion,
            "age_appropriateness": age_appropriateness,
            "iab_category": iab_category
        }
        return jsonify(results)
    except Exception as e:
        print(f"Error in PDF analysis route: {str(e)}")  # Debugging
        return jsonify({'error': str(e)}), 500
