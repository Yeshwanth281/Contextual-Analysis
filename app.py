from flask import Flask, render_template, request, jsonify
from image_analysis import image_analysis_bp, analyze_image_route
from url_analysis import url_analysis_bp, analyze_url_route
from text_analysis import text_analysis_bp, analyze_text_route

app = Flask(__name__)

# Register blueprints
app.register_blueprint(image_analysis_bp, url_prefix='/image')
app.register_blueprint(url_analysis_bp, url_prefix='/url')
app.register_blueprint(text_analysis_bp, url_prefix='/text')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        file_type = request.form.get('file_type')
        if file_type == 'image':
            return analyze_image_route()
        elif file_type == 'url':
            return analyze_url_route()
        elif file_type == 'text':
            return analyze_text_route()
        else:
            return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)
