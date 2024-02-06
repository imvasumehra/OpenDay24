from flask import Flask, render_template, request, jsonify
from denoising import denoise
import os

app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"message": "No file part"})

    file = request.files['file']

    if file.filename == '':
        return jsonify({"message": "No selected file"})

    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        return jsonify({"message": f"File '{file.filename}' uploaded successfully! Path: {file_path}! SR: {denoise()}"})
    except Exception as e:
        return jsonify({"message": f"Error: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
