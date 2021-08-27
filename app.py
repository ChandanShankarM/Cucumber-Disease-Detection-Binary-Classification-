from lodingmodel import *
from flask import Flask
from flask import render_template
import os
from flask import request


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename





app = Flask(__name__)
UPLOAD_FOLDER='C:/Users/moshi/webpage/sample_images'
import joblib
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        
        file_path = os.path.join(
            UPLOAD_FOLDER, secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_result(file_path)
        result=preds
        return result
    return None

@app.route('/predict/remedies')
def show():
    r="mohsin nb"
    return r

if __name__ == "__main__":
    app.run(port=3000,debug=True)
    
