import os

from flask import Flask, render_template, request
from werkzeug import secure_filename

from engine import load_process_audio, predict, NUM2TEXT

app = Flask(__name__)
SAVE_DIR = "storage"


@app.route('/upload', methods=['POST'])
def upload_file():
    f = request.files['file']
    # f.save(secure_filename(f.filename))
    print(f.filename)
    # return 'file uploaded successfully'
    filename = os.path.join(SAVE_DIR, secure_filename(f.filename))
    f.save(filename)
    return secure_filename(f.filename)


@app.route('/predict', methods=['POST'])
def inference():
    f = request.files['file']
    filename = os.path.join(SAVE_DIR, secure_filename(f.filename))
    f.save(filename)
    feature = load_process_audio(filename)
    prediction = predict(feature)
    pred_str = NUM2TEXT[prediction]
    print(pred_str)
    return pred_str


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
