import os

from flask import Flask, render_template, request
from werkzeug import secure_filename

from engine import load_process_audio, predict

app = Flask(__name__)
SAVE_DIR = "storage"

# @app.route('/upload')
# def upload_file():
#     return render_template('upload.html')


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
    pred_str = "Crying baby" if prediction == 0 else "Not crying baby"
    return pred_str


if __name__ == '__main__':
    app.run(debug=True)
