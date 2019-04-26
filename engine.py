import pickle

import librosa
import numpy as np
from scipy.stats import skew
import time

label = ['crying_baby', 'silence', 'noise', 'baby_laugh']
num = list(range(len(label)))

lab2num = dict(zip(label, num))
num2lab = dict(zip(num, label))

MODEL_PATH = "rf.pickle"
SCALER_PATH = "scaler.pickle"
AUDIO_PATH = "Crowd.mp3"


def extract_feature(data: np.ndarray) -> np.ndarray:
    ft1 = librosa.feature.mfcc(data, n_mfcc=30)[..., :210]
    ft2 = librosa.feature.zero_crossing_rate(data)[0][..., :210]
    ft3 = librosa.feature.spectral_rolloff(data)[0][..., :210]
    ft4 = librosa.feature.spectral_centroid(data)[0][..., :210]
    ft5 = librosa.feature.spectral_contrast(data)[0][..., :210]
    ft6 = librosa.feature.spectral_bandwidth(data)[0][..., :210]
    ft1_trunc = np.hstack((np.mean(ft1, axis=1), np.std(ft1, axis=1), skew(ft1, axis=1),
                           np.max(ft1, axis=1), np.median(ft1, axis=1), np.min(ft1, axis=1)))
    ft2_trunc = np.hstack((np.mean(ft2), np.std(ft2), skew(ft2), np.max(ft2), np.median(ft2),
                           np.min(ft2)))
    ft3_trunc = np.hstack((np.mean(ft3), np.std(ft3), skew(ft3), np.max(ft3), np.median(ft3),
                           np.min(ft3)))
    ft4_trunc = np.hstack((np.mean(ft4), np.std(ft4), skew(ft4), np.max(ft4), np.median(ft4),
                           np.min(ft4)))
    ft5_trunc = np.hstack((np.mean(ft5), np.std(ft5), skew(ft5), np.max(ft5), np.median(ft5),
                           np.min(ft5)))
    ft6_trunc = np.hstack((np.mean(ft6), np.std(ft6), skew(ft6), np.max(ft6), np.median(ft6),
                           np.max(ft6)))
    return np.hstack((ft1_trunc, ft2_trunc, ft3_trunc, ft4_trunc, ft5_trunc, ft6_trunc))


def load_process_audio(filename: str) -> list:
    print('[Load and Processing]')
    extractions = []
    for offset in range(5):
        data, _ = librosa.load(filename, offset=offset, duration=5.0)
        feature = extract_feature(data)
        extractions.append(feature)
    extractions = np.array(extractions)
    return extractions


def predict(feature: np.ndarray) -> int:
    print('[Predicting]')
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    X = scaler.transform(feature)
    prediction = model.predict(X)

    if sum(prediction) > len(prediction) / 2.0:
        return 1  # other
    else:
        return 0  # crying baby


if __name__ == "__main__":
    start = time.time()
    feature = load_process_audio(AUDIO_PATH)
    prediction = predict(feature)
    finish = time.time()
    elapsed = finish - start
    pred_str = "Crying baby" if prediction == 0 else "Not crying baby"
    print(pred_str)
    print('Time elapsed :{:.3f} secs'.format(elapsed))
