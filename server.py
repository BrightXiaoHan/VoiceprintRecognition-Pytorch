import argparse
import tempfile
import wave

import librosa
import requests
import soundfile as sf
import uvicorn
from fastapi import FastAPI

from mvector.predict import MVectorPredictor

app = FastAPI()

configs = "./configs/ecapa_tdnn.yml"
use_gpu = False
threshold = 0.45
model_path = "./models/ecapa_tdnn_MelSpectrogram/best_model/"

# 获取识别器
predictor = MVectorPredictor(configs=configs, model_path=model_path, use_gpu=use_gpu)
test_folder = "./dataset/test_data"
test_data = "/home/hanbing/asr/pyannote/data.xlsx"


def varify(audio_file_1, audio_file_2):
    dist = predictor.contrast(audio_file_1, audio_file_2)
    return dist.tolist()


def download(url, audio_file):
    response = requests.get(url, allow_redirects=True)
    open(audio_file, "wb").write(response.content)

    # get sample rate of audio file
    with wave.open(audio_file, "rb") as f:
        rate = f.getframerate()
    if rate != 16000:
        assert rate == 8000
        y, sr = librosa.load(audio_file, sr=8000)
        # resample wav file from 8k to 16k
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sf.write(audio_file, y, 16000)
    return audio_file


@app.post("/yuyispeech/vector/score")
def score(enroll_audio_url, test_audio_url, sample_rate: int = 16000):
    enroll_audio_file = tempfile.NamedTemporaryFile(suffix=".wav")
    download(enroll_audio_url, enroll_audio_file.name)
    test_audio_file = tempfile.NamedTemporaryFile(suffix=".wav")
    download(test_audio_url, test_audio_file.name)

    score = varify(test_audio_file.name, enroll_audio_file.name)

    enroll_audio_file.close()
    test_audio_file.close()
    return {
        "success": True,
        "code": 200,
        "message": {"description": "success"},
        "result": {"score": score},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=80)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
