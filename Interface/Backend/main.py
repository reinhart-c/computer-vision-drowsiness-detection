from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import base64
import torch
import cv2
import numpy as np

from utils.FrameExtractor import FrameExtractor
from utils.FaceDetector import FaceDetector
from utils.FeatureExtractor import FeatureExtractor
from utils.Classifier import Classifier

app = FastAPI()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    for frame, ear, mar, pitch, yaw, roll, prediction in detect_drowsiness():
        _, buffer = cv2.imencode(".jpg", frame)
        # if -1 means features not detected/prediction not made (yet) 
        payload = {
                "frame": base64.b64encode(buffer).decode("utf-8"),
                "features": [ear, mar, pitch, yaw, roll],
                "prediction": prediction,
            }

        await websocket.send_json(payload)


def detect_drowsiness():
    i = 1
    prediction = -1
    frame_extractor = FrameExtractor()
    face_detector = FaceDetector()
    feat_extractor = FeatureExtractor()
    classifier = Classifier()
    for frame in frame_extractor.extract_frames():
        face, landmark = face_detector.extract_face(frame)
        ear, mar, pitch, yaw, roll = feat_extractor.extract_features(face, landmark, frame.shape, i)
        if i >= 50:
            features = feat_extractor.get_window_features()
            prediction = classifier.predict(features)
            i = 25
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.item()
        yield frame, ear, mar, pitch, yaw, roll, prediction
        i += 1