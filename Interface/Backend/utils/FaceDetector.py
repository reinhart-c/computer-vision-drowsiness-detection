import dlib
import torch
from ultralytics import YOLO



class FaceDetector:
    def __init__(self):
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_face = YOLO("models/yolov11n-face.pt").to(self.device_str)
        self.predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

    def extract_face(self, frame):
        pred = self.model_face(frame, device=self.device_str, verbose=False)[0]
        if len(pred) != 0:
            for i, box in enumerate(pred.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                rect = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)
                face_landmark = self.predictor(frame, rect)
                break

            img = frame[y1:y2+1, x1:x2+1]
            return img, face_landmark
        else:
            return -1, -1