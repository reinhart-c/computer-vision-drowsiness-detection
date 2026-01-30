from utils.FrameExtractor import FrameExtractor
from utils.FaceDetector import FaceDetector
from utils.FeatureExtractor import FeatureExtractor
from utils.Classifier import Classifier



print("start")
i = 1
prediction = -1
frame_extractor = FrameExtractor()
face_detector = FaceDetector()
feat_extractor = FeatureExtractor()
classifier = Classifier()
for frame in frame_extractor.extract_frames():
    face, landmark = face_detector.extract_face(frame)
    # print(type(face), type([]))
    ear, mar, pitch, yaw, roll = feat_extractor.extract_features(face, landmark, frame.shape, i)
    if i >= 50:
        features = feat_extractor.get_window_features()
        prediction = classifier.predict(features)
        i = 25
    print(ear, mar, pitch, yaw, roll, prediction.item())
    # yield frame, ear, mar, pitch, yaw, roll, prediction.item()
    i += 1