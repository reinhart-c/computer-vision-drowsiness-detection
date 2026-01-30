import torch
import numpy as np
import joblib

from models.GRUClassifier import GRUClassifier


class Classifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = joblib.load("weights/scaler.pkl")
        self.yolo_scaler = joblib.load("weights/yolo_scaler.pkl")

    def scale_data(self, features):
        features = np.array(features)
        feat_part = self.scaler.transform(features[:, :21])
        yolo_part = self.yolo_scaler.transform(features[:, 21:])
        features_scaled = np.concatenate([feat_part, yolo_part], axis=1)
        return features_scaled
    
    def predict(self, features):
        features = self.scale_data(features)
        xs = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        lengths = torch.tensor([features.shape[0]]).to(self.device)
        model = GRUClassifier(input_dim=features.shape[1], hidden_dim=128, num_classes=3, num_layers=2).to(self.device)
        checkpoint = torch.load("weights/best_model.pth", map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])

        model.eval()
        with torch.no_grad():
            outputs = model(xs, lengths)
            probs = torch.softmax(outputs, dim=-1)
            pred = torch.argmax(probs, dim=-1)

        return pred