import cv2
import os
import torch
import torch.nn as nn
import subprocess
import numpy as np
from ultralytics import YOLO


class FeatureExtractor:
    def __init__(self):
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO("models/yolo11n.pt").to(self.device_str)
        self.model.model = self.model.model.eval().to(self.device_str)

        self.features = []
        self.yolo_features = []

        self.all_features = []


    def calculate_EAR(self, eye_points):
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def calculate_MAR(self, mouth_points):
        A = np.linalg.norm(mouth_points[1] - mouth_points[7])
        B = np.linalg.norm(mouth_points[2] - mouth_points[6])
        C = np.linalg.norm(mouth_points[3] - mouth_points[5])
        D = np.linalg.norm(mouth_points[0] - mouth_points[4])
        mar = (A + B + C) / (2.0 * D)
        return mar

    def calculate_HPE(self, landmarks, frame_shape):
        h, w = frame_shape[:2]

        image_points = np.array([
            landmarks[30],
            landmarks[8],
            landmarks[36],
            landmarks[45],
            landmarks[48],
            landmarks[54]
        ], dtype="double")

        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ])

        # Camera internals
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return np.nan, np.nan, np.nan

        rotation_mat, _ = cv2.Rodrigues(rotation_vector)
        proj_mat = np.hstack((rotation_mat, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_mat)

        pitch, yaw, roll = euler_angles.flatten().astype(float)
        return pitch, yaw, roll
    
    def get_yolo_features(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(self.device_str) / 255.0
        with torch.no_grad():
            # Forward through YOLO backbone only
            backbone_out = self.model.model.model[:10](img)
            pooled = torch.nn.functional.adaptive_avg_pool2d(backbone_out[-1], (1, 1))
            feats = pooled.view(pooled.size(0), -1).squeeze(0).cpu().numpy()
            return feats

    def extract_features(self, face, face_landmark, frame_shape, frame_num):
        if frame_num == 50:
            self.features = self.features[:25]
            self.yolo_features = self.yolo_features[:25]

        if isinstance(face, type(np.array([]))):
            self.yolo_features.append(self.get_yolo_features(face))

            landmarks = np.array([[face_landmark.part(i).x, face_landmark.part(i).y] for i in range(68)])

            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            ear = (self.calculate_EAR(left_eye) + self.calculate_EAR(right_eye)) / 2.0

            mouth = landmarks[60:68]
            mar = self.calculate_MAR(mouth)

            pitch, yaw, roll = self.calculate_HPE(landmarks, frame_shape)
            self.features.append([ear, mar, pitch, yaw, roll])
            return ear, mar, pitch, yaw, roll
        else:
            self.features.append([np.nan]*5)
            self.yolo_features.append([np.nan]*256)
            return -1, -1, -1, -1, -1
    

    def calculate_perclos(self, ear_feats, threshold=0.2):
        if len(ear_feats) == 0:
            return 0.0
        closed = np.sum(np.array(ear_feats) < threshold)
        return closed / len(ear_feats)


    def calculate_blink_stats(self, ear_feats, threshold=0.2):
        blinks = []
        count = 0
        in_blink = False

        for val in ear_feats:
            if val < threshold:
                count += 1
                in_blink = True
            elif in_blink:
                blinks.append(count)
                count = 0
                in_blink = False

        if in_blink:
            blinks.append(count)

        blink_freq = len(blinks)
        blink_dur = np.mean(blinks) if blinks else 0.0
        return blink_freq, blink_dur


    def calculate_yawn_stats(self, mar_feats, threshold=0.5):
        yawns = []
        count = 0
        in_yawn = False

        for val in mar_feats:
            if val > threshold:
                count += 1
                in_yawn = True
            elif in_yawn:
                yawns.append(count)
                count = 0
                in_yawn = False

        if in_yawn:
            yawns.append(count)

        yawn_freq = len(yawns)
        yawn_dur = np.mean(yawns) if yawns else 0.0
        return yawn_freq, yawn_dur


    def calculate_ear_deltas(self, ear_feats):
        return np.mean(np.abs(np.diff(ear_feats))) if len(ear_feats) > 1 else 0.0

    def calculate_mar_deltas(self, mar_feats):
        return np.mean(np.abs(np.diff(mar_feats))) if len(mar_feats) > 1 else 0.0

    def calculate_pose_deltas(self, pitch_feats, yaw_feats, roll_feats):
        d_pitch = np.mean(np.abs(np.diff(pitch_feats))) if len(pitch_feats) > 1 else 0.0
        d_yaw = np.mean(np.abs(np.diff(yaw_feats))) if len(yaw_feats) > 1 else 0.0
        d_roll = np.mean(np.abs(np.diff(roll_feats))) if len(roll_feats) > 1 else 0.0
        return d_pitch, d_yaw, d_roll

    def interpolate_nan(self, arr):
        n = len(arr)
        idx = np.arange(n)
        valid = ~np.isnan(arr)
        if valid.sum() == 0 or valid.sum() == n:
            return arr
        interp = np.copy(arr)
        interp[np.isnan(interp)] = np.interp(idx[np.isnan(interp)], idx[valid], arr[valid])
        return interp

    def fill_missing(self, arr, global_means):
        temp = arr.copy()
        temp = np.array(temp)
        nan = np.isnan(temp)
        temp[nan] = global_means[nan]
        return temp

    def add_feature_deltas(self, window_features):
        if len(window_features) > 2:
            remove_idx = [i for i in range(10, len(window_features[0]))]
            w_features = np.delete(window_features, remove_idx, axis=1)
            # get speed
            deltas = np.diff(w_features, axis=0)
            deltas = np.vstack(([np.zeros(w_features.shape[1])], deltas))
            # get acceleration
            delta2 = np.diff(deltas, axis=0)
            delta2 = np.vstack(([np.zeros(deltas.shape[1])], delta2))
            combined = np.hstack((window_features, deltas, delta2))
        elif len(window_features) == 1:
            combined = np.hstack((window_features, [np.zeros(20)]))
        else:
            remove_idx = [i for i in range(10, len(window_features[0]))]
            w_features = np.delete(window_features, remove_idx, axis=1)
            deltas = np.diff(w_features, axis=0)
            deltas = np.vstack(([np.zeros(w_features.shape[1])], deltas))
            delta2 = np.vstack(([np.zeros(deltas.shape[1])], [np.zeros(deltas.shape[1])]))
            combined = np.hstack((window_features, deltas, delta2))
        return combined
    
    def process_window_features(self, features):
        ear_feats = features[0]
        mar_feats = features[1]
        pitch_feats = features[2]
        yaw_feats = features[3]
        roll_feats = features[4]
        yolo_feats = features[5:]

        valid_ear  = float((~np.isnan(ear_feats)).sum())  / max(1, len(ear_feats))
        valid_mar  = float((~np.isnan(mar_feats)).sum())  / max(1, len(mar_feats))
        valid_hpe = float((~np.isnan(pitch_feats)).sum()) / max(1, len(pitch_feats))
        valid_yolo = float((~np.isnan(yolo_feats[0])).sum()) / max(1, len(yolo_feats[0]))
        confidence = (valid_ear + valid_mar + valid_hpe + valid_yolo) / 4

        ear_interp = self.interpolate_nan(ear_feats)
        mar_interp = self.interpolate_nan(mar_feats)
        pitch_interp = self.interpolate_nan(pitch_feats)
        yaw_interp = self.interpolate_nan(yaw_feats)
        roll_interp = self.interpolate_nan(roll_feats)
        yolo_interp = [self.interpolate_nan(yol) for yol in yolo_feats]


        if np.isnan(ear_interp).all():
            ear_mean, ear_std, blink_freq, blink_dur, perclos, d_ear = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            ear_mean, ear_std = np.mean(ear_interp), np.std(ear_interp)
            blink_freq, blink_dur = self.calculate_blink_stats(ear_interp)
            perclos = self.calculate_perclos(ear_interp)
            d_ear = self.calculate_ear_deltas(ear_interp)

        if np.isnan(mar_interp).all():
            mar_mean, mar_std, yawn_freq, yawn_dur, d_mar = np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            mar_mean, mar_std = np.mean(mar_interp), np.std(mar_interp)
            yawn_freq, yawn_dur = self.calculate_yawn_stats(mar_interp)
            d_mar = self.calculate_mar_deltas(mar_interp)

        if np.isnan(pitch_interp).all() or np.isnan(yaw_interp).all() or np.isnan(roll_interp).all():
            pitch_mean, pitch_std, yaw_mean, yaw_std, roll_mean, roll_std, d_pitch, d_yaw, d_roll = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            pitch_mean, pitch_std = np.mean(pitch_interp), np.std(pitch_interp)
            yaw_mean, yaw_std = np.mean(yaw_interp), np.std(yaw_interp)
            roll_mean, roll_std = np.mean(roll_interp), np.std(roll_interp)
            d_pitch, d_yaw, d_roll = self.calculate_pose_deltas(pitch_interp, yaw_interp, roll_interp)

        yolo_features = []
        for yol in yolo_interp:
            if np.isnan(yol).all():
                yolo_features.append(np.nan)
            else:
                yolo_features.append(np.mean(yol))


        window_features = [
            ear_mean, ear_std, 
            mar_mean, mar_std, 
            pitch_mean, pitch_std, 
            yaw_mean, yaw_std, 
            roll_mean, roll_std, 
            blink_freq, blink_dur, 
            yawn_freq, yawn_dur, 
            perclos, 
            d_ear, d_mar, d_pitch, d_yaw, d_roll, 
            confidence]
        window_features.extend(yolo_features)
        return window_features
    
    def get_window_features(self):
        features = np.array([np.concatenate((np.array(self.features), np.array(self.yolo_features).squeeze()), axis=1)]).squeeze()

        window_features = self.process_window_features(np.transpose(features))

        global_means = np.nanmean(self.all_features, axis=0)
        if np.isnan(window_features).any():
            window_features = self.fill_missing(window_features, global_means)

        self.all_features.append(window_features)
        return self.add_feature_deltas(self.all_features)