import os
import cv2
import dlib
import time
import argparse
import numpy as np
from imutils import video

DOWNSAMPLE_RATIO = 4

def reshape_for_polyline(array):
    return np.array(array, np.int32).reshape((-1, 1, 2))

class FaceLandmarkDetector:

    def __init__(self, args):
        self.count = 0
        # Create the face predictor and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(args.face_landmark_shape_file)

    def constructLandmarks(self, landmarks):
        self.closedFaceParts = {
            "jaw" : reshape_for_polyline(landmarks[0:17]),
            "right_eyebrow" : reshape_for_polyline(landmarks[17:22]),
            "left_eyebrow" : reshape_for_polyline(landmarks[22:27]),
            "nose_bridge" : reshape_for_polyline(landmarks[27:31]),
        };
        self.openFaceParts = {
            "lower_nose" : reshape_for_polyline(landmarks[30:35]),
            "right_eye" : reshape_for_polyline(landmarks[36:42]),
            "left_eye" : reshape_for_polyline(landmarks[42:48]),
            "outer_lip" : reshape_for_polyline(landmarks[48:60]),
            "inner_lip" : reshape_for_polyline(landmarks[60:68]),
        };

    def writePolylines(self, target_image):
        color = (255, 255, 255)
        thickness = 3
        for key in self.closedFaceParts :
            cv2.polylines(target_image, [self.closedFaceParts[key]], False, color, thickness)
        for key in self.openFaceParts :
            cv2.polylines(target_image, [self.openFaceParts[key]], True, color, thickness)
        return target_image
    def resizeFrame(self, frame):
        return cv2.resize(frame, None, fx=1 / DOWNSAMPLE_RATIO, fy=1 / DOWNSAMPLE_RATIO)

    def processFrame(self, frame, frame_resize):
        gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)
        black_image = np.zeros(frame.shape, np.uint8)

        # Perform if there is a face detected
        if len(faces) > 0:
            for face in faces:
                detected_landmarks = self.predictor(gray, face).parts()
                landmarks = [[p.x * DOWNSAMPLE_RATIO, p.y * DOWNSAMPLE_RATIO] for p in detected_landmarks]
                self.constructLandmarks(landmarks);
                black_image = self.writePolylines(black_image);
            return black_image,True;
        else:
            print("No face detected")
            return None,False;
