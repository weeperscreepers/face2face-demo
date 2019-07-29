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
        # Create the face predictor and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(args.face_landmark_shape_file)

    def constructLandmarks(self, landmarks):
        return {
            "closed": {
                "jaw" : reshape_for_polyline(landmarks[0:17]),
                "right_eyebrow" : reshape_for_polyline(landmarks[17:22]),
                "left_eyebrow" : reshape_for_polyline(landmarks[22:27]),
                "nose_bridge" : reshape_for_polyline(landmarks[27:31]),
            },
            "open": {
                "lower_nose" : reshape_for_polyline(landmarks[30:35]),
                "right_eye" : reshape_for_polyline(landmarks[36:42]),
                "left_eye" : reshape_for_polyline(landmarks[42:48]),
                "outer_lip" : reshape_for_polyline(landmarks[48:60]),
                "inner_lip" : reshape_for_polyline(landmarks[60:68]),
            }
        }


    def writePolylines(self, faceLandmarks, target_image):
        color = (255, 255, 255)
        thickness = 1
        cv2.polylines(target_image, [faceLandmarks['closed'][key] for key in faceLandmarks['closed']], False, color, thickness)
        cv2.polylines(target_image, [faceLandmarks['open'][key] for key in faceLandmarks['open']], True, color, thickness)
        return target_image

    def resizeFrame(self, frame):
        return cv2.resize(frame, None, fx=1 / DOWNSAMPLE_RATIO, fy=1 / DOWNSAMPLE_RATIO)

    def processFrame(self, frame, frame_resize, num_faces=-1):
        gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)
        black_image = np.zeros(frame.shape, np.uint8)

        if(num_faces > 0):
            faces = faces[:num_faces]

        # Perform if there is a face detected
        if len(faces) > 0:
            for face in faces:
                detected_landmarks = self.predictor(gray, face).parts()
                landmarks = [[p.x * DOWNSAMPLE_RATIO, p.y * DOWNSAMPLE_RATIO] for p in detected_landmarks]
                black_image = self.writePolylines(
                    self.constructLandmarks(landmarks),
                    black_image
                );
            return black_image,True;
        else:
            print("No face detected")
            return None,False;
