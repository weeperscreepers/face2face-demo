import os
import cv2
import dlib
import time
import argparse
import numpy as np
from imutils import video
from FaceLandmarkDetector import FaceLandmarkDetector

def writeFrames(frameNumber, original, landmarks):
    # Display the resulting frame
    cv2.imwrite("original/{}.png".format(frameNumber), original)
    cv2.imwrite("landmarks/{}.png".format(frameNumber), landmarks)

NUM_FRAMES = 400

def main(args):
    os.makedirs('original', exist_ok=True)
    os.makedirs('landmarks', exist_ok=True)

    cap = cv2.VideoCapture(args.filename)
    fps = video.FPS().start()

    faceProcessor = FaceLandmarkDetector(args)
    count = 0

    while cap.isOpened():
        frameStart = time.time()
        _,frame = cap.read()
        frame_resize = faceProcessor.resizeFrame(frame);
        processedFrame,found = faceProcessor.processFrame(frame, frame_resize)
        if (count > args.number):
            break;
        if (found):
            count = count+1;
            writeFrames(count, frame, processedFrame)

        print('[INFO] elapsed time: {:.2f}'.format(time.time() - frameStart))
        # if count > NUM_FRAMES:
        #     break;
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', dest='filename', type=str, help='Name of the video file.')
    parser.add_argument('--num', dest='number', type=int, help='Number of train data to be created.')
    parser.add_argument('--landmark-model', dest='face_landmark_shape_file', type=str, help='Face landmark model file.')
    args = parser.parse_args()

    main(args)
