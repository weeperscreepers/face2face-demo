import argparse
import cv2
import dlib
import numpy as np
import tensorflow as tf
from imutils import video


from FaceLandmarkDetector import FaceLandmarkDetector

CROP_SIZE = 256

def resize(image):
    """Crop and resize image for pix2pix."""
    height, width, _ = image.shape
    if height != width:
        # crop to correct ratio
        size = min(height, width)
        oh = (height - size) // 2
        ow = (width - size) // 2
        cropped_image = image[oh:(oh + size), ow:(ow + size)]
        image_resize = cv2.resize(cropped_image, (CROP_SIZE, CROP_SIZE))
        return image_resize


def load_graph(frozen_graph_filename):
    """Load a (frozen) Tensorflow model into memory."""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_filename, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph

def main():
    # TensorFlow
    graph = load_graph(args.frozen_model_file)
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    output_tensor = graph.get_tensor_by_name('generate_output/output:0')
    sess = tf.Session(graph=graph)

    # OpenCV
    cap = cv2.VideoCapture(args.video_source)
    fps = video.FPS().start()

    faceProcessor = FaceLandmarkDetector()

    while True:
        _,frame = cap.read()
        frame_resize = faceProcessor.resizeFrame(frame);
        processedFrame = faceProcessor.processFrame(frame_resize)

        # generate prediction
        combined_image = np.concatenate([resize(processedFrame), resize(frame_resize)], axis=1)
        image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR instead of RGB
        generated_image = sess.run(output_tensor, feed_dict={image_tensor: image_rgb})
        image_bgr = cv2.cvtColor(np.squeeze(generated_image), cv2.COLOR_RGB2BGR)
        image_normal = np.concatenate([resize(frame_resize), image_bgr], axis=1)
        image_landmark = np.concatenate([resize(processedFrame), image_bgr], axis=1)

        if args.display_landmark == 0:
            cv2.imshow('frame', image_normal)
        else:
            cv2.imshow('frame', image_landmark)

        fps.update()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    sess.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('--show', dest='display_landmark', type=int, default=0, choices=[0, 1],
                        help='0 shows the normal input and 1 the facial landmark.')
    parser.add_argument('--landmark-model', dest='face_landmark_shape_file', type=str, help='Face landmark model file.')
    parser.add_argument('--tf-model', dest='frozen_model_file', type=str, help='Frozen TensorFlow model file.')

    args = parser.parse_args()

    # Create the face predictor and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.face_landmark_shape_file)

    main()
