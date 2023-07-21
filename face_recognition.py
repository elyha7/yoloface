import os
import cv2
import torch
from PIL import Image
from face_detector import YoloDetector
import numpy as np
import tensorflow as tf
from tensorflow import keras



def load_yolo_face_model():
    model = YoloDetector(target_size=720, device='cpu', min_face=90)
    return model

def face_recognition_yolov5(data_folder):
    # Load yolo_face model
    yolo_model = load_yolo_face_model()
    # load CNN model
    model = tf.keras.models.load_model('model.h5')
    labels = os.listdir(data_folder)

    while True:
        # Open the webcam (you can also use a video file here)
        cap = cv2.VideoCapture(0)

        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                break

            # Perform face detection with YOLOv5
            bboxes, points = yolo_model.predict(frame)
            bboxes = np.array(bboxes)

            # Perform face recognition with OpenCV
            try: 
                x_min, y_min, x_max, y_max = bboxes[0][0][:4]
            except:
                continue
            face_image = frame[y_min:y_max, x_min:x_max]
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_image_pil = Image.fromarray(face_image)
            face_image_pil = face_image_pil.resize((224, 224))
            face_image = np.array(face_image_pil)
            face_image = face_image.astype('float32')
            face_image /= 255
            face_image = np.expand_dims(face_image, axis=0)
            # Predict the label of the image
            prediction = model.predict(face_image)
            # Get the label of the prediction
            print(prediction)
            label = labels[np.argmax(prediction)]
            # Get the confidence of the prediction
            confidence = prediction[0][np.argmax(prediction)]
            # Define the text we want to display on the frame
            text = label + ' (' + str(round(confidence, 2)) + ')'
            # Draw a bounding box around the detected face
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # Display the label and bounding box rectangle on the output frame
            cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)                        

            # Show the frame with bounding boxes and labels
            cv2.imshow('Face Recognition', frame)

            # Exit when 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam and close all windows
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    data_folder = "train"  # Folder containing the labeled training dataset
    face_recognition_yolov5(data_folder)
