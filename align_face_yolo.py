import os
import cv2
import torch
from PIL import Image
from face_detector import YoloDetector
import numpy as np

# yolo-face
model = YoloDetector(target_size=720, device='cpu', min_face=90)

path = os.getcwd() + '\database'
des_path = os.getcwd() + '\\train'

# Create the output folder if it doesn't exist
if not os.path.exists(des_path):
    os.makedirs(des_path)

# Process each subfolder in the image database (each subfolder represents a label)
for label in os.listdir(path):
    label_folder = os.path.join(path, label)

    # Create a folder for the current label in the output folder
    output_label_folder = os.path.join(des_path, label)
    os.makedirs(output_label_folder, exist_ok=True)
    
    for label_file in os.listdir(label_folder):
        image_path = os.path.join(label_folder, label_file)
        
        # Load the input image using OpenCV
        orgimg = np.array(Image.open(image_path))
        orgimg = cv2.cvtColor(orgimg, cv2.COLOR_BGR2RGB)
        bboxes,points = model.predict(orgimg)
        bboxes = np.array(bboxes)
        try: 
            x_min, y_min, x_max, y_max = bboxes[0][0][:4]
        except:
            continue
        face_image = orgimg[y_min:y_max, x_min:x_max]
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image_pil = Image.fromarray(face_image)
        face_image_pil.save(os.path.join(output_label_folder, label_file))
        print("Saved image: ", os.path.join(output_label_folder, label_file))  

print("Done!")
 
