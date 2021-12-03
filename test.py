from face_detector import YoloDetector
import numpy as np
from PIL import Image
import cv2

def show_results(img, xywh, landmarks):
    h,w,c = img.shape
    tl = 5 or round(0.2 * (h + w) / 2) + 1  # line/font thickness

    n = 0
    for i in xywh:
        x1 =i[0]
        y1 = i[1]
        x2 = i[2]
        y2 = i[3]
        cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

        clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

        c=0
        for j in landmarks[n]:
            cv2.circle(img, (j[0],j[1]), tl+1, clors[c], -1)
            c=c+1

        tf = max(tl - 1, 1)  # font thickness    
        cv2.putText(img, 'Face {0}'.format(n), (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        n=n+1

    return img
# gpu=-1 means use cpu, gpu=n where n is gpu device id
model = YoloDetector(target_size=720,gpu=-1,min_face=90)
img = cv2.imread('test_image.jpg')
orgimg = np.array(img)
bboxes,points = model.predict(orgimg)

faces = show_results(img,bboxes,points)
win_name = "visualization"
#Named window for fit-to-window
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.imshow(win_name,faces)
cv2.waitKey(0)
cv2.destroyAllWindows()

