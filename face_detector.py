import joblib
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
import copy
import scipy
import pathlib
from math import sqrt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
from models.common import Conv
from models.yolo import Model
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, \
    scale_coords,scale_coords_landmarks,filter_boxes

class YoloDetector:
    def __init__(self, weights_name='yolov5n_state_dict.pt',config_name='yolov5n.yaml', gpu = 0, min_face=100, target_size=None, frontal=False):
            """
            weights_name: name of file with network weights in weights/ folder.
            config_name: name of .yaml config with network configuration from models/ folder.
            gpu : gpu number (int) or -1 or string for cpu.
            min_face : minimal face size in pixels.
            target_size : target size of smaller image axis (choose lower for faster work). e.g. 480, 720, 1080.
                        None for original resolution.
            frontal : if True tries to filter nonfrontal faces by keypoints location.
            """
            self._class_path = pathlib.Path(__file__).parent.absolute()#os.path.dirname(inspect.getfile(self.__class__))
            self.gpu = gpu
            #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            self.target_size = target_size
            self.min_face = min_face
            self.frontal = frontal
            if self.frontal:
                self.anti_profile = joblib.load(os.path.join(self._class_path, 'models/anti_profile/anti_profile_xgb_new.pkl'))
            self.detector = self.init_detector(weights_name,config_name)

    def init_detector(self,weights_name,config_name):
        print(self.gpu)
        if type(self.gpu) == int and self.gpu >= 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu)
            self.device = 'cuda:0'
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self.device = 'cpu'
        model_path = os.path.join(self._class_path,'weights/',weights_name)
        print(model_path)
        config_path = os.path.join(self._class_path,'models/',config_name)
        state_dict = torch.load(model_path)
        detector = Model(cfg=config_path)
        detector.load_state_dict(state_dict)
        detector = detector.to(self.device).float().eval()
        for m in detector.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True  # pytorch 1.7.0 compatibility
            elif type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        return detector
    
    def _preprocess(self,imgs):
        """
            Preprocessing image before passing through the network. Resize and conversion to torch tensor.
        """
        pp_imgs = []
        for img in imgs:
            h0, w0 = img.shape[:2]  # orig hw
            if self.target_size:
                r = self.target_size / min(h0, w0)  # resize image to img_size
                if r < 1:  
                    img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)

            imgsz = check_img_size(max(img.shape[:2]), s=self.detector.stride.max())  # check img_size
            img = letterbox(img, new_shape=imgsz)[0]
            pp_imgs.append(img)
        pp_imgs = np.array(pp_imgs)
        pp_imgs = pp_imgs.transpose(0, 3, 1, 2)
        pp_imgs = torch.from_numpy(pp_imgs).to(self.device)
        pp_imgs = pp_imgs.float()  # uint8 to fp16/32
        pp_imgs /= 255.0  # 0 - 255 to 0.0 - 1.0
        return pp_imgs
        
    def _postprocess(self, imgs, origimgs, pred, conf_thres, iou_thres):
        """
            Postprocessing of raw pytorch model output.
            Returns:
                bboxes: list of arrays with 4 coordinates of bounding boxes with format x1,y1,x2,y2.
                points: list of arrays with coordinates of 5 facial keypoints (eyes, nose, lips corners).
        """
        bboxes = [[] for i in range(len(origimgs))]
        landmarks = [[] for i in range(len(origimgs))]
        
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)
        
        for i in range(len(origimgs)):
            img_shape = origimgs[i].shape
            h,w = img_shape[:2]
            gn = torch.tensor(img_shape)[[1, 0, 1, 0]]  # normalization gain whwh
            gn_lks = torch.tensor(img_shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]  # normalization gain landmarks
            det = pred[i].cpu()
            scaled_bboxes = scale_coords(imgs[i].shape[1:], det[:, :4], img_shape).round()
            scaled_cords = scale_coords_landmarks(imgs[i].shape[1:], det[:, 5:15], img_shape).round()

            for j in range(det.size()[0]):
                box = (det[j, :4].view(1, 4) / gn).view(-1).tolist()
                box = list(map(int,[box[0]*w,box[1]*h,box[2]*w,box[3]*h]))
                if box[3] - box[1] < self.min_face:
                    continue
                lm = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                lm = list(map(int,[i*w if j%2==0 else i*h for j,i in enumerate(lm)]))
                lm = [lm[i:i+2] for i in range(0,len(lm),2)]
                bboxes[i].append(box)
                landmarks[i].append(lm)
        return bboxes, landmarks

    def get_frontal_predict(self, box, points):
        '''
            Make a decision whether face is frontal by keypoints.
            Returns:
                True if face is frontal, False otherwise.
        '''
        cur_points = points.astype('int')
        x1, y1, x2, y2 = box[0:4]
        w = x2-x1
        h = y2-y1
        diag = sqrt(w**2+h**2)
        dist = scipy.spatial.distance.pdist(cur_points)/diag
        predict = self.anti_profile.predict(dist.reshape(1, -1))[0]
        if predict == 0:
            return True
        else:
            return False

    def predict(self, imgs, conf_thres = 0.3, iou_thres = 0.5):
        '''
            Get bbox coordinates and keypoints of faces on original image.
            Params:
                imgs: image or list of images to detect faces on
                conf_thres: confidence threshold for each prediction
                iou_thres: threshold for NMS (filtering of intersecting bboxes)
            Returns:
                bboxes: list of arrays with 4 coordinates of bounding boxes with format x1,y1,x2,y2.
                points: list of arrays with coordinates of 5 facial keypoints (eyes, nose, lips corners).
        '''
        # Pass input images through face detector
        
        if type(imgs) != list:
            images = [imgs]
        else:
            images = imgs
        origimgs = copy.deepcopy(images)
        
        images = self._preprocess(images)
        with torch.inference_mode(): # change this with torch.no_grad() for pytorch <1.8 compatibility
            pred = self.detector(images)[0]
            #pred = non_max_suppression_face(pred, conf_thres, iou_thres)
        bboxes, points = self._postprocess(images, origimgs, pred, conf_thres, iou_thres)

        return bboxes, points

    def __call__(self,*args):
        return self.predict(*args)

if __name__=='__main__':
    a = YoloFace()