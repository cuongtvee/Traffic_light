import yaml
from utils.license_plate_general import load_model, preprocess_image, get_boxes, visualize_img
from utils.general import non_max_suppression
from utils.general import check_yaml
import numpy as np
import torch
from plate import Plate
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.datasets import LoadStreams, LoadImages, letterbox
class Detector:
    def __init__(self, config_path):
        config = check_yaml(config_path)
        with open(config, errors='ignore') as f:
            hyp = yaml.safe_load(f)
            self.model, self.name = load_model(hyp['model_path'])

            imgsz = hyp['imgsz']
            if isinstance(imgsz, int):
                imgsz = (imgsz, imgsz)
            self.imgsz = imgsz
            
            self.conf_thres = hyp['conf_thres']
            self.iou_thres = hyp['iou_thres']
            self.max_det = hyp['max_det']
            self.agnostic_nms = hyp['agnostic_nms']
            self.class_name = hyp['class_name']
            self.color_map = hyp['color_map']
            self.device = hyp['device']
            self.line_thickness = hyp['line_thickness']
            self.line_size = hyp['line_size']
            self.hide_confidence = hyp['line_size']
            self.names = self.load_classes('./data/coco.names')

    def detect(self, frame):
        # img = torch.from_numpy(img).to(device)
        
        img = letterbox(frame, new_shape=self.imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        # print("img",img)
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)


        pred = self.model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)

        cars = []
        motorbikes = []
        plates=[]
        longplates=[]
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
          # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape)
                # Write results
                det=det.cpu().detach().numpy()
                for *xyxy, conf, cls in det:
                    predicted_class = self.names[int(cls)]
                    x = xyxy[0]
                    y = xyxy[1]
                    w = xyxy[2]-x
                    h = xyxy[3]-y
                    if predicted_class == 'car':
                        cars.append([x, y, w, h, conf,None])
                    elif predicted_class == 'motorbike':
                        motorbikes.append([x, y, w, h, conf,None])
                    elif predicted_class == 'plate':
                        plates.append(Plate([x,y,w,h],conf,0))
                    elif predicted_class == 'longplate':
                        longplates.append(Plate([x,y,w,h],conf,1))
                    else:
                        continue
        for car in cars:
            for plate in plates:
          # print("***************")
                if self.contains(car[:4],plate.tlwh):
                    car[5]=plate
            for plate in longplates:
                if self.contains(car[:4],plate.tlwh):
                    car[5]=plate
        for motorbike in motorbikes:
            for plate in plates:
                if self.contains(motorbike[:4],plate.tlwh):
                    motorbike[5]=plate
            for plate in longplates:
                if self.contains(motorbike[:4],plate.tlwh):
                    motorbike[5]=plate
        dets = [cars, motorbikes]
        return dets
        

    def visualize(self, frame, boxes, using_tracking = False):
        visualized_img = frame.copy()
        for box in boxes:
            visualized_img = visualize_img(img_src = visualized_img, box = box, line_thickness = self.line_thickness, line_size = self.line_size, \
                                            color_map = self.color_map, class_name = self.class_name, \
                                            hide_confidence = self.hide_confidence, using_tracking = using_tracking)

        return visualized_img
    def load_classes(self,path):
        # Loads *.names file at 'path'
        with open(path, 'r') as f:
            names = f.read().split('\n')
        return list(filter(None, names))  # filter removes empty strings (such as last line)
    def contains(self,r1, r2):
    # print(r1,r2)
        return r1[0] < r2[0] < r2[0]+r2[2] < r1[0]+r1[2] and r1[1] < r2[1] < r2[1]+r2[3] < r1[1]+r1[3]









        
        