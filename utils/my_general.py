import torch
import numpy as np
import cv2
import yaml
import os

from models.experimental import attempt_load
from utils.general import  scale_coords, check_yaml
from utils.augmentations import letterbox
from utils.datasets import LoadImages


def get_specific_classes(boxes, cls):
    return np.array([box for box in boxes if int(box[5]) in cls])
    

def load_model(path, train = False):
    model = attempt_load(path, map_location='cuda')  # load FP32 model
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if train:
        model.train()
    else:
        model.eval()
    return model, names

def get_boxes(pred = None, pred_size = None, src_size = None, conf_thres = 0.5):
    boxes = []
    for det in pred:  
        det[:, :4] = scale_coords(pred_size, det[:, :4], src_size).round()
        det = det[:, :6].cpu().detach().numpy()
        for box in det:
            if float(box[4]) > conf_thres:    
                boxes.append(box)
    return np.array(boxes)   

def drop_cls(boxes):          
    """
        Argument: [np.array[x1, y1, x2, y2, confidence, class],..]
        Convert box [x1, y1, x2, y2, confidence, class] to [x1, y1, x2, y2, confidence]
    """
    if boxes.shape[0] == 0:
        return np.empty((0, 5))
    else:
        return boxes[:,: 5]

def preprocess_image(original_image, size = (1280,1280), device = 'cuda'):
    image = letterbox(original_image, size, stride= 8, auto = False)[0]
    image = image.transpose((2, 0, 1))[::-1]
    image = np.ascontiguousarray(image)

    image = torch.from_numpy(image).to(device)
    image = image.float()  
    image = image / 255.0 
    if image.ndimension() == 3:
        image = image.unsqueeze(0)
    return image

def visualize_img(img_src = None, box = None, line_thickness = 3, line_size = 1, color_map = None, class_name = None, hide_confidence = False, using_tracking = True):
    thickness = line_thickness
    line_size = line_size
    font = cv2.FONT_HERSHEY_SIMPLEX

    if using_tracking:
        color = (255, 0, 0)
        display_string = "ID: " + str(box[4])

        visualize_image = cv2.putText(img_src, display_string, 
                                        (int((box[0] + box[2])/2), int((box[1] + box[3])/2)), 
                                        font, line_size, color, thickness)

        visualize_image = cv2.rectangle(visualize_image, 
                                        (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 
                                        color , thickness)

    else:
        color = color_map[int(box[5])]
        if hide_confidence:
            display_string = str(class_name[int(box[5])]) 
        else:
            display_string = "{} [{:.2f}]".format(str(class_name[int(box[5])]), float(box[4]))

        visualize_image = cv2.putText(img_src, display_string, 
                                        (int((box[0] + box[2])/2), int((box[1] + box[3])/2)), 
                                        font, line_size, color, thickness)

        visualize_image = cv2.rectangle(visualize_image, 
                                        (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 
                                        color , thickness)
    return visualize_image

def get_loader(config_path):
    config = check_yaml(config_path)
    with open(config, errors='ignore') as f:
        hyp = yaml.safe_load(f)

        source = hyp['source']
        imgsz = hyp['imgsz']
        stride = hyp['stride']
        auto = hyp['auto']

    video_name = source.split('\\')[-1].split('.')[0]
            
    return LoadImages(source, img_size= imgsz, stride= stride, auto = auto), video_name

def crop_boxes(image, boxes, cls, padding = 5):
    images = []

    for box in boxes:
        if int(box[5]) == cls:
            x1 = int(box[1] - padding) if int(box[1] - padding) > 0 else 0
            x2 = int(box[3] + padding) if int(box[3] + padding) < image.shape[0] else image.shape[0]
            y1 = int(box[0] - padding) if int(box[0] - padding) > 0 else 0
            y2 = int(box[2] + padding) if int(box[2] + padding) < image.shape[1] else image.shape[1]
            cropped_image = image[x1 : x2, y1 : y2, :]
            images.append(cropped_image)

    return images

def crop_box(image, box, padding = 5):
    x1 = int(box[1] - padding) if int(box[1] - padding) > 0 else 0
    x2 = int(box[3] + padding) if int(box[3] + padding) < image.shape[0] else image.shape[0]
    y1 = int(box[0] - padding) if int(box[0] - padding) > 0 else 0
    y2 = int(box[2] + padding) if int(box[2] + padding) < image.shape[1] else image.shape[1]

    return image[x1 : x2, y1 : y2, :]

def make_dir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)

def convert_boxes_to_ratio(boxes, imgsz):
    new_boxes = []
    for box in boxes:
        x_center = (box[0] + box[2])/2
        y_center = (box[1] + box[3])/2
        height = box[3] - box[1]
    
        x_center /= imgsz[1]
        y_center /= imgsz[0]
        height /= imgsz[0]

        new_box = np.array([x_center, y_center, box[4], box[5], height])
        new_boxes.append(new_box)

    return new_boxes

def ioa_batch(bb_vehicle, bb_plate):
    if bb_plate.shape[0] == 0 or bb_vehicle.shape[0] == 0:
        return False, None

    bb_plate = np.expand_dims(bb_plate, 0)
    bb_vehicle = np.expand_dims(bb_vehicle, 1)

    xx1 = np.maximum(bb_vehicle[..., 0], bb_plate[..., 0])
    yy1 = np.maximum(bb_vehicle[..., 1], bb_plate[..., 1])
    xx2 = np.minimum(bb_vehicle[..., 2], bb_plate[..., 2])
    yy2 = np.minimum(bb_vehicle[..., 3], bb_plate[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_plate[..., 2] - bb_plate[..., 0]) * (bb_plate[..., 3] - bb_plate[..., 1]))                         
    return True, (o) 