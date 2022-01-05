import cv2
import numpy as np
import yaml


from utils.general import check_yaml, xywhn2xyxy
from utils.my_general import crop_box

class TrafficLightDetector:
    def __init__(self, config_path, imgsz):
        config = check_yaml(config_path)
        with open(config, errors='ignore') as f:
            hyp = yaml.safe_load(f)
            self.red_light_box = xywhn2xyxy(np.array(hyp['red_light_box']), w = imgsz.shape[0], h = imgsz.shape[1])
            self.yellow_light_box = xywhn2xyxy(np.array(hyp['yellow_light_box']), w = imgsz.shape[0], h = imgsz.shape[1])
            self.green_light_box = xywhn2xyxy(np.array(hyp['green_light_box']), w = imgsz.shape[0], h = imgsz.shape[1])
    
            
    def detect(self, image):
        def is_turn_on(image):
            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            idx = np.where(hsv_image[:,:, :] > 200)[0]
            if idx.shape[0] > 100:
                return True
            return False

        red_light_image = crop_box(image, self.red_light_box, padding = 0)
        if is_turn_on(red_light_image):
            return 1

        yellow_light_image = crop_box(image, self.yellow_light_box, padding = 0)
        if is_turn_on(yellow_light_image):
            return 2

        green_light_image = crop_box(image, self.green_light_box, padding = 0)
        if is_turn_on(green_light_image):
            return 0

        return -1
    

    def visualize(self, image):
        return 0

