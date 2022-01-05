import cv2
import numpy as np
import yaml


from utils.general import check_yaml, xywhn2xyxy
from utils.my_general import crop_box, visualize_img

class TrafficLightDetector:
    def __init__(self, config_path, imgsz):
        config = check_yaml(config_path)
        with open(config, errors='ignore') as f:
            hyp = yaml.safe_load(f)
            traffic_light_box = xywhn2xyxy(np.array(hyp['traffic_light_box']), w = imgsz[1], h = imgsz[0])
            self.red_light_box = traffic_light_box[0]
            self.yellow_light_box = traffic_light_box[1]
            self.green_light_box = traffic_light_box[2]

            print(self.red_light_box)

            self.red_color = (0, 0, 255)
            self.yellow_color = (0, 255, 255)
            self.green_color = (0, 255, 0)
            
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
    

    def visualize(self, image, result):
        visualize_image = image.copy()

        if result == 1:       # red
            box = self.red_light_box
            color = self.red_color
            visualize_image = cv2.rectangle(image, 
                                            (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 
                                            color , 2)
        elif result == 2:    # yellow
            box = self.yellow_light_box
            color = self.yellow_color
            visualize_image = cv2.rectangle(image, 
                                            (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 
                                            color , 2)
        elif result == 0:    # green
            box = self.green_light_box
            color = self.green_color
            visualize_image = cv2.rectangle(image, 
                                            (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 
                                            color , 2)


        return visualize_image

