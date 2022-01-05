import cv2
import os
import time
import argparse
import numpy as np

from detector.detector import Detector
from utils.my_general import get_specific_classes, get_loader, make_dir

from utils.vehicle_plate import get_vehicle_license_plate_pair

from traffic_light_detector.traffic_light_detector import TrafficLightDetector
from utils.sort import Sort


def run(vehicle_config, character_config, traffic_light_config, save_dir, visualize, save_img): 
    vehicle_detector = Detector(vehicle_config)
    character_detector = Detector(character_config)

    #traffic_light_detector = TrafficLightDetector(traffic_light_config)

    license_plate_tracker = Sort(vehicle_config)

    dataset, video_name = get_loader(vehicle_config)                      ## cant not load camera
    print("Load video: ", video_name)

    if save_img:
        save_dir = os.path.join(save_dir, video_name)
        make_dir(save_dir)

    original_imgsz =  dataset.get_imgsz()

    for _, image, _, count in dataset:
        start_time = time.time()
        cv2.imwrite("test.png", image)
        vehicle_boxes = vehicle_detector.detect(image) ## get all classes
        
        car_boxes = get_specific_classes(vehicle_boxes, cls = [0, 1, 4, 5, 6, 7])
        license_plate_boxes = get_specific_classes(vehicle_boxes, cls = [2, 3])

        vehicle_license_plate_in_pairs = get_vehicle_license_plate_pair(car_boxes, license_plate_boxes)    ## pair ([vehicle, plate], ..., [vehicle, plate])

        # red_light_image, yellow_light_image, green_light_image = get_traffic_light_image(image)
        # traffic_light = [is_turn_on(red_light_image), is_turn_on(yellow_light_image), is_turn_on(green_light_image)]    ## true is turn on, false is turn off

        # print(traffic_light)
        # non_cls_boxes = drop_cls(boxes)
        # tracking_boxes = license_plate_tracker.update(non_cls_boxes)
            
        if visualize:
            for pair in vehicle_license_plate_in_pairs:
                car_box = [pair[0]]
                license_plate_box = [pair[1]]
                image = vehicle_detector.visualize(image, car_box, using_tracking = False)
                image = vehicle_detector.visualize(image, license_plate_box, using_tracking = False)    

            # image = vehicle_detector.visualize(image, car_boxes, using_tracking = False)
            # image = vehicle_detector.visualize(image, license_plate_boxes, using_tracking = False)

            result_img = cv2.resize(image, (1280, 720))
            cv2.imshow("Result", result_img)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break


        print("FPS: ", 1/(time.time() - start_time))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vehicle_config', type=str, default= r'config\vehicle_plate_config.yaml', help ='path to config file')
    parser.add_argument('--character_config', type=str, default= r'config\character_config.yaml', help ='path to config file')
    parser.add_argument('--traffic_light_config', type=str, default= r'config\traffic_light_config.yaml', help ='path to config file')
    parser.add_argument('--save_dir', type=str, default= r'output', help='save result in this dir')
    parser.add_argument('--visualize', action='store_true', help='visualize video image')
    parser.add_argument('--save_img', action='store_true', help='save image')
  
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)



        # for tracking_box in tracking_boxes:
        #     raw_plate_img = crop_box(image, tracking_box, 5)
        #     _, enhanced_img = enhance_contrast(raw_plate_img)
        #     plate_img = perspective_transform(enhanced_img) 
            
        #     character_boxes = character_detector.detect(plate_img)
            
        #     new_boxes = convert_boxes_to_ratio(character_boxes, plate_img.shape)
        #     result = get_value(new_boxes)

        #     plate_img = character_detector.visualize(plate_img, character_boxes)

        #     if save_img:
        #         id = tracking_box[4]
        #         track_dir = os.path.join(save_dir, str(int(id)))        
        #         make_dir(track_dir)

        #         img_path = os.path.join(track_dir, str(count) + "_" + result + ".png")
        #         raw_img_path = os.path.join(track_dir, str(count) + "_raw.png")
        #         enhanced_img_path = os.path.join(track_dir, str(count) + "_enhanced.png")
        #         cv2.imwrite(img_path, plate_img)
        #         cv2.imwrite(enhanced_img_path, enhanced_img)
        #         cv2.imwrite(raw_img_path, raw_plate_img)