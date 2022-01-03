import cv2
from utils.license_plate_general import get_loader, perspective_transform, \
                                        drop_cls, crop_box, make_dir, ioa_batch, get_vehicle_license_plate_pair, \
                                        convert_boxes_to_ratio, enhance_contrast, get_value, get_specific_classes

import os
import numpy as np 
def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
	print(x, y)




# green s channel 22, 100
# red v channel 60 150 
# yellow s channel 22, 120

def is_turn_on(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    idx = np.where(hsv_image[:,:, :] > 200)[0]
    if idx.shape[0] > 100:
        return True
    return False

def red_image(image):
    """
        True = turn on
        False = Turn off
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    if np.mean(hsv_image[:,:, 2]) > 105:
        return True
    
    return False

def green_image(image):
    """
        True = turn on
        False = Turn off
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    if np.mean(hsv_image[:,:, 1]) > 60:
        return True
    
    return False

def yellow_image(image):
    """
        True = turn on
        False = Turn off
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    if np.mean(hsv_image[:,:, 1]) > 70:
        return True
    
    return False


vehicle_config =  r'my_models\vehicle_config.yaml'

dataset, video_name = get_loader(vehicle_config)                      ## cant not load camera


yellow_folder = r'my_models\yellow'
red_folder = r'my_models\red'
green_folder = r'my_models\green'

turn_off_folder = r'my_models\off'

for _, image, _, count in dataset:
    red_light = image[34 :75, 1549: 15955]
    yellow_light = image[86 :128, 1549: 15955]
    green_light = image[137 :178, 1549: 15955]
    # cv2.namedWindow("image")
    # cv2.setMouseCallback("image", click_and_crop)

    result_img = cv2.resize(image, (1280, 720))
    cv2.imshow("image", result_img)
    cv2.imshow("red_light", red_light)
    cv2.imshow("yellow_light", yellow_light)
    cv2.imshow("green_light", green_light)

    traffic_light = [is_turn_on(red_light), is_turn_on(yellow_light), is_turn_on(green_light)]
    if is_turn_on(red_light):
        print("RED")
        red_light_path  = os.path.join(red_folder, "red_light_" + str(count) + ".png")
        #cv2.imwrite(red_light_path, red_light)
    else:
        red_light_path  = os.path.join(turn_off_folder, "red_light_" + str(count) + ".png")
        #cv2.imwrite(red_light_path, red_light)
    
    if is_turn_on(green_light):
        print("GREEN")
        green_light_path  = os.path.join(green_folder, "green_light_" + str(count) + ".png")
        #cv2.imwrite(green_light_path, green_light)
    else:
        green_light_path  = os.path.join(turn_off_folder, "green_light_" + str(count) + ".png")
        #cv2.imwrite(green_light_path, green_light)
    
    if is_turn_on(yellow_light):
        print("YELLOW")
        yellow_light_path  = os.path.join(yellow_folder, "yellow_light_" + str(count) + ".png")
        #cv2.imwrite(yellow_light_path, yellow_light)
    else:
        yellow_light_path  = os.path.join(turn_off_folder, "yellow_light_" + str(count) + ".png")
        #cv2.imwrite(yellow_light_path, yellow_light)

    print(traffic_light, "\n \n")
    # red_light_path  = os.path.join(red_folder, "red_light_" + str(count) + ".png")
    # green_light_path  = os.path.join(green_folder, "green_light_" + str(count) + ".png")
    # yellow_light_path  = os.path.join(yellow_folder, "yellow_light_" + str(count) + ".png")
    # cv2.imwrite(red_light_path, red_light)
    # cv2.imwrite(yellow_light_path, yellow_light)
    # cv2.imwrite(green_light_path, green_light)

    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break


# (1549, 34)   (1595, 34)  
#red
# (1549, 75)   (1595, 75)  



# (1549, 86)   (1595, 86)  
#yellow
# (1549, 128)   (1595, 128)  


# (1549, 137)   (1595, 137)  
#green
# (1549, 178)   (1595, 178)  