import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

yellow_folder = r'my_models\yellow'
red_folder = r'my_models\red'
green_folder = r'my_models\green'


turn_on_red_folder = r'my_models\on_yellow'
turn_off_red_folder = r'my_models\off_yellow'

for file in os.listdir(yellow_folder):
    print(file)
    red_image = cv2.imread(os.path.join(yellow_folder, file))


    hsv_red_image = cv2.cvtColor(red_image, cv2.COLOR_RGB2HSV)

    for channel_id, c in zip((0, 1, 2), ("red", "green", "blue")):
        histogram, bin_edges = np.histogram(
            hsv_red_image[:, :, channel_id], bins=256, range=(0, 256)
        )
        plt.plot(bin_edges[0:-1], histogram, color=c)

    plt.xlabel("Color value")
    plt.ylabel("Pixels")

    plt.show()
    test = np.where(hsv_red_image[:,:, :] > 200)
    print(test[0].shape)
    print(test[1].shape)
    print(test[2].shape)
    if np.mean(hsv_red_image[:,:, 2]) > 105:
        print("Turn on")
        # save_path = os.path.join(turn_on_red_folder, file)
        # cv2.imwrite(save_path, red_image)
    else:
        print("turn off")
        # save_path = os.path.join(turn_off_red_folder, file)
        # cv2.imwrite(save_path, red_image)

    cv2.imshow("hsv", hsv_red_image)
    cv2.imshow("image", red_image)

    cv2.waitKey(0)



# green s channel 22, 100
# red v channel 60 150 
# yellow s channel 22, 120