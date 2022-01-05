
import time
import numpy as np

import cv2

import torch
import torch.backends.cudnn as cudnn
from numpy import random
from yolov5.models.experimental import attempt_load

from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.torch_utils import select_device

# from models.models import *
from models.experimental import *
from utils.datasets import *
from utils.general import *
from object_tracking import OT
from utils.license_plate_general import *
from utils.detector import *
begin_script = time.time()




def detect(objects,skip):
    traff_arr = ["xanh","do","vang"]
    traffic_sign = 0 #0: xanh; 1: do; 2: vang
    line= [(112,599),(1220,612)]
    writeVideo_flag=True
    # Run inference
    t0 = time.time()
    skip_frame = 1
    frame_track = 1
    video_capture = cv2.VideoCapture('./video/100.h264')


    w = int(video_capture.get(3))
    h = int(video_capture.get(4))
    shape_img = (w,h)
    detection=Detector('vehicle_config.yaml')
    begin_time = time.time()
    while True:

        ret,frame = video_capture.read()
        res = frame.copy()
        plate_img=frame.copy()
        if ret != True:
            break
        if skip_frame <= skip :
            skip_frame +=1
            continue

    
        dets=detection.detect(frame)

        red_light_image, yellow_light_image, green_light_image = get_traffic_light_image(frame)
        if is_turn_on(red_light_image):
            traffic_sign = 1
        elif is_turn_on(green_light_image):
            traffic_sign = 0
        else :
            traffic_sign = 2


        for det, ob in zip (dets, objects):
            ob.begin_time = begin_time
            ob.frame_track = frame_track#frame_num
            ob.shape_img = shape_img
            #predict kalman filter su dung det
            ob.predict_obtracker(res, det)
            ob.update_obtracker(plate_img)
            if writeVideo_flag: res= ob.visualize(res)
            res= ob.tracking_ob1(res,writeVideo_flag,traffic_sign)
        res = cv2.line(res, line[0], line[1], (255,255,0), 10)

        if writeVideo_flag:
          cv2.putText(res, traff_arr[traffic_sign],(50,50),0, 5e-3 * 300, (0,100,255),10)
          cv2.imshow("re",cv2.resize(res,(640,480)))
          k=cv2.waitKey(1)
          if k==ord(" "):
              k=None
              while(k!=ord(" ")):
                  k=cv2.waitKey(1)
          elif k==ord("q"):
              break
    

    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    
    with torch.no_grad():

      imgsz = 1280
      device = select_device('')
      model = attempt_load('./best.pt', map_location=device)    

      if device.type != 'cpu':
          model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run onc
      counter = []
      writeVideo_flag = True
      skip=2

      car_class = OT(1,'car')
      motorbike = OT(2,'motorbike')
      objects = [car_class, motorbike]
      detect(objects, skip)
      del objects
      print('saved ')
      print('Done')
      print('total time: {} sec'.format(np.round(time.time()-begin_script,3)))
       
          
