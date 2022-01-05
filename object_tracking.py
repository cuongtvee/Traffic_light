from __future__ import division, print_function, absolute_import
# import os
# import datetime
# import time
# import warnings
import cv2
import numpy as np
import argparse
from Char_Detection import *
# from PIL import Image
# from deep_sort import preprocessing
# from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
# from tools import generate_detections1 as gdet
from application_util import visualization
# from deep_sort.detection import Detection as ddet
from collections import deque
from check_moi import*
import math

# data_path = '../Dataset_A'
# from models.experimental import attempt_load
# list_video_path = '../Dataset_A/datasetA_vid_stats.txt'
# id_path = '../Dataset_A/list_video_id.txt'
# zones_path = './add/ROIs'
# roex_path = './add/ROI_e'
# video_path = '../Dataset_A'
# result_path = './submission_output'
# mois_path = './add/movement_description'
# multi_path = './add/movement_multi'
def load_model(path, train = False):
    model = attempt_load(path, map_location='cuda')  # load FP32 model
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if train:
        model.train()
    else:
        model.eval()
    return model, names
np.random.seed(1)
class OT(object):
    
    def __init__(self, class_id, names):
        self.max_cosine_distance = 0.9
        self.nms_max_overlap = 0.5
        self.nn_budget = None
        self.model_filename = 'model_data/market1501.pb'
        self.detections = []
        self.id = class_id
        self.boxes_tracked = []
        self.color = (255, 255, 255)
        self.char_model, names_ = load_model('char_detection.pt')
        self.class_names = names
        metric = None
        self.obtracker = Tracker(metric, id_cam = 0)
        self.pts = [deque(maxlen=1000) for _ in range(999999)]
        #self.point_first = [deque(maxlen=2) for _ in range(999999)]
        self.vis=visualization.Visualization(img_shape=(960,1280,3), update_ms=2000)
        COLORS = np.random.randint(100, 255, size=(255, 3), dtype="uint8")
        self.color = [int(c) for c in COLORS[self.id]]
        self.frame_track = 1
        self.begin_time = 0
        self.shape_img = None
        self.line=[(112,599),(1220,612)]

        
    def predict_obtracker(self, frame, dets):
        #tt = time.time()
        boxs = [d[:4] for d in dets]
        #features = gdet.HOG_feature(frame, boxs)
        #features = gdet.create_his(frame, boxs)
        self.detections = [Detection(det[:4], det[4], None,det[5]) for det in dets]    
        self.obtracker.predict()
        #print('times', np.round(time.time()-tt, 5))
    
    def update_obtracker(self,frame):
        self.obtracker.update(self.detections, self.frame_track,frame)
    def remove_track(self, ids_del):
        for track in self.obtracker.tracks:
            if track.track_id == ids_del:
                self.obtracker.tracks.remove(track)
                break
    
    def tracking_ob1(self,frame,draw,traffic_sign):
        #print('number', len(self.obtracker.tracks))
        for track in self.obtracker.tracks:
            #lay ra nhung track da confirmed, sau do tim khoang cach center voi box vua detect va box da luu gan nhat, tiep tuc tinh toan neu khoang cach >5
            if (not track.is_confirmed()):# or not track.match):
                continue
            bbox = track.to_tlbr()
            color = self.color
            #bbox_center_point(x,y)
            center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
            #track_id[center]
            vel = 10
            dx  =0
            dy = 0
            if len(self.pts[track.track_id]) > 0:
                pl = self.pts[track.track_id][-1]
                vel = math.sqrt((center[0]-pl[0])**2+(center[1]-pl[1])**2)
                if vel < 5: continue
            #them vao mang center
            self.pts[track.track_id].append(center)
            #center point
            if draw: 
                cv2.circle(frame, (center), 4,(0,0,255), -1)

            old = len(self.pts[track.track_id])
            p0 = self.pts[track.track_id][0]
            p1 = self.pts[track.track_id][old-1]
            vel = math.sqrt((p0[0]-p1[0])**2+(p0[1]-p1[1])**2)
            return_res=False
            frame = cv2.line(frame, self.line[0], self.line[1], (255,255,0), 2)

            if vel >= 10:
                return_res=check_intersect(self.line,[p0,p1])


            if return_res and traffic_sign==1:
                track.break_rule=True
                print(track.track_id," out")
                pl=Plate_C(track.plates,0,self.char_model)
                re_string,visualize_image = pl.detect(track.track_id)
                self.remove_track(track.track_id)
            elif return_res and traffic_sign==0:
                self.remove_track(track.track_id)
            if track.break_rule:
                frame = cv2.circle(frame, (p1[0],p1[1]), 30, (0,0,255), -1)

        return frame
    def draw_direc(self, index, direc, frame):
        r = self.ROI[:-1]
        roi1 = r
        roi2 = r[1:]+r[:1]
        x = (roi1[index][0]+roi2[index][0])//2 + 10
        y = (roi1[index][1]+roi2[index][1])//2 + 10
        cv2.putText(frame, str(direc),(x,y),0, 5e-3 * 300, (0,0,255),3)
        return frame

    def visualize(self,frame):
        self.vis.set_image(frame.copy())
        self.vis.draw_detections(self.detections)
        self.vis.draw_trackers(self.obtracker.tracks)
        return self.vis.return_img()
        
    
    
    
    
    
    
    
        
        
        
