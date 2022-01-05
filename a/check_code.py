import cv2
import os
from shapely.geometry import Point, MultiPoint
data_path = '../Dataset_A'
list_video_path = '../Dataset_A/datasetA_vid_stats.txt'
id_path = '../Dataset_A/list_video_id.txt'
zones_path = './add/ROIs'
roex_path = './add/ROI_e'
video_path = '../Dataset_A'
result_path = './submission_output'
mois_path = './add/movement_description'
multi_path = './add/movement_multi'
def load_roi_moi(rois_path, rois_ex_path, mois_path, multi_path, name_video):
    roi = []
    roi_ex = []
    mois = {}
    multi = {}
    list_moi_edge=[]
    cam_index = name_video.split('_')[1]
    if len(cam_index) > 2:
        cam_index = cam_index.split('.')[0]
    with open(os.path.join(rois_path, 'cam_{}.txt'.format(cam_index))) as f:
        for p in f:
            p = p.rstrip("\n")
            p = p.split(',')
            temp = p[2:]
            temp = [int(x) for x in temp]
            list_moi_edge.append(temp)
            roi.append((int(p[0]), int(p[1])))
    roi.append(list_moi_edge)
    
    with open(os.path.join(rois_ex_path, 'cam_{}.txt'.format(cam_index))) as f:
        for p in f:
            p = p.rstrip("\n")
            p = p.split(',')
            temp = p[2:]
            temp = [int(x) for x in temp]
            roi_ex.append((int(p[0]), int(p[1])))

    with open(os.path.join(multi_path, 'cam_{}.txt'.format(cam_index))) as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if len(line) == 0: continue
            a = line.split(',')
            temp = []
            for ii in range(0,len(a)-1,2):
              temp.append((int(a[ii]), int(a[ii+1])))
            print(temp)
            multi[i+1]=temp

    with open(os.path.join(mois_path, 'cam_{}.txt'.format(cam_index))) as f:
      for i, line in enumerate(f):
        line = line.rstrip("\n")
        if len(line) == 0: continue
        a = line.split(',')
        p1 = (int(a[0]),int(a[1]))
        p2 = (int(a[2]),int(a[3]))
        p3 = (int(a[4]),int(a[5]))
        p4 = (int(a[6]),int(a[7]))
        l1 = (int(a[8]),int(a[9]))
        l2 = (int(a[10]),int(a[11]))
        mois[i+1]=[p1,p2,p3,p4,l1,l2]

    return roi, roi_ex, mois, multi
def load_list_video(input_path, id_path):
    names = []
    ids = []
    info = []
    with open(id_path,'r') as f:
        for line in f:
            a = line.split(' ')
            ids.append(a[0])
            names.append(a[-1].split('\n')[0])

    with open(input_path,'r') as f:
        for line in f:
            video_name = line.split('\t')[0]

            try:
                fps = line.split('\t')[1]
                total_frame = int(line.split('\t')[2])
                if fps != 'fps':
                    fps = fps.split('/')[:-1]
                    fps = int(fps[0])
                id = ids[names.index(video_name)]
                info.append([id, video_name, fps, total_frame])
            except:
                pass
    #print(info)
    return info

def convert_multiPoint(MOI):
    sets = []
    for d, ps in MOI.items():
      a = []
      for p in ps: 
        a.append(Point(p))
      sets.append(a)
    return sets
def draw_roi(roi, frame):
    roi_nums = len(roi)-1
    for i in range(roi_nums):
        cv2.putText(frame, str(i)+"direc",roi[i],0, 5e-3 * 300, (0,0,255),2)
        if i < roi_nums-1:
            cv2.line(frame,roi[i],roi[i+1],(0,255,0),2)
        else:
            cv2.line(frame,roi[i],roi[0],(0,255,0),2)
    return frame
video_capture = cv2.VideoCapture("../Dataset_A/cam_1_dawn.mp4")
info_cam = load_list_video(list_video_path, id_path)
for info in info_cam: 
    path = os.path.join(video_path, info[1])
    ROI, ROI_ex, MOI, multi = load_roi_moi(zones_path, roex_path, mois_path, multi_path, info[1])
    setofpoint = convert_multiPoint(multi)
    # frame_delay = load_delay('./add/Dataset_A/time_delay.txt', info[1])
    name = info[1].split('.')[0]
    print("Processing video: ", info)
    # print(multi)
print(ROI)
while True:
    ret,frame = video_capture.read()
    # print(len(ROI))
    
    draw_roi(ROI, frame)
    for i,(d, ps) in enumerate(MOI.items()):
        a = []
        for p in ps:
            frame = cv2.circle(frame, (p[0],p[1]), 2, (255,0,0), 2)
            cv2.putText(frame, str(i),(p[0],p[1]),0, 5e-3 * 300, (0,0,255),2)
    for d, ps in multi.items():
        a = []
        for p in ps:
            frame = cv2.circle(frame, (p[0],p[1]), 2, (255,0,255), 2)

    cv2.imshow("image",frame)
    cv2.imwrite("MOIandMulti.jpg",frame)
    k=cv2.waitKey(1)
    if k==ord(" "):
        k=None
        while(k!=ord(" ")):
            k=cv2.waitKey(1)
    elif k==ord("q"):
        break