import numpy as np
import math
import cv2
from scipy.optimize import linear_sum_assignment 
import torch
import imutils
import random
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.general import non_max_suppression
from models.experimental import attempt_load
PATTERN_CLASS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 
                 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
class Plate_C(object):
    def __init__(self, plates,alpha,model):
        self.alpha = alpha
        self.plates = plates
        self.type = self.get_type()
        self.model=model
    def get_type(self):
        type_arr=[]
        for plate in self.plates:
            type_arr.append(plate[1])
        # print(type_arr)
        type_l=max(type_arr,key=type_arr.count)
        # print(type_l)
        return type_l
    def detect(self,id):
        
        old_char=[]
        
        string_re=" "
        visualize_image = np.zeros((128,128,3),dtype=np.uint8)
        for idx,plate_and_type in enumerate(self.plates):
            plate=plate_and_type[0]
            track_box,visualize_image=self.char_detection_yolo(plate, self.model)
            
            arr_track=self.matching_char(old_char,track_box)
            visualize_image2 = np.zeros((128,128,3),dtype=np.uint8)
            # print("len arr track",len(arr_track))
            chars=[]
            center=[]
            for i in arr_track:
                
            # print(i,clss)
                # print("day la i",i)
                clss=max(i[3],key=i[3].count)
                chars.append("{}".format(clss))
                        
                center.append([int(i[0]),int(i[1])])
                cv2.putText(visualize_image2, "{}".format(str(PATTERN_CLASS[int(clss)])),(int(i[0]),int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,255,255), 1)
            # print("len bf char",len(chars))
            # print("len bf center",len(center))
            if self.type==0:
                degree,string_re,visualize_image2=self.find_line_plate(center,chars,visualize_image2)
            else:
                degree,string_re,visualize_image2=self.find_line_longplate(center,chars,visualize_image2)
            self.alpha+=degree
            old_char=arr_track
            # -------------
            mask_1= np.zeros((128,256,3),dtype=np.uint8)
            cv2.putText(mask_1, "{}     {}".format("predict","after process"),(0,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255,255,255), 1)
            cv2.putText(mask_1, "{}".format(string_re),(60,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255,255,255), 1)
            visualize_image= np.hstack((visualize_image,visualize_image2))
            # print(visualize_image.shape,mask_1.shape)
            visualize_image= np.vstack((visualize_image,mask_1))
            if idx ==len(self.plates)-1:
                idx=random.randint(0, 999)
                cv2.imwrite("img/"+str(id)+"___"+str(idx)+"processed"+".jpg",visualize_image)
                cv2.imwrite("img/"+str(id)+"___"+str(idx)+"raw"+".jpg",plate)
            # -------------
        return string_re,visualize_image
    def find_line_plate(self,center,chars,visualize_image):
        # print("len char",len(chars))
        # print("len center",len(center))
        if len(chars)<3:
            return 0,"",visualize_image
        a_min=1
        a_max=0       
        for i in range(0,len(center)-1):
            for j in range(i+1, len(center)):
                x1, y1 = center[i][0], center[i][1]
                x2, y2 = center[j][0], center[j][1]
                if x1-x2==0:
                    Dmax=[i,j]
                    a_max=99999999999
                    continue
                a=(y1-y2)/(x1-x2)
                b=y1-a*x1
                if abs(a)<abs(a_min):
                    a_min=a
                    Dmin=[i,j]
                if abs(a)>=abs(a_max):
                    a_max=a
                    Dmax=[i,j]

        midpoint=(int((center[Dmax[0]][0]+center[Dmax[1]][0])/2),
                        int((center[Dmax[0]][1]+center[Dmax[1]][1])/2))
        b_pt=midpoint[1]-a_min*midpoint[0]
        point1=(0,int(b_pt))
        point2=(128,int(a_min*128+b_pt))
        angle=math.acos(128/(math.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)))
        if a_min>0:
            angle=-1*angle
        visualize_image = cv2.line(visualize_image, point1, 
                point2, (255,255,255), 2)
        upper=[]
        lower=[]
        for c,ch in zip(center,chars):
            if a_min*c[0]+b_pt-c[1]<0:
                lower.append([c[0],ch])
            else:
                upper.append([c[0],ch])

        upper=sorted(upper, key=lambda x:x[0])
        lower=sorted(lower, key=lambda x:x[0])
        string_re=" "
        for u in upper:
            string_re+=str(PATTERN_CLASS[int(u[1])])
        string_re+="*"
        for l in lower:
            string_re+=str(PATTERN_CLASS[int(l[1])])
        # print(string_re)
        # print("angle",math.degrees(angle))
        if len(center)==2 or abs(a_max)<0.5:
            angle=0
        # print("len",len(string_re))
        # print(string_re)
        return angle,string_re,visualize_image
    def find_line_longplate(self,center,chars,visualize_image):
        # print("len char",len(chars))
        # print("len center",len(center))
        if len(chars)<3:
            return 0,"",visualize_image
        a_min=1
        a_max=0       
        for i in range(0,len(center)-1):
            for j in range(i+1, len(center)):
                x1, y1 = center[i][0], center[i][1]
                x2, y2 = center[j][0], center[j][1]
                if x1-x2==0:
                    Dmax=[i,j]
                    a_max=99999999999
                    continue
                a=(y1-y2)/(x1-x2)
                b=y1-a*x1
                if abs(a)<abs(a_min):
                    a_min=a
                    Dmin=[i,j]

        point1=Dmin[0]
        point2=Dmin[1]
        angle=math.acos(128/(math.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)))
        if a_min>0:
            angle=-1*angle
        visualize_image = cv2.line(visualize_image, point1, 
                point2, (255,255,255), 2)
        # upper=[]
        # lower=[]
        middle=[]
        for c,ch in zip(center,chars):
            # if a_min*c[0]+b_pt-c[1]<0:
            middle.append([c[0],ch])


        middle=sorted(middle, key=lambda x:x[0])
        # lower=sorted(lower, key=lambda x:x[0])
        string_re=" "
        for u in middle:
            string_re+=str(PATTERN_CLASS[int(u[1])])

        return angle,string_re,visualize_image
    
    def matching_char(self,track_arr,arr_case2):
        def distance(a,b):
            dis=math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
            return dis
        def linear_assignment(cost_matrix):
            x, y = linear_sum_assignment(cost_matrix)
            return np.array(list(zip(x, y)))
        dis_mat= np.zeros((len(track_arr),len(arr_case2)),dtype=np.float32)
        for t,trk in enumerate(track_arr):
            for d,det in enumerate(arr_case2):
                dis_mat[t,d] = distance(trk,det)
        # print(dis_mat)
        if (dis_mat).size != 0:
            matched_idx = linear_assignment(dis_mat)
        else:
            matched_idx = np.empty((0,2))
        # print(matched_idx)

        unmatched_trackers, unmatched_detections = [], []
        for t,trk in enumerate(track_arr):
            if(t not in matched_idx[:,0]):
                unmatched_trackers.append(t)

        for d, det in enumerate(arr_case2):
            if(d not in matched_idx[:,1]):
                unmatched_detections.append(d)

        matches = []
        for m in matched_idx:
            if(dis_mat[m[0],m[1]]>30):
                unmatched_trackers.append(m[0])
                unmatched_detections.append(m[1])
            else:
                matches.append(m.reshape(1,2))

        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)
        # print(matches)
        denta_v=[0,0]
        for m in matches:
            denta_v[0]+=arr_case2[m[1]][0]-track_arr[m[0]][0]
            denta_v[1]+=arr_case2[m[1]][1]-track_arr[m[0]][1]


            # print(arr_case2[m[1]][0:2],track_arr[m[0]][0:2])
            track_arr[m[0]][0:2]=arr_case2[m[1]][0:2]

            track_arr[m[0]][2]+=arr_case2[m[1]][2]
            track_arr[m[0]][3]+=arr_case2[m[1]][3]
        if len(matches)>0:
            denta_v[0]=denta_v[0]/len(matches)
            denta_v[1]=denta_v[1]/len(matches)


        for u in unmatched_detections:
            track_arr.append(arr_case2[u])
        for u in unmatched_trackers:
            track_arr[u][0]=track_arr[u][0]+denta_v[0]
            track_arr[u][1]=track_arr[u][1]+denta_v[1]
        return track_arr
    def char_detection_yolo(self,image, model, conf_thres = 0.7, \
                            iou_thres = 0.45, classes = None, \
                            agnostic_nms = False, max_det = 1000):
        #(128,128,3)
        visualize_image = np.zeros((128,128,3),dtype=np.uint8)
        # t=time.time()
        img = self.preprocess_image(image.copy(), (128,128))
        pred = model(img, augment= False, visualize= False)[0]
        new_pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det)
        # print("char detect time",time.time()-t)
        boxes = []
        center=[]
        chars=[]
        track_box=[]
        for det in new_pred:  
            # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
            det = det[:, :6].cpu().round().detach().numpy()
            for box in det:
                # box=list(box)
                if float(box[4]) > 0.5:
                    track_box.append([int((box[0]+box[2])/2),int((box[1]+box[3])/2),[box[4]],[int(box[5])]])
                    cv2.putText(visualize_image, "{}".format(str(PATTERN_CLASS[int(box[5])])),
                    (int(box[0]), int(box[1])+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,255,255), 1)
                # boxes.append(box)
                    boxes.append(box)
        return track_box,visualize_image
    def preprocess_image(self,original_image, size = (1280,1280), device = 'cuda'):
        # original_image = letterbox(original_image,new_shape=size)[0]
        original_image = imutils.rotate_bound(original_image, self.alpha)
        original_image=self.ResizeImg(original_image, size)
        
        image = original_image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        image = np.ascontiguousarray(image)
        
        image = torch.from_numpy(image).to(device)
        image = image.float()  
        image = image / 255.0 
        if image.ndimension() == 3:
            image = image.unsqueeze(0)
        return image

    def ResizeImg(self,img, size):
        h1, w1, _=img.shape
        # print(h1, w1, _)
        h,w= size
        if w1 < h1*(w/h):
            # print(w1/h1)
            char_digit = cv2.resize(img, (int(float(w1/h1)*h), h))
            # print(char_digit.shape[1]/char_digit.shape[0])
            mask = np.zeros((h, w-(int(float(w1/h1)*h)), 3), np.uint8)
            thresh = cv2.hconcat([char_digit, mask])
            trans_x = int(w/2)-int(int(float(w1/h1)*h)/2)
            trans_y = 0
            trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
            height, width = thresh.shape[:2]
            thresh1 = cv2.warpAffine(thresh, trans_m, (width, height))
            return thresh1
            # pass
        else:
            # print(w1/h1)
            char_digit = cv2.resize(img, (w, int(float(h1/w1)*w)))
            # print(char_digit.shape[1]/char_digit.shape[0])
            mask = np.zeros((h-int(float(h1/w1)*w), w, 3), np.uint8)
            thresh = cv2.vconcat([char_digit, mask])
            trans_x = 0
            trans_y = int(h/2)-int(int(float(h1/w1)*w)/2)
            trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
            height, width = thresh.shape[:2]
            thresh1 = cv2.warpAffine(thresh, trans_m, (width, height))
            return thresh1