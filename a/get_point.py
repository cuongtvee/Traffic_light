import cv2
import numpy as np
refPt = []
RrefPt = []
def mouse_event(event, x, y, flags, param):
    global refPt
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)
        refPt.append([x, y])
        # RrefPt.append((int(x), int(y)))
def getPoly(image):
    # cap = cv2.VideoCapture(0)
    # ret, image = cap.read()
    # image = cv2.imread("Score-board.png")
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_event)

    while True:
        cv2.imshow("image", image)
        for mouseP in refPt:
            cv2.circle(image, (mouseP[0], mouseP[1]), 1, (0, 255, 0), 5)
        cv2.waitKey(1)
        if refPt.__len__() == 4:
            break
    cv2.destroyWindow("image")
    RrefPt = np.asarray(refPt)
    return refPt,RrefPt
poly_flag=False
cap = cv2.VideoCapture('./video/100.h264')
while cap.isOpened():
    ret, img = cap.read()
    cv2.imshow("full",img)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
    elif k==ord('g'):
        refPt,RrefPt=getPoly(img)
        poly_flag=True
    # img = cv2.imread("test.png")
    if poly_flag:
        height = img.shape[0]
        width = img.shape[1]

        mask = np.zeros((height, width), dtype=np.uint8)
        # points = np.array([[[10,150],[150,100],[300,150],[350,100],[310,20],[35,10]]])
        cv2.fillPoly(mask, np.array([refPt]), (255))

        res = cv2.bitwise_and(img,img,mask = mask)

        rect = cv2.boundingRect(RrefPt) # returns (x,y,w,h) of the rect
        cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
        print(cropped.shape)
        cv2.imshow("cropped" , cropped )
        cv2.imshow("same size" , res)
    
    # mask = cv2.fillPoly(mask, np.array([refPt]), (255))

