import numpy as np
import cv2

from utils.class_config import CHARACTER_CLASS

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def perspective_transform(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY )

    cont, _ = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    if cont == ():
        return img

    cont = sorted(cont, key= lambda cont:cv2.contourArea(cont), reverse = True)
    a = cont[0].reshape(cont[0].shape[0], cont[0].shape[2])

    rect = cv2.minAreaRect(a)


    box = cv2.boxPoints(rect)
    box = np.int0(box)
    box = np.array(box)
    rect = order_points(box)

    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")
    # # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    if warped.shape[0] < 20:
        return img

    if warped.shape[1] < 20:
        return img
    return warped

def check_is_square_plate(boxes):
    return np.mean([box[-1] for box in boxes]) < 0.65

def sort_boxes_along_x(boxes):
    indice = np.argsort([box[0] for box in boxes]) 
    return np.array(boxes)[indice]

def get_character(boxes):
    result = ""
    for box in boxes:
        result += CHARACTER_CLASS[int(box[-2])]

    return result

def get_value(boxes):
    if boxes == []:
        return "Empty"
    if check_is_square_plate(boxes):
        upper_character = []
        lower_character = []
        for box in boxes:
            if box[1] < 0.5:
                upper_character.append(box)
            else:
                lower_character.append(box)
        
        sorted_upper_character = sort_boxes_along_x(upper_character)
        sorted_lower_character = sort_boxes_along_x(lower_character)
        upper_string = get_character(sorted_upper_character)
        lower_string = get_character(sorted_lower_character)

        result_string = upper_string + "-" + lower_string

    else:
        sorted_character = sort_boxes_along_x(boxes)
        result_string = get_character(sorted_character)

    return result_string

def enhance_contrast(img):
  b_img, g_img, r_img = cv2.split(img)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
  equalized_b_img = clahe.apply(b_img)
  equalized_g_img = clahe.apply(g_img)
  equalized_r_img = clahe.apply(r_img)

  return cv2.merge([equalized_b_img, equalized_g_img, equalized_r_img]),\
         cv2.merge([cv2.equalizeHist(b_img), cv2.equalizeHist(g_img), cv2.equalizeHist(r_img)])