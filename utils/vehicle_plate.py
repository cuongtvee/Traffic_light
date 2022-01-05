import numpy as np

from utils.my_general import ioa_batch

def get_vehicle_license_plate_pair(car_boxes, license_plate_boxes):
    is_sucess, ioa = ioa_batch(car_boxes, license_plate_boxes)
    idx = (np.array([], dtype= np.int64), np.array([], dtype= np.int64))

    idx = (np.array([], dtype= np.int64), np.array([], dtype= np.int64))
    if is_sucess:
        idx = np.where(ioa == 1)
    
    car_idx =  idx[0]
    license_plate_idx = idx[1]

    vehicle_license_plate_in_pairs = []
    
    for car, license_plate in zip(car_idx,license_plate_idx):
        vehicle_license_plate_in_pair = (car_boxes[car], license_plate_boxes[license_plate])
        vehicle_license_plate_in_pairs.append(vehicle_license_plate_in_pair)

    return vehicle_license_plate_in_pairs