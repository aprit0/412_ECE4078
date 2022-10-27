# detect ARUCO markers and estimate their positions
import numpy as np
import cv2
import os, sys

sys.path.insert(0, "{}/util".format(os.getcwd()))
import util.measure as measure

class aruco_detector:
    def __init__(self, robot, marker_length=0.07):
        self.camera_matrix = robot.camera_matrix
        self.distortion_params = robot.camera_dist

        self.marker_length = marker_length
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
    
    def detect_marker_positions(self, img):
        # Perform detection
        corners, ids, rejected = cv2.aruco.detectMarkers(
            img, self.aruco_dict, parameters=self.aruco_params)
        # rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.distortion_params) # use this instead if you got a value error

        if ids is None:
            return [], img
        # remove duplicates in corners
        corn = np.unique(corners)

        # check if there are double ups
        # remove double up with smaller area
        # ids: list of id in order
        # corners: list of corners in order
        id_dict = {}
        for i in range(len(ids)):
            id = int(ids[i])
            corn = corners[i]
            if id in id_dict:
                id_dict[id].append(corn)
            else:
                id_dict[id] = [corn]
        for id in id_dict:
            if len(id_dict[id]) > 1:
                # get max dist of corners and save that one
                max_euc = 0
                max_corn = None
                for id_corn in id_dict[id]:
                    euc = np.linalg.norm(id_corn[0][2] - id_corn[0][0])
                    if max_euc < euc:
                        max_euc = euc
                        max_corn = id_corn
                id_dict[id] = max_corn
            else:
                id_dict[id] = id_dict[id][0]
        ids = list(id_dict.keys())
        ids = np.array([[i] for i in ids])
        vals_dict = id_dict.values()
        corners = [np.array(i) for i in vals_dict]

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_length, self.camera_matrix, self.distortion_params)
        measurements = []
        seen_ids = []
        for i in range(len(ids)):
            idi = ids[i][0]
            lm_tvecs = tvecs[ids==idi].T
            lm_bff2d = np.block([[lm_tvecs[2,:]],[-lm_tvecs[0,:]]])
            lm_bff2d = np.mean(lm_bff2d, axis=1).reshape(-1,1)

            lm_measurement = measure.Marker(lm_bff2d, idi)
            measurements.append(lm_measurement)
        
        # Draw markers on image copy
        img_marked = img.copy()
        cv2.aruco.drawDetectedMarkers(img_marked, corners, ids)

        return measurements, img_marked
