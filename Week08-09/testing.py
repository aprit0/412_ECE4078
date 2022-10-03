import sys, os

import numpy as np
import json
import argparse
import time

fname = "M4_true_map.txt"
with open(fname, 'r') as fd:
    gt_dict = json.load(fd)
    fruit_list = []
    fruit_true_pos = []
    aruco_true_pos = np.empty([10, 2])

    # remove unique id of targets of the same type
    for key in gt_dict:
        x = np.round(gt_dict[key]['x'], 1) # Reading every x coordinates
        y = np.round(gt_dict[key]['y'], 1) # # Reading every y coordinates


        if key.startswith('aruco'):
            if key.startswith('aruco10'):
                aruco_true_pos[9][0] = x
                aruco_true_pos[9][1] = y
            else:
                marker_id = int(key[5]) - 1  # Giving ID to ARUCO markers
                aruco_true_pos[marker_id][0] = x
                aruco_true_pos[marker_id][1] = y
                print(marker_id, x, y)
        else:
            fruit_list.append(key[:-2])
            if len(fruit_true_pos) == 0:
                fruit_true_pos = np.array([[x, y]])
            else:
                fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)
