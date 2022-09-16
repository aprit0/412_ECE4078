# estimate the pose of a target object detected
import numpy as np
import json
import os
from pathlib import Path
import ast
# import cv2
import math
from machinevisiontoolbox import Image
from sklearn.cluster import KMeans
import torch

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import PIL


# use the machinevision toolbox to get the bounding box of the detected target(s) in an image
def get_bounding_box(target_number, image_path):
    image = PIL.Image.open(image_path).resize((640, 480), PIL.Image.NEAREST)
    target = Image(image) == target_number
    blobs = target.blobs()
    [[u1, u2], [v1, v2]] = blobs[0].bbox  # bounding box
    width = abs(u1 - u2)
    height = abs(v1 - v2)
    center = np.array(blobs[0].centroid).reshape(2, )
    box = [center[0], center[1], int(width), int(height)]  # box=[x,y,width,height]
    # plt.imshow(fruit.image)
    # plt.annotate(str(fruit_number), np.array(blobs[0].centroid).reshape(2,))
    # plt.show()
    # assert len(blobs) == 1, "An image should contain only one object of each target type"
    return box


# read in the list of detection results with bounding boxes and their matching robot pose info
def get_image_info(base_dir, file_path, image_poses):
    # there are at most three types of targets in each image
    target_lst_box = [[], [], [], [], []]
    target_lst_pose = [[], [], [], [], []]
    completed_img_dict = {}

    ckpt = os.getcwd() + '/network/scripts/model/model.best.pt'
    # add the bounding box info of each target in each image
    # target labels: 1 = apple, 2 = lemon, 3 = person, 0 = not_a_target
    # weird labels: 0 = apple, 2 = lemon, 3 = orange, 4 = pear, strawberry = 5
    img_vals = set(Image(base_dir / file_path, grey=True).image.reshape(-1))
    model = torch.hub.load('./yolov5', 'custom', path=ckpt, source='local')
    image = PIL.Image.open(base_dir / file_path).resize((640, 480), PIL.Image.NEAREST)
    # image.show()
    out = model(image)
    # generate dict
    for object in out.xyxy[0]:
        try:
            target_num = 1 if int(object[-1]) == 0 else int(object[-1])
            width, height = float(object[2] - object[0]), float(object[3] - object[1])
            x, y = float(object[0]) + width / 2, float(object[1]) + height / 2
            box = [x, y, width, height]
            pose = image_poses  # [i[0] for i in image_poses]
            target_lst_box[target_num - 1].append(box)  # bouncing box of target
            target_lst_pose[target_num - 1].append(np.array(pose).reshape(3, ))  # robot pose
        except ZeroDivisionError:
            pass
    # if there are more than one objects of the same type, combine them
    for i in range(5):
        if len(target_lst_box[i]) > 0:
            box = np.stack(target_lst_box[i], axis=0)
            pose = np.stack(target_lst_pose[i], axis=0)
            completed_img_dict[i + 1] = {'target': box, 'robot': pose}
    return completed_img_dict


# estimate the pose of a target based on size and location of its bounding box in the robot's camera view and the robot's pose
def estimate_pose(base_dir, camera_matrix, completed_img_dict):
    camera_matrix = camera_matrix
    focal_length = camera_matrix[0][0]
    # actual sizes of targets [For the simulation models]
    # You need to replace these values for the real world objects
    target_dimensions = []  # [ W, D, H ]
    apple_dimensions = [0.075448, 0.074871, 0.071889]
    target_dimensions.append(apple_dimensions)
    lemon_dimensions = [0.060588, 0.059299, 0.053017]
    target_dimensions.append(lemon_dimensions)
    pear_dimensions = [0.0946, 0.0948, 0.135]
    target_dimensions.append(pear_dimensions)
    orange_dimensions = [0.0721, 0.0771, 0.0739]
    target_dimensions.append(orange_dimensions)
    strawberry_dimensions = [0.052, 0.0346, 0.0376]
    target_dimensions.append(strawberry_dimensions)

    target_list = ['apple', 'lemon', 'orange', 'pear', 'strawberry']

    target_pose_dict = {}
    # for each target in each detection output, estimate its pose

    for target_num in completed_img_dict.keys():
        for item_num in range(len(completed_img_dict[target_num]['target'])):
            box = completed_img_dict[target_num]['target'][item_num]  # [[x],[y],[width],[height]]
            robot_pose = completed_img_dict[target_num]['robot'][item_num]  # [[x], [y], [theta]]
            true_height = target_dimensions[target_num - 1][2]

            ######### Replace with your codes #########
            # TODO: compute pose of the target based on bounding box info and robot's pose
            theta = robot_pose[2]
            Rotate = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])
            scaling_factor = true_height / box[3]
            depth = scaling_factor * focal_length
            hor = (camera_matrix[0][2] - box[0]) * scaling_factor
            xVector = np.array([depth, hor])
            xVector = Rotate @ xVector

            realWorld = xVector + robot_pose[0:2]

            target_pose = {'x': realWorld[0], 'y': realWorld[1]}
            target_pose_dict[target_list[target_num - 1]] = target_pose

        ###########################################

    return target_pose_dict


def clustering(fruit_list):
    # Transform the data
    X = np.squeeze(np.array(fruit_list))
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    centres = np.clip(kmeans.cluster_centers_, -1.2, 1.2)
    centres = list(centres)
    return [[i[1], i[0]] for i in centres]


# merge the estimations of the targets so that there are at most 3 estimations of each target type
def merge_estimations(target_pose_dict):
    target_map = target_pose_dict
    apple_est, lemon_est, pear_est, orange_est, strawberry_est = [], [], [], [], []
    target_est = {}
    # combine the estimations from multiple detector outputs
    print('target_pose_dict', target_pose_dict)
    for f in target_map:
        for key in target_map[f]:
            if key.startswith('apple'):
                apple_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('lemon'):
                lemon_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('pear'):
                pear_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('orange'):
                orange_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('strawberry'):
                strawberry_est.append(np.array(list(target_map[f][key].values()), dtype=float))

    ######### Replace with your codes #########
    # TODO: the operation below takes the first three estimations of each target type, replace it with a better merge solution
    if len(apple_est) > 2:
        apple_est = clustering(apple_est)
    elif not apple_est:
        apple_est = [[0., 0.], [0., 0.]]
    else:
        apple_est.append([0., 0.])
    if len(lemon_est) > 2:
        lemon_est = clustering(lemon_est)
    elif not lemon_est:
        lemon_est = [[0., 0.], [0., 0.]]
    else:
        lemon_est.append([0., 0.])
    if len(pear_est) > 2:
        pear_est = clustering(pear_est)
    elif not pear_est:
        pear_est = [[0., 0.], [0., 0.]]
    else:
        pear_est.append([0., 0.])
    if len(orange_est) > 2:
        orange_est = clustering(orange_est)
    elif not orange_est:
        orange_est = [[0., 0.], [0., 0.]]
    else:
        orange_est.append([0., 0.])
    if len(strawberry_est) > 2:
        strawberry_est = clustering(strawberry_est)
    elif not strawberry_est:
        strawberry_est = [[0., 0.], [0., 0.]]
    else:
        strawberry_est.append([0., 0.])

    for i in range(2):
        try:
            target_est['apple_' + str(i)] = {'y': apple_est[i][0], 'x': apple_est[i][1]}
        except:
            pass
        try:
            target_est['lemon_' + str(i)] = {'y': lemon_est[i][0], 'x': lemon_est[i][1]}
        except:
            pass
        try:
            target_est['pear_' + str(i)] = {'y': pear_est[i][0], 'x': pear_est[i][1]}
        except:
            pass
        try:
            target_est['orange_' + str(i)] = {'y': orange_est[i][0], 'x': orange_est[i][1]}
        except:
            pass
        try:
            target_est['strawberry_' + str(i)] = {'y': strawberry_est[i][0], 'x': strawberry_est[i][1]}
        except:
            pass
    ###########################################

    return target_est


if __name__ == "__main__":
    # camera_matrix = np.ones((3,3))/2
    fileK = "{}intrinsic.txt".format('./calibration/param/')
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    base_dir = Path('./')

    # a dictionary of all the saved detector outputs
    image_poses = {}
    with open(base_dir / 'lab_output/images.txt') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line)
            image_poses[pose_dict['imgfname']] = pose_dict['pose']

    # estimate pose of targets in each detector output
    target_map = {}
    for file_path in image_poses.keys():
        print(file_path)
        completed_img_dict = get_image_info(base_dir, file_path, image_poses[file_path])
        target_map[file_path] = estimate_pose(base_dir, camera_matrix, completed_img_dict)

    # merge the estimations of the targets so that there are at most 3 estimations of each target type
    target_est = merge_estimations(target_map)

    # save target pose estimations
    with open(base_dir / 'lab_output/targets.txt', 'w') as fo:
        print(target_est)
        json.dump(target_est, fo)

    print('Estimations saved!')
