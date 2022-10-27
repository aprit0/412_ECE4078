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
    detPandas = out.pandas().xyxy[0]
    print('TPE: OUT\n',out.xyxy[0],'\n')
    # generate dict
    object = out.xyxy[0][0]
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
    orange_dimensions = [0.0721, 0.0771, 0.0739]
    target_dimensions.append(orange_dimensions)
    pear_dimensions = [0.0946, 0.0948, 0.135] #135
    target_dimensions.append(pear_dimensions)
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
            print('TRUE HEIGHT\n',  target_list[target_num-1], true_height)
            print('robot_pose\n', robot_pose)

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


def clustering(fruit_list, n_clusters=2):
    # Remove extreme outliers
    new_list = []
    for point in fruit_list:
        if abs(point[0]) < 1.5 and abs(point[1]) < 1.5:
            new_list.append(point)
    if len(new_list) > 0:
        # Transform the data
        if len(new_list) == 1:
            X = np.array(new_list)
        else:
            X = np.squeeze(np.array(new_list))
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
            centres = np.clip(kmeans.cluster_centers_, -1.2, 1.2)
            if n_clusters == 1:
               pass
            else:
                dist = np.linalg.norm(centres[0] - centres[1])
                print(f'\ndist {dist}\n')
                if dist < 0.4:
                    kmeans = KMeans(n_clusters=1, random_state=0).fit(X)
                    centres = list(np.clip(kmeans.cluster_centers_, -1.2, 1.2))
        except Exception as e:
            print('Clusterfuck', e)
            kmeans = KMeans(n_clusters=1, random_state=0).fit(X)
            centres = list(np.clip(kmeans.cluster_centers_, -1.2, 1.2))
    else:
        centres = [[0., 0.]]*n_clusters
    return [[i[1], i[0]] for i in centres]


# merge the estimations of the targets so that there are at most 3 estimations of each target type
def merge_estimations(target_pose_dict, search_list):
    keys = ['apple', 'lemon', 'pear', 'orange', 'strawberry']
    target_map = {key:[] for key in keys}
    # combine the estimations from multiple detector outputs
    print('target_pose_dict', target_pose_dict)
    for f in target_pose_dict:
        for key in target_pose_dict[f]:

            target_map[key].append(np.array(list(target_pose_dict[f][key].values()), dtype=float))

    print('---target_map---\n', target_map)
    ######### Replace with your codes #########
    # TODO: the operation below takes the first three estimations of each target type, replace it with a better merge solution
    target_est = {}
    for key in target_map.keys():
        if key in search_list:
            n_clusters = 1
        else:
            n_clusters = 2
        est = clustering(target_map[key], n_clusters)
        for i in range(n_clusters):
            print('TPE key:', key, i, est)
            try:
                target_est[f'{key}_{i}'] = {'y': est[i][0], 'x': est[i][1]}
            except Exception as e:
                print('TPE: ', e)

    print('---target_est---\n', target_est)
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
        print('target_pose_est\n', target_est)
        json.dump(target_est, fo)

    print('Estimations saved!')
