# m4 - autonomous fruit searching
# Shamelessly stolen from Aidan's repo https://github.com/aprit0/Robot-4191/blob/main/Robot-4191/SPAM.py
# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
import random
import math
from operate06 import *
from matplotlib import pyplot as plt
import pyastar2d

# import dependencies and set random seed
seed_value = 5
# 1. set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 2. set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# import utility functions
sys.path.insert(0, "{}/utility".format(os.getcwd()))
from util.pibot import PenguinPi # access the robot
import util.DatasetHandler as dh # save/load functions
import util.measure as measure # measurements
import shutil # python package for file operations

# import SLAM components you developed in M2
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import CV components
sys.path.insert(0,"{}/network/".format(os.getcwd()))
sys.path.insert(0,"{}/network/scripts".format(os.getcwd()))
from network.scripts.detector import Detector


class Operate:
    def __init__(self, args):
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)

        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = PenguinPi(args.ip, args.port)

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length = 0.07) # size of the ARUCO markers

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion':[0, 0],
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = True
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        # a 5min timer
        self.count_down = 300
        self.start_time = time.time()
        self.control_clock = time.time()
        # initialise images
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        self.detector_output = np.zeros([240,320], dtype=np.uint8)
        if args.ckpt == "":
            self.detector = None
            self.network_vis = cv2.imread('pics/8bit/detector_splash.png')
        else:
            self.detector = Detector(args.ckpt, use_gpu=False)
            self.network_vis = np.ones((240, 320,3))* 100
        # self.bg = pygame.image.load('pics/gui_mask.jpg')

        # ----
        self.robot_pose = [0., 0., 0.]

    # wheel control
    def control(self):
        if args.play_data:
            lv, rv = self.pibot.set_velocity()
        else:
            lv, rv = self.pibot.set_velocity(
                self.command['motion'])
        if not self.data is None:
            self.data.write_keyboard(lv, rv)
        dt = time.time() - self.control_clock
        drive_meas = measure.Drive(lv, rv, dt)
        self.control_clock = time.time()
        return drive_meas
    # camera control
    def take_pic(self):
        self.img = self.pibot.get_image()
        if not self.data is None:
            self.data.write_image(self.img)

    # SLAM with ARUCO markers
    def update_slam(self, drive_meas):
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        # if self.request_recover_robot:
        is_success = self.ekf.recover_from_pause(lms)
        if is_success:
            print('XXXXXXXXXXXXXXXXXXXxx')
        #     if is_success:
        #         self.notification = 'Robot pose is successfuly recovered'
        #         self.ekf_on = True
        #     else:
        #         self.notification = 'Recover failed, need >2 landmarks!'
        #         self.ekf_on = False
        #     self.request_recover_robot = False
        # elif self.ekf_on: # and not self.debug_flag:
        self.robot_pose = self.ekf.predict(drive_meas) 
        self.ekf.add_landmarks(lms)
        if lms:
            self.robot_pose = self.ekf.update(lms)
            state = self.robot_pose
            print('xxxxxxxxxxxx')
            time.sleep(1)
        self.robot_pose[-1] += 0
        print('state: update_slam', self.robot_pose)
        return self.robot_pose

    # using computer vision to detect targets
    def detect_target(self):
        if self.command['inference'] and self.detector is not None:
            self.detector_output, self.network_vis = self.detector.detect_single_image(self.img)
            self.command['inference'] = False
            self.file_output = (self.detector_output, self.ekf)
            self.notification = f'{len(np.unique(self.detector_output))-1} target type(s) detected'

    # save raw images taken by the camera
    def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        if self.command['save_image']:
            image = self.pibot.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'

    # wheel and camera calibration for SLAM
    def init_ekf(self, datadir, ip):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            scale /= 2
        fileB = "{}baseline.txt".format(datadir)
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)

    # save SLAM map
    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            self.notification = 'Map is saved'
            self.command['output'] = False
        # save inference with the matching robot pose and detector labels
        if self.command['save_inference']:
            if self.file_output is not None:
                #image = cv2.cvtColor(self.file_output[0], cv2.COLOR_RGB2BGR)
                self.pred_fname = self.output.write_image(self.file_output[0],
                                                          self.file_output[1])
                self.notification = f'Prediction is saved to {operate.pred_fname}'
            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_inference'] = False

    # keyboard teleoperation
    def update_keyboard(self):
        for event in pygame.event.get():
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['motion'][0] = min(self.command['motion'][0]+1, 1)
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['motion'][0] = max(self.command['motion'][0]-1, -1)
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['motion'][1] = min(self.command['motion'][1]+1, 1)
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['motion'][1] = max(self.command['motion'][1]-1, -1)
            # stop
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
            # save image
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['output'] = True
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm +=1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()
            # run SLAM
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_observed_markers = len(self.ekf.taglist)
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True
                    else:
                        self.notification = '> 2 landmarks is required for pausing'
                elif n_observed_markers < 3:
                    self.notification = '> 2 landmarks is required for pausing'
                else:
                    if not self.ekf_on:
                        self.request_recover_robot = True
                    self.ekf_on = not self.ekf_on
                    if self.ekf_on:
                        self.notification = 'SLAM is running'
                    else:
                        self.notification = 'SLAM is paused'
            # run object detector
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['inference'] = True
            # save object detection outputs
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_inference'] = True
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
        if self.quit:
            pygame.quit()
            sys.exit()

    def get_distance_robot_to_goal(self):
        '''
	Compute Euclidean distance between the robot and the goal location
	:param robot_state: 3D vector (x, y, theta) representing the current state of the robot
	:param goal: 3D Cartesian coordinates of goal location
        '''
        if np.array(self.waypoint).shape[0] < 3:
            goal = np.hstack((self.waypoint, np.array([0])))
        else:
            goal = self.waypoint

        x_goal, y_goal, _ = goal
        x, y, theta = self.robot_pose
        x_diff = x_goal - x
        y_diff = y_goal - y

        rho = np.hypot(x_diff, y_diff)

        return rho

    def clamp_angle(self, rad_angle=0, min_value=-np.pi, max_value=np.pi):
        """
        Restrict angle to the range [min, max]
        :param rad_angle: angle in radians
        :param min_value: min angle value
        :param max_value: max angle value
        """
        if min_value > 0:
            min_value *= -1
        angle = (rad_angle + max_value) % (2 * np.pi) + min_value
        return angle



    def get_angle_robot_to_goal(self):
        """
        Compute angle to the goal relative to the heading of the robot.
        Angle is restricted to the [-pi, pi] interval
        :param robot_state: 3D vector (x, y, theta) representing the current state of the robot
        :param goal: 3D Cartesian coordinates of goal location
        """
        if np.array(self.waypoint).shape[0] < 3:
            goal = np.hstack((self.waypoint, np.array([0])))
        else:
            goal = self.waypoint
        x_goal, y_goal, _ = goal
        x, y, theta = self.robot_pose
        x_diff = x_goal - x
        y_diff = y_goal - y
        alpha = self.clamp_angle(theta - np.arctan2(y_diff, x_diff))
        print('th', theta, np.arctan2(y_diff, x_diff))

        return alpha

    def drive_to_point(self):
        distance_to_goal = self.get_distance_robot_to_goal()
        angle_to_waypoint = self.get_angle_robot_to_goal()

        print('ang/dist', angle_to_waypoint, distance_to_goal)
        while distance_to_goal > 0.18:  # ensure it is within 30cm from goal
            time.sleep(0.05)
            print('pose', self.robot_pose)
            # keep travelling to goal
            print('ang/dist', angle_to_waypoint, distance_to_goal)
            if abs(angle_to_waypoint) > np.pi / 45:  # if the angle to the waypoint is above a threshold, turn
                # Turn
                if angle_to_waypoint < 0:  # turn left
                    self.command['motion'] = [0, 1]
                else:  # turn right
                    self.command['motion'] = [0, -1]
            else:
                # Drive straight
                self.command['motion'] = [1, 0]

            drive_meas = self.control()
            state = self.update_slam(drive_meas)
            print('goal', self.waypoint)
            distance_to_goal = self.get_distance_robot_to_goal()
            angle_to_waypoint = self.get_angle_robot_to_goal()
        # stop driving
        self.command['motion'] = [0, 0]
        drive_meas = self.control()
        self.update_slam(drive_meas)
        print("Arrived at [{}, {}]".format(self.waypoint[0], self.waypoint[1]))


    def turnaround(self):
        angle_to_waypoint = self.get_angle_robot_to_goal()
        while (abs(angle_to_waypoint) > np.pi / 90):  # if the angle to the waypoint is above a threshold, turn
            # Turn
            if angle_to_waypoint > 0:  # turn left
                self.command['motion'] = [0, -1]
            if angle_to_waypoint < 0:  # turn left
                self.command['motion'] = [0, 1]

            #self.draw(canvas)
            drive_meas = self.control()
            self.update_slam(drive_meas)

            #pygame.display.update()

            angle_to_waypoint = self.get_angle_robot_to_goal()

        # stop driving
        self.command['motion'] = [0, 0]
        drive_meas = self.control()
        self.update_slam(drive_meas)
        print("Arrived at [{}, {}]".format(self.waypoint[0], self.waypoint[1]))



class Circle:
    def __init__(self, c_x, c_y, radius=0.1):
        self.center = np.array([c_x, c_y])
        self.radius = radius

    def is_in_collision_with_points(self, points):
        dist = []
        for point in points:
            dx = self.center[0] - point[0]
            dy = self.center[1] - point[1]

            dist.append(dx * dx + dy * dy)
        if np.min(dist) <= self.radius ** 2:
            return True
        return False


def read_True_map(fname):
    """read the ground truth map and output the pose of the aruco markers and 3 types of target fruit to search

    @param fname: filename of the map
    @return:
        1) list of target fruits, e.g. ['apple', 'pear', 'lemon']
        2) locations of the target fruits, [[x1, y1], ..... [xn, yn]]
        3) locations of aruco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
        fruit_list = []
        fruit_True_pos = []
        aruco_True_pos = np.empty([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)  # reading every x coordinates
            y = np.round(gt_dict[key]['y'], 1)  # reading every y coordinates

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_True_pos[0][0] = x
                    aruco_True_pos[0][1] = y
                else:
                    marker_id = int(key[5])  # giving id to aruco markers
                    aruco_True_pos[marker_id][0] = x
                    aruco_True_pos[marker_id][1] = y
            else:
                fruit_list.append(key[:-2])
                if len(fruit_True_pos) == 0:
                    fruit_True_pos = np.array([[x, y]])
                else:
                    fruit_True_pos = np.append(fruit_True_pos, [[x, y]], axis=0)

        return fruit_list, fruit_True_pos, aruco_True_pos


def read_search_list():
    """read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open('search_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_True_pos):
    """print out the target fruits' pos in the search order

    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_True_pos: positions of the target fruits
    """

    print("search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(3):
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_True_pos[i][0], 1),
                                                  np.round(fruit_True_pos[i][1], 1)))
        n_fruit += 1


def get_robot_pose(kalman):
    ####################################################
    # todo: replace with your codes to estimate the phose of the robot
    # we strongly recommend you to use your slam code from m2 here

    # update the robot pose [x,y,theta]
    # robot_pose = [0.0,0.0,0.0] # replace with your calculation
    robot_pose = kalman.robot.state.flatten()  # kalman.get_state_vector()# replace with your calculation
    ####################################################

    return robot_pose



#####################################################################################################

class item_in_map:
    def __init__(self, name, measurement):
        self.tag = name
        self.coordinates = np.array([[measurement[0]], [measurement[1]]])

def pose_to_pixel(pose,map_dimension,map_resolution):
    morigin = map_dimension / 2.0
    # pose maps from - map_dimension : map_dimension
    map = lambda old_value, old_min, old_max, new_min, new_max: ((old_value - old_min) / (old_max - old_min)) * (
                new_max - new_min) + new_min
    pixel_x = map(pose[0], morigin, -morigin,
                  1, (map_dimension / map_resolution))
    pixel_y = map(pose[1], morigin, -morigin,
                  1, (map_dimension / map_resolution))
    return int(pixel_x), int(pixel_y)

def pixel_to_pose(pixel,map_dimension,map_resolution):
    morigin = map_dimension / 2.0
    map = lambda old_value, old_min, old_max, new_max, new_min: ((old_value - old_min) / (old_max - old_min)) * (
                new_max - new_min) + new_min
    pose_x = map(pixel[0], 1, (map_dimension / map_resolution), - morigin,
                 morigin)
    pose_y = map(pixel[1], 1, (map_dimension / map_resolution), -morigin,
                 morigin)
    return pose_x, pose_y

# main loop
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--ckpt", default='network/scripts/model/model.best.pth')
    parser.add_argument("--map", type=str, default='M4_true_map_3fruits.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=40000)
    args, _ = parser.parse_known_args()


    pygame.font.init()
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)

    # width, height = 700, 660
    # canvas = pygame.display.set_mode((width, height))
    # pygame.display.set_caption('ECE4078 2021 Lab')
    # pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    # canvas.fill((0, 0, 0))
    # splash = pygame.image.load('pics/loading.png')
    # pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
    #                  pygame.image.load('pics/8bit/pibot2.png'),
    #                  pygame.image.load('pics/8bit/pibot3.png'),
    #                  pygame.image.load('pics/8bit/pibot4.png'),
    #                  pygame.image.load('pics/8bit/pibot5.png')]
    # pygame.display.update()

    start = False

    # counter = 40
    # while not start:
    #     for event in pygame.event.get():
    #         if event.type == pygame.KEYDOWN:
    #             start = True
    #     canvas.blit(splash, (0, 0))
    #     x_ = min(counter, 600)
    #     if x_ < 600:
    #         canvas.blit(pibot_animate[counter % 10 // 2], (x_, 565))
    #         pygame.display.update()
    #         counter += 2

    operate = Operate(args)
    ppi = PenguinPi(args.ip, args.port)
    operate.ekf.reset()
    operate.robot_pose = [[0],[0],[0]]
    ############################################################################################
    ########################################################################################

    # read in the True map
    fruits_list, fruits_True_pos, aruco_True_pos = read_True_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_True_pos)

    aruco_list = []
    fruits = []
    f_id = {'apple': 0,
            'lemon': 1,
            'orange': 2,
            'pear': 3,
            'strawberry': 4}
    f_dict = {}
    for i in range(len(fruits_True_pos)):
        item = item_in_map(f_id[fruits_list[i]], fruits_True_pos[i])
        f_dict[fruits_list[i]] = fruits_True_pos[i]
        fruits.append(item)
    print('fruits', fruits[0].coordinates, fruits[0].tag)
    operate.ekf.add_landmarks(fruits)  # will add known
    for i in range(len(aruco_True_pos)):
        print(str(i), aruco_True_pos[i])
        item = item_in_map(str(i), aruco_True_pos[i])
        aruco_list.append(item)
    #checking
    # aruco_list[0].tag = str(10)
    operate.ekf.add_landmarks(aruco_list)  # will add known

    # The following code is only a skeleton code the semi-auto fruit searching task
    # implement RRT, loop is for the number of fruits

    goal = [[f_dict[i][0], f_dict[i][1]] for i in search_list ]
    obstacles_aruco = []
    lms_xy = operate.ekf.markers[:2, :]
    for i in range(len(operate.ekf.markers[0, :])):
        obstacles_aruco.append(Circle(lms_xy[0, i], lms_xy[1, i]))
    expand_dis = 0.4
    print("obstacles:", obstacles_aruco)
    print(start)
    start = list(operate.ekf.robot.state[0:2, :])

    # --- occupancy grid
    map_resolution = 0.1 # metres / pixel
    map_dimension = 1.4 * 3 # metres
    map_size = int(map_dimension / map_resolution)
    map_arr = np.ones((map_size, map_size)) # shape = 28*28
    def pad(map, item_list, full_pad=True):
        if full_pad:
            pad = 3
        else:
            pad = 1
        for item in item_list:
            try:
                map[item[0] - pad: item[0] + pad, item[1] - pad: item[1] + pad] = np.inf
            except:
                pass
        return map

    def str_pull(route):
        if len(route) > 2:
            new_path = []#[route[0]]
            deriv = lambda p0, p1: (p1[1] - p0[1])/(p1[0] - p0[0] + 1e-16)
            dt_old = deriv(route[1], route[0])
            for i in range(2, len(route)):
                dt = deriv(route[i], route[i-1])
                if dt == dt_old:
                    pass
                else:
                    new_path.append(route[i])
                dt_old = dt
            return new_path
        else:
            return route

    x_start,y_start = pose_to_pixel(start,map_dimension,map_resolution)
    start_map_frame = [x_start,y_start]
    goal_map_frame = []
    for g in goal:
        x_goal,y_goal = pose_to_pixel(g,map_dimension,map_resolution)
        goal_map_frame.append([x_goal,y_goal])

    # adding obstacles and heuristic into the map
    obstacles_map_frame = []
    for item in list(aruco_True_pos):
        x_obs, y_obs = pose_to_pixel(item, map_dimension, map_resolution)
        obstacles_map_frame.append([x_obs, y_obs])
    map_arr = pad(map_arr, obstacles_map_frame, full_pad=True)
    obstacles_map_frame = []
    for item in list(fruits_True_pos):
        x_obs, y_obs = pose_to_pixel(item, map_dimension, map_resolution)
        obstacles_map_frame.append([x_obs, y_obs])
    map_arr = pad(map_arr, obstacles_map_frame, full_pad=False)



    operate.robot_pose = [[0],[0],[0]]

    # debug start and end
    map_arr = np.array(map_arr, dtype=np.float32)
    for g in goal_map_frame:
        map_arr[start_map_frame[0] - 1: start_map_frame[0] +1, start_map_frame[1] - 1 : start_map_frame[1] - 1] = 1
        map_arr[map_arr==1] = 5
        path = pyastar2d.astar_path(map_arr, start_map_frame, g, allow_diagonal=True)
        print('map', map_arr.shape, map_dimension, map_resolution)
        print('start/goal val', map_arr[start_map_frame[0], start_map_frame[1]], map_arr[goal_map_frame[0], goal_map_frame[1]])
        print('obstacles', obstacles_map_frame)
        new_goal = g
        new_start = start_map_frame
        while type(path) == type(None):
            new_goal = [i - 1 if i > int(map_size/2) else i + 1 for i in new_goal]
            new_start = [i - 1 if i > int(map_size/2) else i + 1 for i in new_start]
            map_arr[new_start[0] - 1: new_start[0] +1, new_start[1] - 1 : new_start[1] - 1] = 1
            map_arr[new_goal[0] - 1: new_goal[0] +1, new_goal[1] - 1 : new_goal[1] - 1] = 1
            path = pyastar2d.astar_path(map_arr, new_start, new_goal, allow_diagonal=True)
            print('path failed', new_goal, goal_map_frame)
            time.sleep(0.5)


        route = [[float(i[0]), float(i[1])] for i in path]
        print('pre_route\n', len(route))
        # route = str_pull(route)
        # print('post_route\n', len(route))

        path_pose = []
        map_arr[map_arr==np.inf] = 255
        map_arr[map_arr==1] = 1
        # import scipy.misc
        from PIL import Image
        # attach path to map
        for item in route:
            map_arr[int(item[0]), int(item[1])] = 128
        #im_array = np.array([map_arr, map_arr, map_arr]).T
        #print(im_array.shape)
        im = Image.fromarray(map_arr)
        im = im.convert('RGB')

        im.save("your_file.png")
        #scipy.misc.imsave('outfile.jpg', map_arr)

        # converting from map frame to pose
        for item in route:
            x_obs, y_obs = pixel_to_pose(item, map_dimension, map_resolution)
            path_pose.append([x_obs, y_obs])


        # for g in goal:
        # map: wall = np.inf, empty space = 1.
        # String pulling
        # def str_pull: identifies vertecies in lines, and removes redundant waypoints using gradients

        

        #route = route[::-1]

        _ = input("press enter to start moving:... \nstart -- {},\nend -- {},\nroute -- {}".format(
            (pixel_to_pose(new_start, map_dimension, map_resolution), start), (pixel_to_pose(new_goal, map_dimension, map_resolution), goal), path_pose))
        for point in path_pose:
            operate.waypoint = [point[0], point[1]]
            operate.drive_to_point()

        x_start,y_start = pose_to_pixel(operate.robot_pose,map_dimension,map_resolution)
        start_map_frame = [x_start,y_start]

