# m5 - autonomous fruit searching
# basic python packages
import sys, os
# import Copy
import cv2
import numpy as np
import json
import argparse
import time
import random
import math
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
from util.pibot import PenguinPi  # access the robot
import util.DatasetHandler as dh  # save/load functions
import util.measure as measure  # measurements
import pygame  # python package for GUI
import shutil  # python package for file operations
from TargetPoseEst import *

# import SLAM components you developed in M2
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf_old import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco
from slam.mapping_utils import *

# import CV components
sys.path.insert(0, "{}/network/".format(os.getcwd()))
sys.path.insert(0, "{}/network/scripts".format(os.getcwd()))
from network.scripts.detector import Detector


class Operate:
    def __init__(self, args):
        self.args = args
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
            self.ekf.robot, marker_length=0.07)  # size of the ARUCO markers


        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.choiceSlam = int(input("Do you want to load map?: "))
        if self.choiceSlam:
            self.output = dh.OutputWriter('lab_output', 'a')
            self.ekf = self.output.read_map(self.ekf)
        else:
            self.output = dh.OutputWriter('lab_output', 'w')
        self.command = {'motion': [0, 0],
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = True
        self.file_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        # a 5min timer
        self.count_down = 300
        self.start_time = time.time()
        self.control_clock = time.time()
        # initialise images
        self.img = np.zeros([240, 320, 3], dtype=np.uint8)
        self.aruco_img = np.zeros([240, 320, 3], dtype=np.uint8)
        self.detector_output = np.zeros([240, 320], dtype=np.uint8)
        if args.ckpt == "":
            self.detector = None
            self.network_vis = cv2.imread('pics/8bit/detector_splash.png')
        else:
            self.detector = Detector(args.ckpt, use_gpu=False)
            self.network_vis = np.ones((240, 320,3))* 100
        self.bg = pygame.image.load('pics/gui_mask.jpg')

        # ----
        self.robot_pose = [0., 0., 0.]

        # slam map load

    # wheel control
    def control(self):
        tick_multi = 2 if self.args.world == 'sim' else 1
        tick_turn = 5 * tick_multi
        tick_drive = 20 * tick_multi
        lv, rv = self.pibot.set_velocity(
            self.command['motion'], tick=tick_drive, turning_tick=tick_turn)
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
        self.take_pic()
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        is_success = False
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(lms)
        if is_success:
            self.robot_pose = self.ekf.robot.state[:3]
        else:
            pose_predict = self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            pose_update = self.ekf.update(lms)
            if lms:
                self.robot_pose = pose_update
            else:
                self.robot_pose = pose_predict
        # print('state: update_slam', self.robot_pose, lms)
        return self.robot_pose

    # using computer vision to detect targets
    def detect_target(self):
        if self.command['inference'] and self.detector is not None:
            im = self.img.copy()
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            self.detector_output, self.network_vis = self.detector.detect_single_image(self.img)
            self.command['inference'] = False
            self.file_output = (im, self.ekf)
            self.notification = f'{len(self.detector_output.index)} target type(s) detected'

    # save raw images taken by the camera
    def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        if self.command['save_image']:
            image = self.pibot.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
            self.command['save_image'] = False
            self.notification = f'{self.image_id} images out of 7?'
            self.image_id += 1

    # wheel and camera calibration for SLAM
    def init_ekf(self, datadir, ip):
        if self.args.world == 'sim':
            fileK = "{}intrinsic.txt".format(datadir)
        else:
            fileK = "{}intrinsic_pibot.txt".format(datadir)

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
                # image = cv2.cvtColor(self.file_output[0], cv2.COLOR_RGB2BGR)
                self.pred_fname = self.output.write_image(self.file_output[0],
                                                          self.file_output[1])
                self.notification = f'{self.output.image_count} images out of 7?'
            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_inference'] = False

    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption,
                                            False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1] - 25))

    # get stuff from targetpost Est
    def getTargetPose(self):
        fileK = "{}intrinsic.txt".format('./calibration/param/')
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        base_dir = Path('./')
        search_list = read_search_list(self.args.searchList)

        # a dictionary of all the saved detector outputs
        image_poses = {}
        with open(base_dir / 'lab_output/images.txt') as fp:
            for line in fp.readlines():
                pose_dict = ast.literal_eval(line)
                image_poses[pose_dict['imgfname']] = pose_dict['pose']
        print('image poses', image_poses)
        # estimate pose of targets in each detector output
        target_map = {}
        for file_path in image_poses.keys():
            print(file_path)
            completed_img_dict = get_image_info(base_dir, file_path, image_poses[file_path])
            target_map[file_path] = estimate_pose(base_dir, camera_matrix, completed_img_dict)
        # merge the estimations of the targets so that there are at most 3 estimations of each target type
        target_est = merge_estimations(target_map, search_list)
        # save target pose estimations
        with open(base_dir / 'lab_output/targets.txt', 'w') as fo:
            print('operate target_est\n', target_est)
            json.dump(target_est, fo)

        print('Estimations saved!')
        ################################################################################

    # keyboard teleoperation
    def update_keyboard(self):
        for event in pygame.event.get():
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['motion'][0] = min(self.command['motion'][0] + 1, 1)
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['motion'][0] = max(self.command['motion'][0] - 1, -1)
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['motion'][1] = min(self.command['motion'][1] + 1, 1)
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['motion'][1] = max(self.command['motion'][1] - 1, -1)
            # stop
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
            # save image
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True

            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['output'] = True
                # print('Maerkd--', self.ekf.markers)
                # print('#####################')
                # print('tags--', self.ekf.taglist)
                # print('###########################')
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm += 1
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
            try:
                x = int(input('1: Quit and Save, 2: Quit and dont Save'))
            except:
                print('Retry')
                x = 0
            if x == 1:
                self.output.write_map(self.ekf)
                self.getTargetPose()
                # run auto_fruit_search_0
                # pygame.quit()
            elif x == 2:
                pass
            else:
                self.quit = False


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
        # print('th', theta, np.arctan2(y_diff, x_diff))

        return alpha

    def drive_to_point(self):
        self.args.world = 'real'
        self.request_recover_robot = True
        distance_to_goal = self.get_distance_robot_to_goal()
        angle_to_waypoint = self.get_angle_robot_to_goal()

        print('ang/dist', angle_to_waypoint, distance_to_goal)
        while distance_to_goal > 0.18:  # ensure it is within 30cm from goal
            time.sleep(0.05)
            print('pose', self.robot_pose, self.waypoint)
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

    # paint the GUI
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(320, 480 + v_pad),
                                            not_pause=self.ekf_on)
        canvas.blit(ekf_view, (2 * h_pad + 320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view,
                                position=(h_pad, v_pad)
                                )

        # for target detector (M3)
        detector_view = cv2.resize(self.network_vis,
                                   (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view,
                                position=(h_pad, 240 + 2 * v_pad)
                                )

        # canvas.blit(self.gui_mask, (0, 0))
        self.put_caption(canvas, caption='SLAM', position=(2 * h_pad + 320, v_pad))
        self.put_caption(canvas, caption='Detector',
                         position=(h_pad, 240 + 2 * v_pad))
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))

        notifiation = TEXT_FONT.render(self.notification,
                                       False, text_colour)
        canvas.blit(notifiation, (h_pad + 10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain) % 2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2 * h_pad + 320 + 5, 530))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)

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


def read_True_map(fnameArcuo, fnameFruit):
    """read the ground truth map and output the pose of the aruco markers and 3 types of target fruit to search

    @param fname: filename of the map
    @return:
        1) list of target fruits, e.g. ['apple', 'pear', 'lemon']
        2) locations of the target fruits, [[x1, y1], ..... [xn, yn]]
        3) locations of aruco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    f_aruco = open(fnameArcuo, 'r')
    gt_dict = json.load(f_aruco)
    f_fruit = open(fnameFruit, 'r')
    fr_dict = json.load(f_fruit)
    gt_dict.update(fr_dict)
    print('loaded both dictionaries', gt_dict)
    fruit_list = []
    fruit_True_pos = []
    aruco_True_pos = np.empty([11, 2])
    aruco_True_pos[0] = [50, 50]
    # remove unique id of targets of the same type
    for key in gt_dict:
        x = np.round(gt_dict[key]['x'], 1)  # reading every x coordinates
        y = np.round(gt_dict[key]['y'], 1)  # reading every y coordinates

        if key.startswith('aruco'):
            if key.startswith('aruco10'):
                aruco_True_pos[10][0] = x
                aruco_True_pos[10][1] = y
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


def read_search_list(fname):
    """read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open(fname, 'r') as fd:
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
        for i in range(len(fruit_list)):
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


def pose_to_pixel(pose, map_dimension, map_resolution):
    morigin = map_dimension / 2.0
    # pose maps from - map_dimension : map_dimension
    map = lambda old_value, old_min, old_max, new_min, new_max: ((old_value - old_min) / (old_max - old_min)) * (
            new_max - new_min) + new_min
    pixel_x = map(pose[0], morigin, -morigin,
                  1, (map_dimension / map_resolution))
    pixel_y = map(pose[1], morigin, -morigin,
                  1, (map_dimension / map_resolution))
    return int(pixel_x), int(pixel_y)


def pixel_to_pose(pixel, map_dimension, map_resolution):
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
    parser.add_argument("--ckpt", default='/home/ece4078/412_ECE4078/Final_demo/network/scripts/model/best_m.pt')
    # parser.add_argument("--map", type=str, default='M4_3fruits_lab4.txt')
    parser.add_argument("--map", type=str, default='aruco_true.txt')
    parser.add_argument("--world", type=str, default='sim') # ------------- CHANGE to 'real'
    parser.add_argument("--mapFruit", type=str, default='./lab_output/targets.txt')
    parser.add_argument("--searchList", type=str, default='./M5_lab4_sim_search_list.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=40000)
    args, _ = parser.parse_known_args()

    pygame.font.init()
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)

    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2021 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                     pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()

    start = False

    # UNCOMMENT FOR M5
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
    # UNCOMMENT FOR M5
    while not operate.quit:
        operate.update_keyboard()
        operate.take_pic()
        drive_meas = operate.control()
        operate.update_slam(drive_meas)
        operate.record_data()
        operate.save_image()
        operate.detect_target()
        # visualise
        operate.draw(canvas)
        pygame.display.update()

    # operate.robot_pose = [[0], [0], [0]]
    ############################################################################################
    ########################################################################################

    # read in the True map
    # fruits_list, fruits_True_pos, aruco_True_pos = read_True_map(args.map)
    fruits_list, fruits_True_pos, aruco_True_pos = read_True_map(args.map, args.mapFruit)
    search_list = read_search_list(args.searchList)
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
        if fruits_list[i] not in f_dict:
            f_dict[fruits_list[i]] = [fruits_True_pos[i]]
        else:
            f_dict[fruits_list[i]].append(fruits_True_pos[i])
        fruits.append(item)
    print('fruits', fruits[0].coordinates, fruits[0].tag)
    # operate.ekf.add_landmarks(fruits)  # will add known
    for i in range(len(aruco_True_pos)):
        print('list aruco', str(i), aruco_True_pos[i])
        item = item_in_map(str(i), aruco_True_pos[i])
        aruco_list.append(item)
    # checking
    # aruco_list[0].tag = str(10)
    # operate.ekf.add_landmarks(aruco_list)  # will add known

    # The following code is only a skeleton code the semi-auto fruit searching task
    # implement RRT, loop is for the number of fruits

    goal = [[f_dict[i][0][0], f_dict[i][0][1]] for i in search_list]
    print('\n\n GOAL ----------------\n', goal, '\n\n')

    obstacles_aruco = []
    lms_xy = operate.ekf.markers[:2, :]
    for i in range(len(operate.ekf.markers[0, :])):
        obstacles_aruco.append(Circle(lms_xy[0, i], lms_xy[1, i]))
    expand_dis = 0.4
    start = list(operate.ekf.robot.state[0:2, :])

    # --- occupancy grid
    map_resolution = 0.1  # metres / pixel
    map_dimension = 1.4 * 3  # metres
    map_size = int(map_dimension / map_resolution)
    map_arr = np.zeros((map_size, map_size))  # shape = 28*28


    def pad_map(input_arr, obstacle_list, pad_max=np.inf, offset=3):
        for coord in obstacle_list:
            try:
                input_arr[coord[0], coord[1]] = pad_max
            except:
                pass
        input_arr[:, 0] = pad_max
        input_arr[:, input_arr.shape[0]-1] = pad_max
        input_arr[0, :] = pad_max
        input_arr[input_arr.shape[0]-1, :] = pad_max
        # input_arr should be an array of [0,1] only
        for rows in range(input_arr.shape[0]):
            for cols in range(input_arr.shape[1]):
                if input_arr[rows,cols] != pad_max:
                    # Build a list of all the euclidean distances and find the shortest one
                    euc_list = []
                    for coord in obstacle_list:
                        euc = 2 / (0.1 * np.sqrt(np.sum((np.array(coord) - np.array([rows,cols])) ** 2)) **0.9)
                        euc_list.append(euc)
                    input_arr[rows,cols] = max(euc_list)
        return np.array(input_arr, dtype=np.float32)


    def str_pull(route):
        if len(route) > 2:
            new_path = []  # [route[0]]
            deriv = lambda p0, p1: (p1[1] - p0[1]) / (p1[0] - p0[0] + 1e-16)
            dt_old = deriv(route[1], route[0])
            for i in range(2, len(route)):
                dt = deriv(route[i], route[i - 1])
                if dt == dt_old:
                    pass
                else:
                    new_path.append(route[i])
                dt_old = dt
            return new_path
        else:
            return route


    x_start, y_start = pose_to_pixel(start, map_dimension, map_resolution)
    start_map_frame = [x_start, y_start]
    goal_map_frame = []
    for g in goal:
        x_goal, y_goal = pose_to_pixel(g, map_dimension, map_resolution)
        goal_map_frame.append([x_goal, y_goal])

    # adding obstacles and heuristic into the map
    obstacles_map_frame = []
    for item in list(aruco_True_pos):
        x_obs, y_obs = pose_to_pixel(item, map_dimension, map_resolution)
        obstacles_map_frame.append([y_obs, x_obs])
    for item in list(fruits_True_pos):
        x_obs, y_obs = pose_to_pixel(item, map_dimension, map_resolution)
        obstacles_map_frame.append([x_obs, y_obs])
    map_arr = pad_map(map_arr, obstacles_map_frame, offset=42/2)
    plt.imshow(map_arr)
    plt.colorbar()
    plt.title(f'max: {np.max(map_arr)}, min: {np.min(map_arr)}')
    plt.savefig('map_og.png')
    plt.close()

    null = pose_to_pixel([0., 0.], map_dimension, map_resolution)
    for g in goal_map_frame:
        map_arr[start_map_frame[0], start_map_frame[1]] = 1
        map_arr[g[0], g[1]] = 1
        # map_arr[map_arr == 1] = 5
        path = pyastar2d.astar_path(map_arr, start_map_frame, g, allow_diagonal=True)
        print('map', map_arr.shape, map_dimension, map_resolution)
        print('start/goal val', map_arr[start_map_frame[0], start_map_frame[1]],
              map_arr[goal_map_frame[0], goal_map_frame[1]])
        print('obstacles', obstacles_map_frame)
        new_goal = g
        new_start = start_map_frame
        while type(path) == type(None):
            new_goal = [i - 1 if i > int(map_size / 2) else i + 1 for i in new_goal]
            new_start = [i - 1 if i > int(map_size / 2) else i + 1 for i in new_start]
            map_arr[new_start[0] - 1: new_start[0] + 1, new_start[1] - 1: new_start[1] - 1] = 1
            map_arr[new_goal[0] - 1: new_goal[0] + 1, new_goal[1] - 1: new_goal[1] - 1] = 1
            path = pyastar2d.astar_path(map_arr, new_start, new_goal, allow_diagonal=True)
            print('path failed', new_goal, goal_map_frame)
            time.sleep(0.5)

        route = [[float(i[0]), float(i[1])] for i in path]
        viz_map = np.copy(map_arr)
        print(viz_map.shape)
        path_pose = []
        viz_map[viz_map == np.inf] = 50
        print(f'\n---------\n{(np.max(viz_map), np.min(viz_map))}')
        # viz_map = np.interp(viz_map, (np.max(viz_map), np.min(viz_map)), (0, 255))

        # attach path to map
        for item in route:
            viz_map[int(item[0]), int(item[1])] = 50
        from PIL import Image
        im = Image.fromarray(viz_map)
        im = im.convert('RGB')
        im.save("your_file.png")
        plt.imshow(viz_map)#np.rot90(viz_map, k=-1))
        plt.colorbar()
        plt.title(f'max: {np.max(viz_map)}, min: {np.min(viz_map)}')
        plt.savefig('map.png')

        # converting from map frame to pose
        for item in route:
            x_obs, y_obs = pixel_to_pose(item, map_dimension, map_resolution)
            path_pose.append([x_obs, y_obs])
        print('Path: ', route)
        time.sleep(2)
        for point in path_pose:
            operate.waypoint = [point[0], point[1]]
            operate.drive_to_point()
        print('arrive at: ', operate.robot_pose)
        time.sleep(3)
        x_start, y_start = pose_to_pixel(operate.robot_pose, map_dimension, map_resolution)
        start_map_frame = [x_start, y_start]

