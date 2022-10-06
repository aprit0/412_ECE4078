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

# import slam components
# sys.path.insert(0, "{}/slam".format(os.getcwd()))
# from slam.ekf import EKF
# from slam.robot import Robot
# import slam.aruco_detector as aruco

# import CV components
sys.path.insert(0,"{}/network/".format(os.getcwd()))
sys.path.insert(0,"{}/network/scripts".format(os.getcwd()))
from network.scripts.detector import Detector
# import utility functions
# sys.path.insert(0, "{}/util".format(os.getcwd()))
# from pibot import PenguinPi
import measure as measure


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


class RRTC:
    """
    class for rrt planning
    """

    class node:
        """
        rrt node
        """

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    def __init__(self, start=np.zeros(2), goal=np.array([120, 90]), obstacle_list=None, width=160, height=100,
                 expand_dis=3.0, path_resolution=0.1, max_points=200):
        """
        setting parameter
        start:start position [x,y]
        goal:goal position [x,y]
        obstacle_list: list of obstacle objects
        width, height: search area
        expand_dis: min distance between random node and closest node in rrt to it
        path_resolution: step size to considered when looking for node to expand
        """
        self.start = self.node(start[0], start[1])
        self.end = self.node(goal[0], goal[1])
        self.width = width
        self.height = height
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.max_nodes = max_points
        self.obstacle_list = obstacle_list
        self.start_node_list = []  # tree from start
        self.end_node_list = []  # tree from end

    def planning(self):
        """
        rrt path planning
        """
        self.start_node_list = [self.start]
        self.end_node_list = [self.end]
        while len(self.start_node_list) + len(self.end_node_list) <= self.max_nodes:

            # print('start',len(self.start_node_list))
            # print(len(self.end_node_list))
            # todo: complete the planning method ----------------------------------------------------------------
            # 1. sample and add a node in the start tree
            rand_sample = self.get_random_node()
            expansion_id = self.get_nearest_node_index(self.start_node_list, rand_sample)
            expansion_node = self.start_node_list[expansion_id]

            # 2. check whether trees can be connected, check if the sampled node is less than expand_dis
            # check for colission but 'expand using steer'to set up the path then add after checking collision status
            nearby_node = self.steer(expansion_node, rand_sample, self.expand_dis)

            if self.is_collision_free(nearby_node):
                self.start_node_list.append(nearby_node)

                # check if there is a node lesser than expand_dis in end_list
                end_expand_id = self.get_nearest_node_index(self.end_node_list, nearby_node)
                end_expand_node = self.end_node_list[end_expand_id]

                check_dist, _ = self.calc_distance_and_angle(end_expand_node, nearby_node)

                # 3. add the node that connects the trees and generate the path
                if check_dist < self.expand_dis:  # True path is found
                    final_path = self.steer(end_expand_node, nearby_node, self.expand_dis)
                    if self.is_collision_free(final_path):
                        # note: it is important that you return path found as:
                        self.end_node_list.append(final_path)
                        return self.generate_final_course(len(self.start_node_list) - 1, len(self.end_node_list) - 1)
                        # 4. sample and add a node in the end tree
                rand_sample_2 = self.get_random_node()
                expansion_2 = self.get_nearest_node_index(self.end_node_list, rand_sample_2)
                expansion_node_2 = self.end_node_list[expansion_2]

                # check whether trees can be connected
                # check for colission but 'expand using steer'to set up the path then add after checking collision status
                nearby_node_2 = self.steer(expansion_node_2, rand_sample_2, self.expand_dis)

                if self.is_collision_free(nearby_node_2):
                    self.end_node_list.append(nearby_node_2)
                pass

            # 5. swap start and end trees
            self.end_node_list, self.start_node_list = self.start_node_list, self.end_node_list
        # endtodo ----------------------------------------------------------------------------------------------
        return None  # cannot find path

    # ------------------------------do not change helper methods below ----------------------------
    def steer(self, from_node, to_node, extend_length=float("inf")):
        """
        given two nodes from_node, to_node, this method returns a node new_node such that new_node
        is “closer” to to_node than from_node is.
        """

        new_node = self.node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        # how many intermediate positions are considered between from_node and to_node
        n_expand = math.floor(extend_length / self.path_resolution)

        # compute all intermediate positions
        for _ in range(n_expand):
            new_node.x += self.path_resolution * cos_theta
            new_node.y += self.path_resolution * sin_theta
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
        new_node.parent = from_node

        return new_node

    def is_collision_free(self, new_node):
        """
        determine if nearby_node (new_node) is in the collision-free space.
        """
        if new_node is None:
            return True
        points = np.vstack((new_node.path_x, new_node.path_y)).T
        for obs in self.obstacle_list:
            in_collision = obs.is_in_collision_with_points(points)
            if in_collision:
                return False

        return True  # safe

    def generate_final_course(self, start_mid_point, end_mid_point):
        """
        reconstruct path from start to end node
        """
        # first half

        node = self.start_node_list[start_mid_point]
        path = []
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        # other half
        node = self.end_node_list[end_mid_point]
        path = path[::-1]
        other_sect = []
        while node.parent is not None:
            other_sect.append([node.x, node.y])
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        other_sect.append([node.x, node.y])
        distances = []
        for i in range(len(path) - 1):
            distances.append(np.linalg.norm(np.array(path[i]) - np.array(path[i + 1])))
        return path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        x = self.width * np.random.random_sample()
        y = self.height * np.random.random_sample()
        rnd = self.node(x, y)
        return rnd

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        # compute euclidean disteance between rnd_node and all nodes in tree
        # return index of closest element
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta


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






class controller(Operate):
    def __init__(self, args, TITLE_FONT, TEXT_FONT):
        self.operate = Operate(args, TITLE_FONT, TEXT_FONT)
        # Goal is the immediate destination, waypoints is the list of destinations
        self.dist_between_points = lambda pose, goal: np.linalg.norm(np.array(pose) - np.array(goal))
        self.angle_between_points = lambda pose, goal: np.arctan2(goal[1] - pose[1], goal[0] - pose[0])

        # variables
        self.get_pose = lambda: [i[0] for i in self.operate.ekf.robot.state[:3]]
        self.pose = self.get_pose()  # x, y, theta
        self.state = {'Turn': 0}
        self.waypoints = []
        self.goal = [0., 0.]
        self.control_clock = 0


        # Params
        self.dist_from_goal = 0.07
        self.max_angle = np.pi / 10  # Maximum offset angle from goal before correction
        self.min_angle = self.max_angle * 0.75  # Maximum offset angle from goal after correction
        self.goal_reached = False
        self.look_ahead = 0.2

        while not self.operate.quit:
           self.operate.update_keyboard()
           self.operate.take_pic()
           drive_meas = self.operate.control()
           self.operate.update_slam(drive_meas)
           self.operate.record_data()
           self.operate.save_image()
           self.operate.detect_target()
           # visualise
           self.operate.draw(canvas)
           pygame.display.update()

    def get_path(self, path):
        self.goal_reached = False
        print("pose-->", self.pose)
        self.waypoints = path
        while (len(self.waypoints) > 1 and self.dist_between_points(self.pose[:2],
                                                                    self.waypoints[0]) < self.look_ahead):
            print('way len', len(self.waypoints))
            self.waypoints.pop(0)

        self.goal = self.waypoints[0]
        print('fin get_path')
        x = self.main()
        return x

    def main(self):
        '''
        Aim: take in waypoint, travel to waypoint
        '''
        self.control_clock = time.time()
        while not self.goal_reached:
            angle_to_rotate = self.calculate_angle_from_goal()
            dist_to_goal = self.dist_between_points(self.pose[:2], self.goal)
            print('pose, goal', self.pose, self.goal)
            print('Dist2Goal: {:.3f} || Ang2Goal: {:.3f}'.format(dist_to_goal, math.degrees(angle_to_rotate)))
            if dist_to_goal > self.dist_from_goal:
                # Check if we need to rotate or drive straight
                if (self.state['Turn'] == 0 and abs(angle_to_rotate) > self.max_angle) or self.state['Turn'] == 1:
                    # Drive curvy
                    self.state['Turn'] = 1
                    if abs(angle_to_rotate) < self.min_angle:
                        self.state['Turn'] = 0
                    print('Drive turn')
                    self.drive(ang_to_rotate=angle_to_rotate)
                elif self.state['Turn'] == 0:
                    # Drive straight
                    print('Drive straight')
                    self.drive(ang_to_rotate=0)
                else:
                    print('BOI YO DRIVING BE SHITE', self.state['Turn'], angle_to_rotate)
            else:
                # Waypoint reached
                if len(self.waypoints) == 1 or len(self.waypoints) == 0:
                    # Destination reached
                    print('Goal achieved')
                    self.drive(0, 0)
                    self.goal_reached = True
                else:
                    # look for next waypoint
                    print('WAYPOINT ACHIEVED')
                    self.waypoints.pop(0)
                    self.goal = self.waypoints[0]
        return self.pose


    def drive(self, ang_to_rotate=0, value=0.5):
        curve = 0.0
        direction = np.sign(ang_to_rotate)
        turn_speed = 10
        drive_speed = 50

        if direction == 0 and value != 0:
            # drive straight
            # self.operate.command['motion'] = [1, 0]
            lv, rv = self.operate.pibot.set_velocity([1, 0], tick=drive_speed, time=0)
        else:
            # turn
            lv, rv = self.operate.pibot.set_velocity([0, int(direction)], turning_tick=turn_speed,
                                             time=0)  # set_velocity([0, dir], turning_tick=wheel_vel, time=turn_time)
            # self.operate.command['motion'] = [0, direction]
        drive_meas = measure.Drive(lv, rv, time.time() - self.control_clock)  # this is our drive message to update the slam
        self.control_clock = time.time()
        self.update_slam(drive_meas)
        time.sleep(0.01)
        self.pose = self.get_pose()

    def calculate_angle_from_goal(self):
        angle_to_rotate = self.angle_between_points(self.pose, self.goal) - self.pose[2]
        # Ensures minimum rotation
        if angle_to_rotate < -np.pi:
            angle_to_rotate += 2 * np.pi
        if angle_to_rotate > np.pi:
            angle_to_rotate -= 2 * np.pi
        return angle_to_rotate

    def update_slam(self, drive_meas):
        self.operate.take_pic()
        lms, aruco_img = self.operate.aruco_det.detect_marker_positions(self.operate.img)
        is_success = self.operate.ekf.recover_from_pause(lms)
        if True:#not is_success:
            print('NOT SUCCESS')
            self.operate.ekf.predict(drive_meas)
            self.operate.ekf.add_landmarks(lms)  # will only add if something new is seen
            self.operate.ekf.update(lms)
        else:
            print('----------------success----')

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
                  1, (map_dimension / map_resolution) - 1)
    pixel_y = map(pose[1], morigin, -morigin,
                  1, (map_dimension / map_resolution) - 1)
    return int(pixel_x), int(pixel_y)

def pixel_to_pose(pixel,map_dimension,map_resolution):
    morigin = map_dimension / 2.0
    map = lambda old_value, old_min, old_max, new_max, new_min: ((old_value - old_min) / (old_max - old_min)) * (
                new_max - new_min) + new_min
    pose_x = map(pixel[0], (map_dimension / map_resolution) - 1, 1, morigin,
                 -morigin)
    pose_y = map(pixel[1],(map_dimension / map_resolution) - 1,1, morigin,
                 -morigin)
    return pose_x, pose_y

# main loop
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--ckpt", default='network/scripts/model/model.best.pth')
    parser.add_argument("--map", type=str, default='M4_true_map_5fruits.txt')
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

    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter % 10 // 2], (x_, 565))
            pygame.display.update()
            counter += 2

    operate = controller(args, TITLE_FONT, TEXT_FONT)

    ############################################################################################
    ########################################################################################


    # ppi = operate.pibot

    # read in the True map
    fruits_list, fruits_True_pos, aruco_True_pos = read_True_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_True_pos)

    # the needed parameters
    # fileS = "calibration/param/scale.txt"
    # scale = np.loadtxt(fileS, delimiter=',')
    # fileB = "calibration/param/baseline.txt"
    # baseline = np.loadtxt(fileB, delimiter=',')
    # fileI = "calibration/param/intrinsic.txt"
    # intrinsic = np.loadtxt(fileI, delimiter=',')
    # fileD = "calibration/param/distCoeffs.txt"
    # dist_coeffs = np.loadtxt(fileD, delimiter=',')

    # waypoint = [0.0, 0.0]
    # robot_pose = [0.0, 0.0, 0.0]

    # intialise slam
    # if args.ip == 'localhost':
    #    scale /= 2
    #    #pass
    # robot = Robot(baseline, scale, intrinsic, dist_coeffs)
    # operate.ekf = operate.ekf # Occupancy list
    # operate.aruco_det = operate.aruco_det

    # since we know where the markes is, chaneg the self.markers in the ekf operate.ekf
    # do a for loop for all markers including fruits
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
    operate.operate.ekf.add_landmarks(fruits)  # will add known
    for i in range(len(aruco_True_pos)):
        print(str(i), aruco_True_pos[i])
        item = item_in_map(str(i), aruco_True_pos[i])
        aruco_list.append(item)
    #checking
    time.sleep(2)
    # aruco_list[0].tag = str(10)
    operate.operate.ekf.add_landmarks(aruco_list)  # will add known
    # print(operate.ekf.markers)
    # print(operate.ekf.taglist)
    # print(operate.ekf.P)

    # The following code is only a skeleton code the semi-auto fruit searching task
    # implement RRT, loop is for the number of fruits

    # g_offset = 0.3
    # off = lambda x: x - g_offset if x > 0 else x + g_offset
    goal = [[f_dict[i][0], f_dict[i][1]] for i in search_list ]
    # goal = [f_dict[i] - [g_offset, g_offset] for i in search_list]
    obstacles_aruco = []
    lms_xy = operate.operate.ekf.markers[:2, :]
    for i in range(len(operate.operate.ekf.markers[0, :])):
        obstacles_aruco.append(Circle(lms_xy[0, i], lms_xy[1, i]))
    expand_dis = 0.4
    print("obstacles:", obstacles_aruco)
    print(start)
    start = list(operate.operate.ekf.robot.state[0:2, :])

    # --- occupancy grid
    map_resolution = 0.1 # metres / pixel
    map_dimension = 1.4 * 2 # metres
    map_size = int(map_dimension / map_resolution)
    map_arr = np.ones((map_size, map_size)) # shape = 28*28
    '''
    Now add obstacles to the map in the correct location
    - convert centre of obstacles to map frame
    - input obstacles into map of value np.inf
    - pad obstacles by 1 in x or y
    heuristic = [[-1, 0], [1, 0], [0, 1], [0, -1]]
    for point in obstacles:
        for new_point in heuristic:
            map[point + new_point] = np.inf 
    convert start pose to pixel frame
    convert goal pose to pixel frame
    visualise path as well plz
    '''
    def pad(map, item_list, full_pad=True):
        if full_pad:
            pad = 2
        else:
            pad = 1
        for item in item_list:
            try:
                map[item[0] - pad: item[0] + pad, item[1] - pad: item[1] + pad] = np.inf
            except:
                pass
        return map

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

    # debug start and end
    map_arr = np.array(map_arr, dtype=np.float32)
    for g in goal_map_frame:
        map_arr[start_map_frame[0], start_map_frame[1]] = 1
        path = pyastar2d.astar_path(map_arr, start_map_frame, g, allow_diagonal=True)
        print('map', map_arr.shape, map_dimension, map_resolution)
        print('start/goal val', map_arr[start_map_frame[0], start_map_frame[1]], map_arr[goal_map_frame[0], goal_map_frame[1]])
        print('obstacles', obstacles_map_frame)
        new_goal = g
        new_start = start_map_frame
        while type(path) == type(None):
            new_goal = [i - 1 if i > int(map_size/2) else i + 1 for i in new_goal]
            new_start = [i - 1 if i > int(map_size/2) else i + 1 for i in new_start]
            map_arr[new_start[0], new_start[1]] = 1
            map_arr[new_goal[0], new_goal[1]] = 1
            path = pyastar2d.astar_path(map_arr, new_start, new_goal, allow_diagonal=True)
            print('path failed', new_goal, goal_map_frame, path)
            time.sleep(0.5)

        path_pose = []

        # converting from map frame to pose
        for item in path:
            x_obs, y_obs = pixel_to_pose(item, map_dimension, map_resolution)
            path_pose.append([x_obs, y_obs])


        # for g in goal:
        # map: wall = np.inf, empty space = 1.
        # String pulling
        # def str_pull: identifies vertecies in lines, and removes redundant waypoints using gradients

        route = [[float(i[0]), float(i[1])] for i in path_pose]
        #route = route[::-1]

        _ = input("press enter to start moving:... \nstart -- {},\nend -- {},\nroute -- {}".format(
            (pixel_to_pose(new_start, map_dimension, map_resolution), start), (pixel_to_pose(new_goal, map_dimension, map_resolution), goal), route))
        start = operate.get_path(route)

        x_start,y_start = pose_to_pixel(start,map_dimension,map_resolution)
        start_map_frame = [x_start,y_start]
