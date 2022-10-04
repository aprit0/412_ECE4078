# m4 - autonomous fruit searching

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

# import dependencies and set random seed
seed_value = 5
# 1. set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 2. set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# import slam components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "{}/util".format(os.getcwd()))
from pibot import PenguinPi
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
                 expand_dis=3.0, path_resolution=0.5, max_points=200):
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
                    aruco_True_pos[9][0] = x
                    aruco_True_pos[9][1] = y
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


# waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# additional improvements:
# you may use different motion model parameters for robot driving on its own or driving while pushing a fruit
# try changing to a fully automatic delivery approach: develop a path-finding algorithm that produces the waypoints

def drive_to_point(waypoint, robot_pose, ppi, kalman, aruco_marks):
    baseline = kalman.robot.wheels_width
    print('baseline: ', baseline)
    scale = kalman.robot.wheels_scale
    print('scale: ', scale)

    ####################################################
    # todo: replace with your codes to make the robot drive to the waypoint
    # one simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point

    # waypoint: [x,y]
    # robot_pose: [x,y,theta]

    theta = np.arctan2((waypoint[1] - robot_pose[1]), (waypoint[0] - robot_pose[0]))
    error_theta = theta - robot_pose[2]  # rotate robot by error_theta
    # print(error_theta/(2*np.pi)*360)

    # finding distance to move
    pos_dif = [robot_pose[0] - waypoint[0], robot_pose[1] - waypoint[1]]
    dist = np.sqrt(pos_dif[0] ** 2 + pos_dif[1] ** 2)  # direct straight distance between robot and waypoint

    dist_to_waypoint = 0  # 0.5/2 #distance within waypoint with which we want to stop

    dist_to_travel = dist - dist_to_waypoint  # robot moves by dist_to_travel

    # setting wheel velocity
    drive_speed = 50  # tick
    turn_speed = 20  # ticks

    # turn towards the waypoint
    # turn_time = (error_theta/scale)*baseline*(1/wheel_vel)
    turn_time = error_theta * baseline / (turn_speed * scale * 2)
    dir = 1
    if turn_time < 0:
        dir = -1
        turn_time = np.abs(turn_time)

    print("turning for {:.2f} seconds".format(turn_time))

    # since we want slam to run simultaneously  with drive, implement slam update in [[i set velocity]]
    while turn_time > 10:  # to update slam every one sec
        lv, rv = ppi.set_velocity([0, dir], turning_tick=turn_speed,
                                  time=1)  # set_velocity([0, dir], turning_tick=wheel_vel, time=turn_time)
        drive_meas = measure.drive(lv, rv, 1)  # this is our drive message to update the slam
        update_slam(drive_meas, aruco_marks, ppi, kalman)
        turn_time -= 1
    lv, rv = ppi.set_velocity([0, dir], turning_tick=turn_speed,
                              time=turn_time)  # set_velocity([0, dir], turning_tick=wheel_vel, time=turn_time)
    drive_meas = measure.drive(lv, rv, turn_time)  # this is our drive message to update the slam
    update_slam(drive_meas, aruco_marks, ppi, kalman)
    print('state: ', kalman.robot.state.flatten())

    time.sleep(1)

    # after turning, drive straight to the waypoint
    drive_time = dist_to_travel / (drive_speed * scale)
    print("driving for {:.2f} seconds".format(drive_time))
    while drive_time > 10:
        lv, rv = ppi.set_velocity([1, 0], tick=drive_speed, time=1)
        drive_meas = measure.drive(lv, rv, 1)  # this is our drive message to update the slam
        update_slam(drive_meas, aruco_marks, ppi, kalman)
        drive_time -= 1
        time.sleep(1)
    lv, rv = ppi.set_velocity([1, 0], tick=drive_speed,
                              time=drive_time)  # set_velocity([0, dir], turning_tick=wheel_vel, time=turn_time)
    drive_meas = measure.drive(lv, rv, drive_time)  # this is our drive message to update the slam
    update_slam(drive_meas, aruco_marks, ppi, kalman)
    ####################################################
    print('state: ', kalman.robot.state.flatten())

    print("arrived at [{}, {}]".format(waypoint[0], waypoint[1]))


def get_robot_pose(kalman):
    ####################################################
    # todo: replace with your codes to estimate the phose of the robot
    # we strongly recommend you to use your slam code from m2 here

    # update the robot pose [x,y,theta]
    # robot_pose = [0.0,0.0,0.0] # replace with your calculation
    robot_pose = kalman.robot.state.flatten()  # kalman.get_state_vector()# replace with your calculation
    ####################################################

    return robot_pose


def update_slam(drive_meas, aruco, pibot, kalman):
    image = pibot.get_image()
    lms, aruco_img = aruco.detect_marker_positions(image)
    kalman.predict(drive_meas)
    kalman.add_landmarks(lms)  # will only add if something new is seen
    kalman.update(lms)


# aiden 4191######################################################################

def calculate_angle_from_goal(pose, goal):
    pose = [i[0] for i in pose]
    print("pose --> ", pose)
    print("goal -->", goal)
    angle_between_points = lambda pose, goal: np.arctan2(goal[1] - pose[1], goal[0] - pose[0])
    angle_to_rotate = angle_between_points(pose, goal) - pose[2]
    print("angle-->", angle_to_rotate)
    # ensures minimum rotation
    if angle_to_rotate < -np.pi:
        angle_to_rotate += 2 * np.pi
    if angle_to_rotate > np.pi:
        angle_to_rotate -= 2 * np.pi
    return angle_to_rotate


class controller:
    def __init__(self, operate):
        # Goal is the immediate destination, waypoints is the list of destinations
        self.dist_between_points = lambda pose, goal: np.linalg.norm(np.array(pose) - np.array(goal))
        self.angle_between_points = lambda pose, goal: np.arctan2(goal[1] - pose[1], goal[0] - pose[0])

        # variables
        self.operate = operate
        self.ekf = operate.ekf
        self.aruco = operate.aruco_det
        self.get_pose = lambda: [i[0] for i in self.ekf.robot.state[:3]]
        self.pose = self.get_pose()  # x, y, theta
        self.goal = goal  # x, y
        self.state = {'Turn': 0}
        self.waypoints = []
        self.goal = [0., 0.]


        # Params
        self.dist_from_goal = 0.05
        self.max_angle = np.pi / 18  # Maximum offset angle from goal before correction
        self.min_angle = self.max_angle * 0.5  # Maximum offset angle from goal after correction
        self.goal_reached = False
        self.look_ahead = 0.2

    def get_path(self, path):
        print("pose-->", self.pose)
        self.waypoints = path
        while (len(self.waypoints) > 1 and self.dist_between_points(self.pose[:2],
                                                                    self.waypoints[0]) < self.look_ahead):
            print('way len', len(self.waypoints))
            self.waypoints.pop(0)

        self.goal = self.waypoints[0]
        self.main()

    def main(self):
        '''
        Aim: take in waypoint, travel to waypoint
        '''
        while not self.goal_reached:
            angle_to_rotate = self.calculate_angle_from_goal()
            dist_to_goal = self.dist_between_points(self.pose[:2], goal)
            print('pose, goal', self.pose, self.goal)
            print('Dist2Goal: {:.3f} || Ang2Goal: {:.3f}'.format(dist_to_goal, math.degrees(angle_to_rotate)))
            if dist_to_goal > self.dist_from_goal:
                # Check if we need to rotate or drive straight
                if (self.state['Turn'] == 0 and abs(angle_to_rotate) > self.max_angle) or self.state['Turn'] == 1:
                    # Drive curvy
                    self.state['Turn'] = 1
                    if abs(angle_to_rotate) < self.min_angle:
                        self.state['Turn'] = 0
                    self.drive(ang_to_rotate=angle_to_rotate)
                elif self.state['Turn'] == 0:
                    # Drive straight
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
                    self.waypoints.pop(0)
                    self.goal = self.waypoints[0]


    def drive(self, ang_to_rotate=0, value=0.5):
        curve = 0.0
        direction = np.sign(ang_to_rotate)
        turn_speed = 20
        drive_speed = 50

        if direction == 0:
            # drive straight
            lv, rv = ppi.set_velocity([1, 0], tick=drive_speed, time=1)
        else:
            # turn
            lv, rv = ppi.set_velocity([0, direction], turning_tick=turn_speed,
                                      time=1)  # set_velocity([0, dir], turning_tick=wheel_vel, time=turn_time)
        drive_meas = measure.Drive(lv, rv, 1)  # this is our drive message to update the slam
        update_slam(drive_meas, self.aruco, self.operate.pibot, self.ekf)
        time.sleep(2)
        update_slam(drive_meas, self.aruco, self.operate.pibot, self.ekf)
        self.pose = self.get_pose()

    def calculate_angle_from_goal(self):
        angle_to_rotate = self.angle_between_points(self.pose, self.goal) - self.pose[2]
        # Ensures minimum rotation
        if angle_to_rotate < -np.pi:
            angle_to_rotate += 2 * np.pi
        if angle_to_rotate > np.pi:
            angle_to_rotate -= 2 * np.pi
        return angle_to_rotate


#####################################################################################################

class item_in_map:
    def __init__(self, name, measurement):
        self.tag = name
        self.coordinates = np.array([[measurement[0]], [measurement[1]]])


# main loop
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=40000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--ckpt", default='network/scripts/model/model.best.pth')
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

    operate = Operate(args, TITLE_FONT, TEXT_FONT)

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
    ############################################################################################
    ########################################################################################

    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=40000)
    args, _ = parser.parse_known_args()

    ppi = operate.pibot

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
    operate.ekf.add_landmarks(fruits)  # will add known
    for i in range(len(aruco_True_pos)):
        item = item_in_map(str(i), aruco_True_pos[i])
        aruco_list.append(item)
    aruco_list[0].tag = str(10)
    operate.ekf.add_landmarks(aruco_list)  # will add known
    # print(operate.ekf.markers)
    # print(operate.ekf.taglist)
    # print(operate.ekf.P)

    # The following code is only a skeleton code the semi-auto fruit searching task
    # implement RRT, loop is for the number of fruits

    start = list(operate.ekf.robot.state[0:2, :])
    g_offset = 0.3
    # goal = [fruits[0].coordinates[0][0] - g_offset, fruits[0].coordinates[1][0] ]#- g_offset]
    goal = [f_dict[i] - [g_offset, g_offset] for i in search_list]
    obstacles_aruco = []
    lms_xy = operate.ekf.markers[:2, :]
    for i in range(len(operate.ekf.markers[0, :])):
        obstacles_aruco.append(Circle(lms_xy[0, i], lms_xy[1, i]))
    expand_dis = 0.4
    print("obstacles:", obstacles_aruco)
    print(start)
    print(goal)
    rrt = RRTC(start=start, goal=goal[0], width=2.4, height=2.4, obstacle_list=obstacles_aruco, expand_dis=expand_dis,
               path_resolution=0.2)
    # try:
    route = rrt.planning()
    # except Exception as e:
    #     print(e)
    route = [[float(i[0]), float(i[1])] for i in route]
    print("route -->", route)
    print("length-->", len(route))

    # x = [i[0] for i in route]
    # y = [i[1] for i in route]
    # plt.scatter(x,y)
    # plt.show()
    input("press enter to start moving:... start {},\n end {},\n route{}".format(start, goal, route[0]))
    # for i in range(len(route) - 2, -1, -1):
    #     print('begin')
    #     destination = route[i]
    #     waypoint = [destination[0], destination[1]]
    greg = controller(operate)
    greg.get_path(route)

    # for i in range(len(route) - 2, -1, -1):
    #     print('begin')
    #     destination = route[i]
    #     waypoint = [destination[0], destination[1]]
    #     # try:
    #     robot_pose = get_robot_pose(operate.ekf)
    #     drive_to_point(waypoint, robot_pose, ppi, operate.ekf, operate.aruco_det)
    #     # except Exception as e:
    #         # print(e)
    #     # estimate the robot's pose
    #     print('get pose')
    #     robot_pose = get_robot_pose(operate.ekf)
    #     print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint, robot_pose))
    #
    #     # exit
    #     ppi.set_velocity([0, 0])
    #     print('stopping for 3 seconds')
    #     time.sleep(3)
