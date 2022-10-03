# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
import random
import math

# Import dependencies and set random seed
seed_value = 5
# 1. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 2. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)


# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "{}/util".format(os.getcwd()))
from pibot import PenguinPi
import measure as measure

class Circle:
    def __init__ (self,c_x, c_y, radius = 0.3):
        self.center = np.array([c_x,c_y])
        self.radius = radius

    def is_in_collision_with_points(self,points):
        dist = []
        for point in points:
            dx = self.center[0] - point[0]
            dy = self.center[1] - point[1]

            dist.append(dx*dx + dy*dy)
        if np.min(dist) <= self.radius **2:
            return True
        return False


class RRTC:
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
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
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacle_list: list of obstacle objects
        width, height: search area
        expand_dis: min distance between random node and closest node in rrt to it
        path_resolution: step size to considered when looking for node to expand
        """
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.width = width
        self.height = height
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.max_nodes = max_points
        self.obstacle_list = obstacle_list
        self.start_node_list = []  # Tree from start
        self.end_node_list = []  # Tree from end

    def planning(self):
        """
        rrt path planning
        """
        self.start_node_list = [self.start]
        self.end_node_list = [self.end]
        while len(self.start_node_list) + len(self.end_node_list) <= self.max_nodes:
            # print(self.start_node_list)
            # print(self.end_node_list)
            # TODO: Complete the planning method ----------------------------------------------------------------
            # 1. Sample and add a node in the start tree
            rand_sample = self.get_random_node()
            expansion_id = self.get_nearest_node_index(self.start_node_list, rand_sample)
            expansion_node = self.start_node_list[expansion_id]

            # 2. Check whether trees can be connected, check if the sampled node is less than expand_dis
            # check for colission but 'expand using steer'to set up the path then add after checking collision status
            nearby_node = self.steer(expansion_node, rand_sample, self.expand_dis)

            if self.is_collision_free(nearby_node):
                self.start_node_list.append(nearby_node)

                # check if there is a node lesser than expand_dis in end_list
                end_expand_id = self.get_nearest_node_index(self.end_node_list, nearby_node)
                end_expand_node = self.end_node_list[end_expand_id]

                check_dist, _ = self.calc_distance_and_angle(end_expand_node, nearby_node)

                # 3. Add the node that connects the trees and generate the path
                if check_dist < self.expand_dis:  # true path is found
                    final_path = self.steer(end_expand_node, nearby_node, self.expand_dis)
                    if self.is_collision_free(final_path):
                        # Note: It is important that you return path found as:
                        self.end_node_list.append(final_path)
                        return self.generate_final_course(len(self.start_node_list) - 1, len(self.end_node_list) - 1)
                        # 4. Sample and add a node in the end tree
                rand_sample_2 = self.get_random_node()
                expansion_2 = self.get_nearest_node_index(self.end_node_list, rand_sample_2)
                expansion_node_2 = self.end_node_list[expansion_2]

                # Check whether trees can be connected
                # check for colission but 'expand using steer'to set up the path then add after checking collision status
                nearby_node_2 = self.steer(expansion_node_2, rand_sample_2, self.expand_dis)

                if self.is_collision_free(nearby_node_2):
                    self.end_node_list.append(nearby_node_2)

            # 5. Swap start and end trees
            self.end_node_list, self.start_node_list = self.start_node_list, self.end_node_list
        # ENDTODO ----------------------------------------------------------------------------------------------
        return None  # cannot find path

    # ------------------------------DO NOT change helper methods below ----------------------------
    def steer(self, from_node, to_node, extend_length=float("inf")):
        """
        Given two nodes from_node, to_node, this method returns a node new_node such that new_node
        is “closer” to to_node than from_node is.
        """

        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        # How many intermediate positions are considered between from_node and to_node
        n_expand = math.floor(extend_length / self.path_resolution)

        # Compute all intermediate positions
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
        Determine if nearby_node (new_node) is in the collision-free space.
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
        Reconstruct path from start to end node
        """
        # First half

        node = self.start_node_list[start_mid_point]
        path = []
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        # Other half
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
        rnd = self.Node(x, y)
        return rnd

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        # Compute Euclidean disteance between rnd_node and all nodes in tree
        # Return index of closest element
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

def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 3 types of target fruit to search

    @param fname: filename of the map
    @return:
        1) list of target fruits, e.g. ['apple', 'pear', 'lemon']
        2) locations of the target fruits, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)  # Reading every x coordinates
            y = np.round(gt_dict[key]['y'], 1)  # Reading every y coordinates

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5]) # Giving ID to ARUCO markers
                    aruco_true_pos[marker_id][0] = x
                    aruco_true_pos[marker_id][1] = y
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        return fruit_list, fruit_true_pos, aruco_true_pos


def read_search_list():
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open('search_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order

    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """

    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(3):
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1


# Waypoint navigation
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
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
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

    print("Turning for {:.2f} seconds".format(turn_time))

    # since we want slam to run simultaneously  with drive, implement slam update in [[i set velocity]]
    while turn_time > 10:  # to update slam every one sec
        lv, rv = ppi.set_velocity([0, dir], turning_tick=turn_speed,
                                  time=1)  # set_velocity([0, dir], turning_tick=wheel_vel, time=turn_time)
        drive_meas = measure.Drive(lv, rv, 1)  # this is our drive message to update the slam
        update_slam(drive_meas, aruco_marks, ppi, kalman)
        turn_time -= 1
    lv, rv = ppi.set_velocity([0, dir], turning_tick=turn_speed,
                              time=turn_time)  # set_velocity([0, dir], turning_tick=wheel_vel, time=turn_time)
    drive_meas = measure.Drive(lv, rv, turn_time)  # this is our drive message to update the slam
    update_slam(drive_meas, aruco_marks, ppi, kalman)
    print('state: ', kalman.robot.state.flatten())

    time.sleep(3)

    # after turning, drive straight to the waypoint
    drive_time = dist_to_travel / (drive_speed * scale)
    print("Driving for {:.2f} seconds".format(drive_time))
    while drive_time > 10:
        lv, rv = ppi.set_velocity([1, 0], tick=drive_speed, time=1)
        drive_meas = measure.Drive(lv, rv, 1)  # this is our drive message to update the slam
        update_slam(drive_meas, aruco_marks, ppi, kalman)
        drive_time -= 1
        time.sleep(1)
    lv, rv = ppi.set_velocity([1, 0], tick=drive_speed,
                              time=drive_time)  # set_velocity([0, dir], turning_tick=wheel_vel, time=turn_time)
    drive_meas = measure.Drive(lv, rv, drive_time)  # this is our drive message to update the slam
    update_slam(drive_meas, aruco_marks, ppi, kalman)
    ####################################################
    print('state: ', kalman.robot.state.flatten())

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))


def get_robot_pose(kalman):
    ####################################################
    # TODO: replace with your codes to estimate the phose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here

    # update the robot pose [x,y,theta]
    # robot_pose = [0.0,0.0,0.0] # replace with your calculation
    robot_pose = kalman.robot.state.flatten()  # kalman.get_state_vector()# replace with your calculation
    ####################################################

    return robot_pose

def update_slam(drive_meas, aruco, pibot, kalman):
    image = pibot.get_image()
    lms, aruco_img = aruco.detect_marker_positions(image)
    kalman.predict(drive_meas)
    kalman.add_landmarks(lms,1) #will only add if something new is seen
    kalman.update(lms)

class item_in_map:
    def __init__(self,name, measurement):
        self.tag = name
        self.coordinates = np.array([[measurement[0]],[measurement[1]]])


# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=40000)
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip, args.port)

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    print('aruco: ', aruco_true_pos)

    # the needed parameters
    fileS = "calibration/param_simulation/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param_simulation/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    fileI = "calibration/param_simulation/intrinsic.txt"
    intrinsic = np.loadtxt(fileI, delimiter=',')
    fileD = "calibration/param_simulation/distCoeffs.txt"
    dist_coeffs = np.loadtxt(fileD, delimiter=',')

    waypoint = [0.0, 0.0]
    robot_pose = [0.0, 0.0, 0.0]

    # intialise slam
    # if args.ip == 'localhost':
    #    scale /= 2
    #    #pass
    robot = Robot(baseline, scale, intrinsic, dist_coeffs)
    kalman_filter = EKF(robot) # Occupancy list
    markers_matrix = aruco.aruco_detector(robot, marker_length=0.07)

    # since we know where the markes is, chaneg the self.markers in the ekf kalman_filter
    # do a for loop for all markers including fruits
    aruco_list = []
    fruits = []
    for i in range(len(fruits_true_pos)):
        item = item_in_map(fruits_list[i], fruits_true_pos[i])
        fruits.append(item)
    kalman_filter.add_landmarks(fruits, 0)  # will add known
    for i in range(len(aruco_true_pos)):
        item = item_in_map(str(i), aruco_true_pos[i])
        aruco_list.append(item)
    aruco_list[0].tag = str(10)
    kalman_filter.add_landmarks(aruco_list, 0)  # will add known
    # print(kalman_filter.markers)
    # print(kalman_filter.taglist)
    # print(kalman_filter.P)

    # The following code is only a skeleton code the semi-auto fruit searching task
    # implement RRT, loop is for the number of fruits

    start = [0, 0]
    goal = [fruits[0].coordinates[0][0], fruits[0].coordinates[1][0]]
    obstacles_aruco = []
    for item in aruco_list:
        obstacles_aruco.append(Circle(item.coordinates[0], item.coordinates[1]))
    expand_dis = 0.4
    print(start)
    print(goal)
    rrt = RRTC(start=start, goal=goal, width=1, height=1, obstacle_list=obstacles_aruco, expand_dis=expand_dis,
               path_resolution=0.2)
    route = rrt.planning()
    print(route)
    input("press enter to start moving:...")
    for i in range(len(route) - 2, -1, -1):
        destination = route[i]
        waypoint = [destination[0], destination[1]]
        drive_to_point(waypoint, robot_pose, ppi, kalman_filter, markers_matrix)
        # estimate the robot's pose
        robot_pose = get_robot_pose(kalman_filter)
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint, robot_pose))

        # exit
        ppi.set_velocity([0, 0])
        print('stopping for 3 seconds')
        time.sleep(3)



