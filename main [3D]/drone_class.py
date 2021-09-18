import numpy as np
import math
import scipy.stats as stats
from sklearn.cluster import KMeans
from scipy import interpolate
import cv2
import os
import time
import imutils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf


class Drone:
    def __init__(self):
        self.drone_ID = 0                                       # unique ID identifying the drone
        self.flock_leader = 0                                   # if we decide to introduce a "flock leader" who has to compute the "fake" waypoints for everyone
        self.position = np.empty([0, 3], dtype='float16')       # current position of the drone
        self.index_position = np.array([0, 0, 0], dtype='int16')
        self.orientation = np.array([0, math.pi/2])             # orientation in radians
        self.position_history = np.empty([0, 3], dtype='float16')# vector containing all the past values of the drone position
        self.radius = 0.2                                       # drone dimensions [m]
        self.speed = np.zeros([3])                              # current speed
        self.max_speed = None                                   # max abs of the speed vector
        self.motion_options = None
        self.vision_range = None                                # vision range of the drone [m]
        self.max_vision_angle = None                            # max angle of vision [degrees]
        self.vision_resolution = None                           # how many ray are used in the vision range
        self.map_dimension_x = None                             # initialization of map dimension. Precise number if the map is known, upper approx. if map is unknown
        self.map_dimension_y = None
        self.map_dimension_z = None
        self.map_resolution_x = None                            # resolution of the potential field matrix (dimension of the cells)
        self.map_resolution_y = None
        self.map_resolution_z = None
        self.n_cell_x = None                                    # number of cell along the x_coordinate dimension of the map (computed from the data above)
        self.n_cell_y = None
        self.n_cell_z = None
        self.min_index_x = None                                 # limit indexes of the internal map matrix
        self.max_index_x = None
        self.min_index_y = None
        self.max_index_y = None
        self.min_index_z = None
        self.max_index_z = None
        self.initialization_completed = False                   # only false at the beginning of the simulation to allow initialization of the map matrix
        self.potential_field_map = None
        self.attractive_constant = None                         # attractive potential gain
        self.layer_attractive = None
        self.x_cohordinate_matrix = None                        # used to speed up a lot the attractive pot. field computation
        self.y_cohordinate_matrix = None
        self.z_cohordinate_matrix = None
        self.obstacle_map = None
        self.obstacle_matrix = None
        self.obstacle_list = np.empty([0,3], dtype='float16')   # stores all obstacles position as indexes of the corresponding cell
        self.obstacle_list_security_length = 50000              # maximum length of the obstacle list
        self.position_list_security_length = 5000                # maximum length of the position history list
        self.newfound_obstacle_list = np.empty([0, 3], dtype='float16')# temporarily stores newfound obstacles position
        self.newfound_obstacle_share_list = np.empty([0, 3])
        self.obstacle_safe_distance = None                      # safe distance from obstacles [m] --> used to build fixed obstacle repulsive matrix
        self.fixed_obstacle_matrix = None                       # potential matrix that represents the presence of an obstacle. APF gradient
                                                                # is decreasing in the neighborhood of the point where the ostacle is located.
                                                                # The fixed obstacle matrix is constant (always the same) and computed once and
                                                                # for all by a dedicated function. Dimensions of this matrix are given by
                                                                # the fixed_obstacle_matrix_dimensions property (this matrix is, in general,
                                                                # much smaller than the potential_field_map matrix
        self.fixed_obstacle_matrix_dimensions = None            # dimension of the matrix initialized in the line above. Dimensions are saved as [n°rows, n°columns] i.e. [len(y_coordinate), len(x_coordinate)]
        self.drone_safe_distance = None                         # safe distance between drones [m] --> used to build drone (mobile obstacles) repulsive matrix
        self.mobile_obstacle_matrix = None                      # repulsive matrix associated with the presence of a mobile obstacle (drone) --> is a 3D matrix, each layer along z
                                                                # represents a different size of the repulsive area
        self.mobile_obstacle_matrix_dimensions = None           #dimension of the matrices used to represent mobile obstacles (other drones) and their future positions
        self.mobile_obstacle_last_position = np.empty([0,4])    # 3-column matrix. 1st and 2nd columns are x_coordinate and y_coordinate of the mobile obstacle matrices.
                                                                # 3rd column indicates which circle is being used (big-medium-small). This matrix
                                                                # is used to delete mobile obstacle circles before drawing the new ones (after
                                                                # the other drone position and velocity has changed)
        self.mobile_matrix_peak_value = None                    # value of the potential in the middle of the 2D distribution for mobile obstacle matrix
        self.fixed_matrix_peak_value = None                     # value of the potential in the middle of the 2D distribution for fixed obstacle matrix
        self.minima_matrix_peak_value = None                    # value of the potential in the middle of the 2D distribution for local minima obstacle matrixes
        self.other_drones_current_position = None               # 4 columns: [x_coordinate, y_coordinate, v_x, v_y], 1 row for each drone
        self.layer_experience = None                            # matrix containing the "experience" component of the potential field
        self.goal_updated = False                               # signals that the goal has changed with respect to the previous one and that the pot. field layer must be updated
        self.goal = [0, 0, 0]                                      # current goal
        self.old_goal = self.goal                               # last goal (used to correctly manage the potential field matrix update)
        self.min_goal_distance = 0.3                            # min distance to reach the goal
        self.path_planning_model = None                         # RL model for path planning
        self.coverage_model = None                              # RL model for coverage
        self.local_minima_flag = False                          # flag for the local minima zone
        self.max_step_apf_descent = None                        # max step performed in classic version of apf algorithm in order to escape from local minima
        self.local_minima_matrix = None                         # matrix to add in order to escape from the local minima zone
        self.minima_obstacle_dimension = None                   # dimension of obstacles add in order to avoid local minima
        self.step_counter = None                                # counter for step passed in the classic version of the path planning algorithm
        self.observation_size = None                            # dimension of the observable matrix for the RL algorithm
        self.saved_vision_angle = None                          # variable that stores the vision angle information
        self.saved_vision_resolution = None                     # variable that stores the vision resolution information
        self.total_explorable_elements = None                   # totality of cells to explore
        self.covered_area_matrix = None                         # matrix that contains information about coverage
        self.single_coverage = [0]                              # list that contains the % of the single exploration of the drone
        self.predict_length = None                              # length of the predicted trajectory
        self.smoothed_trajectory_points = [0, 0, 0]                # list that will contain the points of the interpolated trajectory
        self.proximity_sensor_range = None                      # proximity sensor range [m]
        self.proximity_sensor_resolution = 50                   # resolution of the proximity sensor (n° of ray in a circle)
        self.emergency_path_planning_flag = False               # flag that change the emergency status
        self.emergency_step_counter = None
        self.n_drones = None
        self.id_list = []
        self.goal_steps = 0
        self.max_steps_single_goal = None
        self.column_indexes = None
        self.row_indexes = None
        self.potential_field_sweep_steps = None
        self.spheric_spiral_coordinates = np.empty(3)


    def __str__(self):  # overwrites print. This way the command print(drone_obj) can be used to inspect the current position and objective of a certain drone
        x = 0
        y = 1
        z = 2
        return "drone %i is located in (%.2f, %.2f, %.2f) and its goal is (%.2f, %.2f, %.2f)" % (self.drone_ID, self.position[x], self.position[y], self.position[z], self.goal[x], self.goal[y], self.goal[z])

    def set_drone_ID(self, ID):
        ID = int(round(ID))
        self.drone_ID = ID

    def set_initial_position(self, intial_position):
        x = 0
        y = 1
        z = 2
        intial_position = np.array(intial_position, dtype="f")
        self.position = intial_position
        self.index_position[x] = self.pos2index(self.position[x])
        self.index_position[y] = self.pos2index(self.position[y])
        self.index_position[z] = self.pos2index(self.position[z])

    def set_random_initial_position(self):
        x = 0
        y = 1
        z = 2
        location_found = False
        repeat_counter = 0
        initial_position = [np.random.random() * (self.map_dimension_x - 1), np.random.random() * (self.map_dimension_y - 1), np.random.random() * (self.map_dimension_z - 1)]
        while not location_found:
            repeat_counter += 1
            initial_position = [np.random.random() * (self.map_dimension_x - 1), np.random.random() * (self.map_dimension_y - 1), np.random.random() * (self.map_dimension_z - 1)]
            idx_x = int(self.pos2index(initial_position[0]))
            idx_y = int(self.pos2index(initial_position[1]))
            idx_z = int(self.pos2index(initial_position[2]))
            if self.obstacle_map[idx_z, idx_x, idx_y] != 1:
                location_found = True
            elif repeat_counter > 5:
                break
        self.position = initial_position
        self.index_position[x] = self.pos2index(self.position[x])
        self.index_position[y] = self.pos2index(self.position[y])
        self.index_position[z] = self.pos2index(self.position[z])
        return

    def set_observation_size(self, N):
        self.observation_size = N

    def set_vision_settings(self, N, max_angle, resolution):
        self.vision_range = N
        self.max_vision_angle = max_angle
        self.vision_resolution = resolution

    def set_RL_path_planning_model(self, model):
        self.path_planning_model = model

    def set_RL_coverage_model(self, model):
        self.coverage_model = model

    def set_initial_goal(self, initial_goal):
        new_goal = np.array(initial_goal, dtype="f")
        self.old_goal = self.goal               # stores previous goal in memory
        self.goal = new_goal
        self.update_attractive_layer()

    def set_predict_length(self, predict_length):
        self.predict_length = predict_length

    def set_max_steps_with_single_goal(self, max_steps):
        self.max_steps_single_goal = max_steps

    def set_drone_safe_distance(self, safe_distance):
        self.drone_safe_distance = safe_distance

    def set_fixed_obstacles_safe_distance(self, safe_distance):
        self.obstacle_safe_distance = safe_distance

    def set_minima_obstacles_dimension(self, obstacle_dimension):
        self.minima_obstacle_dimension = obstacle_dimension

    def set_matrix_peak_value(self, mobile_peak_value, fixed_peak_value, minima_peak_value):
        self.mobile_matrix_peak_value = mobile_peak_value
        self.fixed_matrix_peak_value = fixed_peak_value
        self.minima_matrix_peak_value = minima_peak_value

    def max_steps_apf_descent_path_planning(self, N):
        self.max_step_apf_descent = N

    def set_attractive_constant(self, attractive_constant):
        self.attractive_constant = attractive_constant

    def set_max_speed(self, max_speed):
        self.max_speed = max_speed

    def min_obstacles_distance(self, proximity_sensor_range):
        self.proximity_sensor_range = proximity_sensor_range

    def import_map_properties(self, map_object):
        # collects data about the map (only for simulation purpose)
        self.map_dimension_x = map_object.x_dimension
        self.map_dimension_y = map_object.y_dimension
        self.map_dimension_z = map_object.z_dimension
        self.map_resolution_x = map_object.x_resolution
        self.map_resolution_y = map_object.y_resolution
        self.map_resolution_z = map_object.z_resolution
        self.obstacle_map = map_object.obstacle_map

    def initialize(self, n_drones,  **kwargs):
        # import map data if not already imported (requires kwarg "map_obj"
        if self.map_dimension_x is None or self.map_dimension_y is None or self.map_resolution_x is None or self.map_resolution_y is None:
            succesfull_import_flag = False
            for key, value in kwargs.items():
                if key == "map_obj":
                    self.import_map_properties(value)   # value = map_object
                    succesfull_import_flag = True
            if not succesfull_import_flag:
                raise Warning("Missing map object in drone_class\initialize")
        # note: remember to run "import_map_properties" previously in order to properly initialize all the matrices
        self.n_cell_x = int(round(self.map_dimension_x / self.map_resolution_x))    # number of cells along x_coordinate and y_coordinate (in the potential map)
        self.n_cell_y = int(round(self.map_dimension_y / self.map_resolution_y))
        self.n_cell_z = int(round(self.map_dimension_z / self.map_resolution_z))

        self.min_index_x = 0
        self.max_index_x = int(math.floor(self.map_dimension_x / self.map_resolution_x)) # alternative: self.n_cell_x-1
        self.min_index_y = 0
        self.max_index_y = int(math.floor(self.map_dimension_y / self.map_resolution_x)) # alternative: self.n_cell_y-1
        self.min_index_z = 0
        self.max_index_z = int(math.floor(self.map_dimension_z / self.map_resolution_z))
        #initialize position history vector
        self.position_history = np.zeros([1, 3])
        # self.position_history[0, :] = self.position
        # initialize potential world as matrix of zeros
        self.potential_field_map = np.zeros((self.n_cell_z, self.n_cell_x, self.n_cell_y))
        self.layer_attractive = np.zeros((self.n_cell_z, self.n_cell_x, self.n_cell_y))
        self.layer_experience = np.zeros((self.n_cell_z, self.n_cell_x, self.n_cell_y))
        self.covered_area_matrix = np.zeros((self.n_cell_z, self.n_cell_x, self.n_cell_y))
        self.obstacle_matrix = np.zeros((self.n_cell_z, self.n_cell_x, self.n_cell_y))
        self.total_explorable_elements = self.covered_area_matrix.size
        x = np.linspace(0, self.map_dimension_x, self.n_cell_x)
        # y = np.linspace(0, self.map_dimension_y, self.n_cell_y)
        z = np.linspace(0, self.map_dimension_z, self.n_cell_z)
        self.x_cohordinate_matrix = np.ones([self.n_cell_z, self.n_cell_x, self.n_cell_y]) * x
        self.y_cohordinate_matrix = self.x_cohordinate_matrix.transpose(0,2,1)
        self.z_cohordinate_matrix = np.ones([self.n_cell_z, self.n_cell_x, self.n_cell_y])
        for ii in range(self.n_cell_z):
            self.z_cohordinate_matrix[ii, :, :] *= z[ii]

        # indexes_x = np.arange(0, self.n_cell_x)
        # indexes_y = np.arange(0, self.n_cell_y)
        # self.column_indexes = indexes_y * np.ones([self.n_cell_x, 1])
        # self.row_indexes = (indexes_x * np.ones([self.n_cell_y, 1])).T
        # initialize the motion option for classic algorithm
        self.motion_options = np.array(np.meshgrid([-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2])).T.reshape(-1, 3)
        idx = int(np.where((self.motion_options == [0, 0, 0]).all(axis = 1))[0])                 # find index of the (0, 0) row in the list of possible motions
        self.motion_options = np.delete(self.motion_options, idx, axis=0)                        # axis=0 --> delete rows
        # initialize obstacle matrices
        self.compute_fixed_obstacle_matrix()    # for fixed obstacles
        self.compute_mobile_obstacle_matrices() # for mobile obstacles
        self.compute_local_minima_matrix()  # for local minima problem
        self.compute_spherical_spiral()
        self.n_drones = n_drones
        self.single_coverage.append((np.count_nonzero(self.covered_area_matrix) / n_drones)/ self.total_explorable_elements * 100)
        self.other_drones_current_position = np.empty((0, 5))   # initialize vector with other drones positions and velocities
        self.initialization_completed = True    # initialization completed

    def update_attractive_layer(self):
        x = 0
        y = 1
        z = 2
        self.potential_field_map -= self.layer_attractive   # removes contribute of the previous goal attractive field
        self.layer_attractive = self.attractive_constant * np.sqrt((self.x_cohordinate_matrix - self.goal[x])**2 + (self.y_cohordinate_matrix - self.goal[y])**2 + (self.z_cohordinate_matrix - self.goal[z])**2).transpose(0, 2, 1)
        self.potential_field_map += self.layer_attractive
        self.goal_updated = True

    def reset_local_minima_obstacles(self):
        self.potential_field_map -= self.layer_experience                           # remove the contribution of experience layer from potential field map
        self.layer_experience = np.zeros((self.n_cell_x, self.n_cell_y))            # reset the experience layer

    def get_other_drones_positions(self, drone_list):
        self.other_drones_current_position = np.empty((0, 5))       # initialize empty list of mobile obstacles position --> reset after subraction cycle
        for drone in drone_list:
            if drone.drone_ID != self.drone_ID:
                state = np.append(drone.position, drone.speed)
                state = np.append(state, drone.drone_ID)
                self.other_drones_current_position = np.vstack([self.other_drones_current_position, state])

    def share_new_obstacle_positions(self, drone_list):
        for drone in drone_list:
            if drone.drone_ID != self.drone_ID:
                drone.newfound_obstacle_list = np.vstack([drone.newfound_obstacle_list, self.newfound_obstacle_share_list])
                # drone.newfound_obstacle_list = np.unique(drone.newfound_obstacle_list, axis=0)
        self.newfound_obstacle_share_list = np.empty([0, 3])
        return

    def share_covered_area(self, drone_list):
        for drone in drone_list:
            if drone.drone_ID != self.drone_ID:
                drone.covered_area_matrix += self.covered_area_matrix
                drone.covered_area_matrix = np.clip(drone.covered_area_matrix, 0, 1)
        return

    def matrix_placer(self, main_matrix, matrix, index_position_x, index_position_y, index_position_z, sign):
        # M = small matrix to be placed
        # A = big matrix on which M has to be pasted
        # sign: +1 or -1 (+1 to paste, -1 to subtract)
        M_x_dimension, M_y_dimension = matrix.shape                 # NB: we assume that the dimension of M are odd numbers in both directions
        delta_x = (M_x_dimension-1) / 2
        delta_y = (M_y_dimension-1) / 2
        min_x = int(max(index_position_x - delta_x, self.min_index_x))   # min & max indexes indicating the portion of the big matrix in which M has to be pasted
        max_x = int(min(index_position_x + delta_x + 1, self.max_index_x))
        min_y = int(max(index_position_y - delta_y, self.min_index_y))
        max_y = int(min(index_position_y + delta_y + 1, self.max_index_y))
        min_x_A = int(delta_x - (index_position_x - min_x))
        max_x_A = int(delta_x + (max_x - index_position_x))
        min_y_A = int(delta_y - (index_position_y - min_y))
        max_y_A = int(delta_y + (max_y - index_position_y))
        main_matrix[index_position_z, min_x:max_x, min_y:max_y] += sign * matrix[min_x_A:max_x_A, min_y_A:max_y_A]

    def compute_mobile_obstacle_matrices(self):
        """
        computes the matrix containing the repulsive potential associated with the presence of a mobile obstacle (other drone)
        :returns: modifies the value of the "mobile_obstacle_matrix" property of each drone object
        """
        mean = 0
        n_points_x = int(round(self.drone_safe_distance / self.map_resolution_x))  # computes dimension of the repulsive matrix along x_coordinate
        n_points_y = int(round(self.drone_safe_distance / self.map_resolution_y))  # computes dimension of the repulsive matrix along y_coordinate
        n_points_x = math.floor(n_points_x / 2) * 2 + 1  # "trick" to make n_points_x to be odd every time
        n_points_y = math.floor(n_points_y / 2) * 2 + 1
        self.mobile_obstacle_matrix_dimensions = [n_points_x, n_points_y]   # saves info as [n° of rows, n° of columns] --> NOTE: bith numbers are ALWAYS odd
        N = 3                                                               # n° of different matrices that will be created --> different sizes of the repulsive area
        n = [3, 4.5, 7]                                                   # n-sigma to have 3 gaussians with 3 different widths
        self.mobile_obstacle_matrix = np.zeros([N, n_points_x, n_points_y]) # initialization of the 3D matrix containing all the repulsive matrices for drones
        for ii in range(0,N):
            sigma = 1/n[ii]
            x_vec = np.linspace(-1, 1, n_points_x, endpoint=True).reshape(n_points_x, 1)    # column vector
            y_vec = np.linspace(-1, 1, n_points_y, endpoint=True).reshape(1, n_points_y)    # row vector
            dist_x = np.array(stats.norm.pdf(x_vec, mean, sigma))   # normal distribution for x_coordinate direction
            dist_y = np.array(stats.norm.pdf(y_vec, mean, sigma))   # normal distribution for y_coordinate direction
            dist_x = dist_x - min(dist_x)                           # to have the border of the distribution = 0
            dist_y = dist_y - min(dist_y.T)                           # same as line above
            dist_x = dist_x * math.sqrt(self.mobile_matrix_peak_value)/max(dist_x)     # rescale the distribution to match the desired max value
            dist_y = dist_y * math.sqrt(self.mobile_matrix_peak_value)/max(dist_y.T)     #  NOTE: max could be the wrong function to be used here (and also in the line above)
            self.mobile_obstacle_matrix[ii,:,:] = dist_x*dist_y

    def compute_fixed_obstacle_matrix(self):
        """
        computes the matrix containing the repulsive potential associated with the presence of a fixed obstacle
        :returns: modifies the value of the "fixed_obstacle_matrix" property of each drone object
        """
        mean = 0
        sigma = 1/3         # variance --> increase the denominator to obtain a smaller repulsive area around the obstacle
        n_points_x = int(round(self.obstacle_safe_distance/self.map_resolution_x))      # computes dimension of the repulsive matrix along x_coordinate
        n_points_y = int(round(self.obstacle_safe_distance/self.map_resolution_y))      # computes dimension of the repulsive matrix along y_coordinate
        n_points_x = math.floor(n_points_x/2)*2 + 1                                     # "trick" to make n_points_x to be odd every time
        n_points_y = math.floor(n_points_y/2)*2 + 1
        self.fixed_obstacle_matrix_dimensions = [n_points_x, n_points_y]                # saves info as [n° of rows, n° of columns] --> NOTE: bith numbers are ALWAYS odd
        x_vec = np.linspace(-1, 1, n_points_x, endpoint=True).reshape(n_points_x, 1)    # column vector
        y_vec = np.linspace(-1, 1, n_points_y, endpoint=True).reshape(1, n_points_y)    # row vector
        dist_x = np.array(stats.norm.pdf(x_vec, mean, sigma))                           # normal distribution for x_coordinate direction
        dist_y = np.array(stats.norm.pdf(y_vec, mean, sigma))                           # normal distribution for y_coordinate direction
        dist_x = dist_x - min(dist_x)                                                   # to have the border of the distribution = 0
        dist_y = dist_y - min(dist_y.T)                                                 # same as line above
        dist_x = dist_x * math.sqrt(self.fixed_matrix_peak_value)/max(dist_x)           # rescale the distribution to match the desired max value
        dist_y = dist_y * math.sqrt(self.fixed_matrix_peak_value)/max(dist_y.T)         #  NOTE: max could be the wrong function to be used here (and also in the line above)
        self.fixed_obstacle_matrix = dist_x * dist_y                                    # the result is a matrix with n_points_x columns (x_coordinate axis) and n_points_y rows (y_coordinate axis)

    def compute_local_minima_matrix(self):
        mean = 0
        sigma = 1 / 3                                                                       # variance --> increase the denominator to obtain a smaller repulsive area around the obstacle
        n_points_x = int(round(self.minima_obstacle_dimension / self.map_resolution_x))     # computes dimension of the repulsive matrix along x_coordinate
        n_points_y = int(round(self.minima_obstacle_dimension / self.map_resolution_y))     # computes dimension of the repulsive matrix along y_coordinate
        n_points_x = math.floor(n_points_x / 2) * 2 + 1                                     # "trick" to make n_points_x to be odd every time
        n_points_y = math.floor(n_points_y / 2) * 2 + 1
        x_vec = np.linspace(-1, 1, n_points_x, endpoint=True).reshape(n_points_x, 1)        # column vector
        y_vec = np.linspace(-1, 1, n_points_y, endpoint=True).reshape(1, n_points_y)        # row vector
        dist_x = np.array(stats.norm.pdf(x_vec, mean, sigma))                               # normal distribution for x_coordinate direction
        dist_y = np.array(stats.norm.pdf(y_vec, mean, sigma))                               # normal distribution for y_coordinate direction
        dist_x = dist_x - min(dist_x)                                                       # to have the border of the distribution = 0
        dist_y = dist_y - min(dist_y.T)                                                     # same as line above
        dist_x = dist_x * math.sqrt(self.minima_matrix_peak_value) / max(dist_x)            # rescale the distribution to match the desired max value
        dist_y = dist_y * math.sqrt(self.minima_matrix_peak_value) / max(dist_y.T)          # NOTE: max could be the wrong function to be used here (and also in the line above)
        self.local_minima_matrix = dist_x * dist_y                                          # matrix for overcome local minima

    def compute_spherical_spiral(self):
        a = 0.05
        t = np.linspace(150, 0, 100)
        x_spheric_spiral_points = self.max_speed*3 * np.cos(t)*np.cos(np.arctan(a*t))
        y_spheric_spiral_points = self.max_speed*3 * np.sin(t)*np.cos(np.arctan(a*t))
        z_spheric_spiral_points = self.max_speed * np.sin(np.arctan(a*t))
        self.spheric_spiral_coordinates = np.array([x_spheric_spiral_points, y_spheric_spiral_points, z_spheric_spiral_points])

    def rotate_spiral(self, theta, phi):
        rot_z = np.array([[math.cos(theta), -math.sin(theta), 0],[math.sin(theta), math.cos(theta), 0],[0, 0, 1]])
        rot_y = np.array([[math.cos(phi), 0, math.sin(phi)],[0, 1, 0],[-math.sin(theta), 0, math.cos(theta)]])
        rotate_coordinates = np.matmul(np.matmul(rot_z, rot_y), self.spheric_spiral_coordinates)
        first_x = round(self.max_speed * math.sin(phi) * math.cos(theta), 2)
        first_y = - round(self.max_speed * math.sin(phi) * math.sin(theta), 2)
        first_z = round(self.max_speed * math.cos(phi), 2)
        rotate_coordinates = np.insert(rotate_coordinates, 0, [first_x, first_y, first_z], axis=1)
        return rotate_coordinates

    def update_fixed_obstacles(self, drone_list):
        self.newfound_obstacle_list = np.unique(self.newfound_obstacle_list, axis=0)
        for obstacle_position_x, obstacle_position_y, obstacle_position_z in self.newfound_obstacle_list:                # runs over the whole list of newly detected obstacles
            x_idx = int(math.floor(obstacle_position_x/self.map_resolution_x))    # index of x_coordinate-cell corresponding to obstacle position
            y_idx = int(math.floor(obstacle_position_y/self.map_resolution_y))    # index of y_coordinate-cell corresponding to obstacle position
            z_idx = int(math.floor(obstacle_position_z/self.map_resolution_z))
            self.obstacle_matrix[z_idx, x_idx, y_idx] = 1
            if self.check_fixed_obstacle_position(obstacle_position_x, obstacle_position_y, obstacle_position_z):
                self.matrix_placer(self.potential_field_map, self.fixed_obstacle_matrix, x_idx, y_idx, z_idx, +1)  # add the repulsive matrix to the global APF matrix
                self.obstacle_list = np.vstack([self.obstacle_list, np.array([obstacle_position_x, obstacle_position_y, obstacle_position_z])])  # append the newfound obstacles to the complete obstacle list
                self.newfound_obstacle_share_list = np.vstack([self.newfound_obstacle_share_list, [obstacle_position_x, obstacle_position_y, obstacle_position_z]])
        self.share_new_obstacle_positions(drone_list)  # update the obstacles found in this iteration
        self.obstacle_list = np.unique(self.obstacle_list, axis=0)
        if len(self.obstacle_list) >= self.obstacle_list_security_length:
            raise Warning("Memory limit reached for obstacle list - obstacle list can not be updated")
        self.newfound_obstacle_list = np.empty([0, 3])    # empty newfound obstacle list -> it will be replenished during the next scanning phase

    def update_mobile_obstacles(self):
        for x_idx, y_idx, z_idx, ii in self.mobile_obstacle_last_position:
            self.matrix_placer(self.potential_field_map, self.mobile_obstacle_matrix[int(ii),:,:], x_idx, y_idx, z_idx, -1)
        self.mobile_obstacle_last_position = np.empty([0,4])
        # d0 = 0.2 * self.drone_safe_distance                        
        for x, y, z, vx, vy, vz, _ in self.other_drones_current_position:
            theta = math.atan2(vy, vx)
            x_idx = int(math.floor((x + self.max_speed * math.cos(theta)) / self.map_resolution_x))
            y_idx = int(math.floor((y + self.max_speed * math.sin(theta)) / self.map_resolution_y))
            z_idx = int(math.floor((z + self.max_speed * math.sin(theta)) / self.map_resolution_z))
            self.matrix_placer(self.potential_field_map, self.mobile_obstacle_matrix[0, :, :], x_idx, y_idx, z_idx, +1)
            self.mobile_obstacle_last_position = np.vstack([self.mobile_obstacle_last_position, [x_idx, y_idx, z_idx, 0]])
            # theta = math.atan2(vy, vx)                                 # direction of movement
            # norm_v = np.hypot(vx, vy)                                  # magnitude of the speed
            # d1 = d0 * norm_v/self.max_speed                            # linear law
            # d2 = (d0 + 0.3) * norm_v/self.max_speed                    # distance between 2nd and 3rd circles is a bit smaller
            # d_vec = [0, d1, d2]
            # for ii in range(0, len(d_vec)):
                # x_idx = int(math.floor((x + d_vec[ii] * math.cos(theta)) / self.map_resolution_x))  # compute index of the cell containing the drone
                # y_idx = int(math.floor((y + d_vec[ii] * math.sin(theta)) / self.map_resolution_y))
                # self.matrix_placer(self.potential_field_map, self.mobile_obstacle_matrix[int(ii),:,:], x_idx, y_idx, +1)
                # self.mobile_obstacle_last_position = np.vstack([self.mobile_obstacle_last_position, [x_idx, y_idx, ii]])

    def generate_border_obstacles(self):
        nx = int(round(self.map_dimension_x / self.map_resolution_x))         # n° of points along x_coordinate
        ny = int(round(self.map_dimension_y / self.map_resolution_y))         # n° of points along y_coordinate
        nz = int(round(self.map_dimension_z / self.map_resolution_z))  # n° of points along y_coordinate
        mask = np.ones([ny, nx], dtype=bool)
        mask_zx = np.ones([nz, nx], dtype=bool)
        mask_zy = np.ones([nz, ny], dtype=bool)
        x = np.linspace(0, self.map_dimension_x-0.001, self.n_cell_x)
        y = np.linspace(0, self.map_dimension_y-0.001, self.n_cell_y)
        z = np.linspace(0, self.map_dimension_z-0.001, self.n_cell_z)
        x_cohordinate_matrix, y_cohordinate_matrix = np.meshgrid(x, y)
        x_cohordinate_matrix_2, zx_cohordinate_matrix = np.meshgrid(x, z)
        y_cohordinate_matrix_2, zy_cohordinate_matrix = np.meshgrid(y, z)
        obstacle_position_x = np.reshape(x_cohordinate_matrix[mask], [-1, 1])
        obstacle_position_y = np.reshape(y_cohordinate_matrix[mask], [-1, 1])
        obstacle_position_zy = np.reshape(zy_cohordinate_matrix[mask_zy], [-1, 1])
        obstacle_position_zx = np.reshape(zx_cohordinate_matrix[mask_zx], [-1, 1])
        obstacle_position_x_2 = np.reshape(x_cohordinate_matrix_2[mask_zx], [-1, 1])
        obstacle_position_y_2 = np.reshape(y_cohordinate_matrix_2[mask_zy], [-1, 1])

        top_obstacles = np.hstack([obstacle_position_x, obstacle_position_y, np.zeros([len(obstacle_position_x), 1])])
        bot_obstacles = np.hstack([obstacle_position_x, obstacle_position_y, np.ones([len(obstacle_position_x), 1])*self.map_dimension_z-0.001])
        side_R_obstacles = np.hstack([np.ones([len(obstacle_position_y_2), 1])*self.map_dimension_x-0.001, obstacle_position_y_2, obstacle_position_zy])
        side_L_obstacles = np.hstack([np.zeros([len(obstacle_position_y_2), 1]), obstacle_position_y_2, obstacle_position_zy])
        side_F_obstacles = np.hstack([obstacle_position_x_2, np.ones([len(obstacle_position_x_2), 1])*self.map_dimension_x-0.001, obstacle_position_zx])
        side_B_obstacles = np.hstack([obstacle_position_x_2, np.zeros([len(obstacle_position_x_2), 1]), obstacle_position_zx])

        border_obstacles = np.vstack([top_obstacles, bot_obstacles, side_R_obstacles, side_L_obstacles, side_F_obstacles, side_B_obstacles])
        self.obstacle_matrix[0, :, :] = 1
        self.obstacle_matrix[:, 0, :] = 1
        self.obstacle_matrix[:, :, 0] = 1
        self.obstacle_matrix[-1, :, :] = 1
        self.obstacle_matrix[:, -1, :] = 1
        self.obstacle_matrix[:, :, -1] = 1
        self.covered_area_matrix[0, :, :] = 1
        self.covered_area_matrix[:, 0, :] = 1
        self.covered_area_matrix[:, :, 0] = 1
        self.covered_area_matrix[-1, :, :] = 1
        self.covered_area_matrix[:, -1, :] = 1
        self.covered_area_matrix[:, :, -1] = 1
        self.newfound_obstacle_list = np.append(self.newfound_obstacle_list, border_obstacles, axis=0)

    def check_fixed_obstacle_position(self, x_idx, y_idx, z_idx):
        found_idx = np.where((self.obstacle_list == [z_idx, x_idx, y_idx]).all(axis=1))[0]          # check for obstacles in the same position along x_coordinate (1st column of the obstacle list array)
        if found_idx.size > 0:  # if there is any match i.e. there is already an obstacle in the same position, a "False" flag is returned and obstacle is ignored in "update_fixed_obstacles"
            return False
        else:
            return True
        # alternative: return found_idx_size > 0

    def check_collision(self, drone_list):
        if self.obstacle_map[self.index_position[2], self.index_position[0], self.index_position[1]] == 1:
            print('Drone ' + str(self.drone_ID) + ' is crasched')
            for drone in drone_list:
                self.id_list.append(drone.drone_ID)
            drone_index = self.id_list.index(self.drone_ID)
            drone_list.pop(drone_index)

            if len(drone_list) == 0:
                print('END OF SIMULATION')
                time.sleep(3)
                exit(0)
        return

    def check_goal(self):
        idx_x = int(self.pos2index(self.goal[0]))
        idx_y = int(self.pos2index(self.goal[1]))
        idx_z = int(self.pos2index(self.goal[2]))
        if self.potential_field_map[idx_z, idx_x, idx_y] > 300 or self.goal_steps > self.max_steps_single_goal:
            self.goal_updated = False
            self.local_minima_flag = False
            self.goal_steps = 0
        else:
            self.goal_steps += 1
        return

    def potential_field_path_planning(self):
        """
        ADVANCED POTENTIAL FIELD PATH PLANNING
        The basic idea is to privilege the movement toward the objective rather than the movement in the direction of the lowest
        possible potential. To achieve this objective, the drone tries to move directly toward the objective. If the potential
        that direction is lower (minus an optional offset) than the current potential, then the drone moves without exploring
        any other movement possibility. Otherwise, it moves 1° to the right, then 1° to the left and so on until it finds a
        suitable movement direction.
        """
        x = 0
        y = 1
        z = 2
        self.potential_field_sweep_steps = 0
        if np.linalg.norm(self.goal-self.position) < self.min_goal_distance:                               # objective has been found (inside the same cell of the drone)
            self.speed[x] = 0
            self.speed[y] = 0
            self.speed[z] = 0
            self.position[x], self.position[y], self.position[z] = self.goal[x], self.goal[y], self.goal[z]
            self.goal_updated = False
        else:
            current_cell_idx_x = int(math.floor(self.position[x] / self.map_resolution_x))
            current_cell_idx_y = int(math.floor(self.position[y] / self.map_resolution_y))
            current_cell_idx_z = int(math.floor(self.position[z] / self.map_resolution_z))
            current_potential = self.potential_field_map[current_cell_idx_z, current_cell_idx_x, current_cell_idx_y]
            potential_offset = 0
            sweep = 0
            theta, phi = self.cartesian2spheric()
            print(math.degrees(theta), math.degrees(phi))
            coordinates = self.rotate_spiral(theta, phi)
            while sweep<100:
                new_position_x = self.position[x] + coordinates[x, sweep]
                new_position_y = self.position[y] + coordinates[y, sweep]
                new_position_z = self.position[z] + coordinates[z, sweep]
                new_cell_idx_x = self.pos2index(new_position_x)
                new_cell_idx_y = self.pos2index(new_position_y)
                new_cell_idx_z = self.pos2index(new_position_z)
                new_potential = self.potential_field_map[new_cell_idx_z, new_cell_idx_x, new_cell_idx_y]
                if new_potential <= (current_potential + potential_offset):
                    self.orientation[0] = theta
                    self.orientation[1] = phi
                    self.speed[x] = round(self.max_speed * math.sin(phi) * math.cos(theta), 2)
                    self.speed[y] = round(self.max_speed * math.sin(phi) * math.sin(theta), 2)
                    self.speed[z] = round(self.max_speed * math.cos(phi), 2)
                    self.position[x] = new_position_x
                    self.position[y] = new_position_y
                    self.position[z] = new_position_z
                    self.position_history = np.round(np.vstack([self.position_history, self.position]), 2)
                    self.index_position[x] = self.pos2index(self.position[x])
                    self.index_position[y] = self.pos2index(self.position[y])
                    self.index_position[z] = self.pos2index(self.position[z])
                        # self.interpolate_trajectory()
                        # if self.oscillation_detection():
                        #     # oscillation detected
                        #     self.local_minima_flag = True                       # switch to the other algorithm
                        #     self.step_counter = 0                               # reset the step counter for classic algorithm
                    return
                else:
                    sweep += 1

            self.position[x], self.position[y], self.position[z] = self.position[x], self.position[y], self.position[z]         # if no suitable movement direction has been found, the drone stays still (see line below)
            self.speed[x], self.speed[y], self.speed[z] = 0, 0, 0
            self.local_minima_flag = True               # switch to the other algorithm
            self.step_counter = 0
            self.position_history = np.vstack([self.position_history, self.position])
            if len(self.position_history) > self.position_list_security_length:
                self.position_history = np.delete(self.position_history, 0, axis=0)
            return

    def emergency_path_planning(self):
        # search best movement direction (~ optimization process)
        x = 0  # index 0 is assigned to the variable x_coordinate to help readability
        y = 1  # same as above
        z = 2
        idx_current_x = math.floor(self.position[x] / self.map_resolution_x)            # x_coordinate index of the current drone cell (inside the potential matrix)
        idx_current_y = math.floor(self.position[y] / self.map_resolution_y)            # y_coordinate index of the current drone cell (inside the potential matrix)
        idx_current_z = math.floor(self.position[z] / self.map_resolution_z)
        min_nearby_potential = float("inf")                                             # initialize minimum potential found in the nearby cells
        idx_min_nearby_potential_x, idx_min_nearby_potential_y, idx_min_nearby_potential_z = -1, -1, -1                 # initialize indexes representing nearby cell where the minimum potential is found
        for i in range(0, len(self.motion_options)):
            idx_new_x = int(idx_current_x + self.motion_options[i][x])
            idx_new_y = int(idx_current_y + self.motion_options[i][y])
            idx_new_z = int(idx_current_z + self.motion_options[i][z])
            if idx_new_x >= (self.map_dimension_x / self.map_resolution_x) or \
                    idx_new_y >= (self.map_dimension_y / self.map_resolution_y) or \
                    idx_new_z >= (self.map_dimension_z / self.map_resolution_z) or \
                    idx_new_x < 0 or idx_new_y < 0 or idx_new_z < 0:
                nearby_potential = float("inf")                                         # outside area
            else:
                nearby_potential = self.potential_field_map[idx_new_z, idx_new_x, idx_new_y]
            if nearby_potential < min_nearby_potential:                                 # if the potential of the nearby cell currently being examined is
                                                                                        # lower than the previous best one, the new one is selected for
                                                                                        # the best movement direction
                min_nearby_potential = nearby_potential
                idx_min_nearby_potential_x = idx_new_x
                idx_min_nearby_potential_y = idx_new_y
                idx_min_nearby_potential_z = idx_new_z
        self.position[x] = idx_min_nearby_potential_x * self.map_resolution_x
        self.position[y] = idx_min_nearby_potential_y * self.map_resolution_y
        self.position[z] = idx_min_nearby_potential_z * self.map_resolution_z
        self.index_position[x] = self.pos2index(self.position[x])
        self.index_position[y] = self.pos2index(self.position[y])
        self.index_position[z] = self.pos2index(self.position[z])
        # self.interpolate_trajectory()
        # if self.oscillation_detection():
        #     self.local_minima_flag = True                   # oscillation detected switch to the other algorithm
        #     self.step_counter = 0
        theta, phi = self.cartesian2spheric()
        self.speed[x] = round(self.max_speed * math.sin(phi) * math.cos(theta), 2)
        self.speed[y] = round(self.max_speed * math.sin(phi) * math.sin(theta), 2)
        self.speed[z] = round(self.max_speed * math.cos(phi), 2)
        self.position_history = np.vstack([self.position_history, self.position])
        # if self.emergency_step_counter >= 3:
        #     self.emergency_path_planning_flag = False
        # else:
        #     self.emergency_step_counter += 1
        # return

    # def potential_field_path_planning_classic(self):
    #     # search best movement direction (~ optimization process)
    #     x = 0  # index 0 is assigned to the variable x_coordinate to help readability
    #     y = 1  # same as above
    #     idx_current_x = math.floor(self.position[x] / self.map_resolution_x)            # x_coordinate index of the current drone cell (inside the potential matrix)
    #     idx_current_y = math.floor(self.position[y] / self.map_resolution_y)            # y_coordinate index of the current drone cell (inside the potential matrix)
    #     min_nearby_potential = float("inf")                                             # initialize minimum potential found in the nearby cells
    #     idx_min_nearby_potential_x, idx_min_nearby_potential_y = -1, -1                 # initialize indexes representing nearby cell where the minimum potential is found
    #     for i in range(0, len(self.motion_options)):
    #         idx_new_x = int(idx_current_x + self.motion_options[i][x])
    #         idx_new_y = int(idx_current_y + self.motion_options[i][y])
    #         if idx_new_x >= (self.map_dimension_x / self.map_resolution_x) or \
    #                 idx_new_y >= (self.map_dimension_y / self.map_resolution_y) or \
    #                 idx_new_x < 0 or idx_new_y < 0:
    #             nearby_potential = float("inf")                                         # outside area
    #         else:
    #             nearby_potential = self.potential_field_map[idx_new_x][idx_new_y]
    #         if nearby_potential < min_nearby_potential:                                 # if the potential of the nearby cell currently being examined is
    #                                                                                     # lower than the previous best one, the new one is selected for
    #                                                                                     # the best movement direction
    #             min_nearby_potential = nearby_potential
    #             idx_min_nearby_potential_x = idx_new_x
    #             idx_min_nearby_potential_y = idx_new_y
    #     self.position[x] = idx_min_nearby_potential_x * self.map_resolution_x
    #     self.position[y] = idx_min_nearby_potential_y * self.map_resolution_y
    #     self.index_position[x] = self.pos2index(self.position[x])
    #     self.index_position[y] = self.pos2index(self.position[y])
    #     self.get_orientation(idx_min_nearby_potential_x - idx_current_x, idx_min_nearby_potential_y - idx_current_y)
    #     self.speed[x] = round(self.max_speed * math.cos(self.orientation), 2)
    #     self.speed[y] = round(self.max_speed * math.sin(self.orientation), 2)
    #     self.step_counter += 1
    #     self.matrix_placer(self.potential_field_map, self.local_minima_matrix, idx_min_nearby_potential_x, idx_min_nearby_potential_y, 1)       # obstacle placed in potential field matrix
    #     self.matrix_placer(self.layer_experience, self.local_minima_matrix, idx_min_nearby_potential_x, idx_min_nearby_potential_y, 1)          # matrix placed also in experience matrix
    #     if self.step_counter >= self.max_step_apf_descent:                                                          # condition for switching to the main algorithm
    #         self.local_minima_flag = False                                          # exit from the classic verison of the algorithm
    #     self.position_history = np.vstack([self.position_history, self.position])
    #     return

    def reinforcement_learning_path_planning(self):
        x = 0
        y = 1
        z = 2
        if np.linalg.norm(np.array(self.goal) - np.array(self.position)) < self.min_goal_distance:                               # objective has been found (inside the same cell of the drone)
            self.speed[x] = 0
            self.speed[y] = 0
            self.speed[z] = 0
            self.position[x], self.position[y], self.position[z] = self.goal[x], self.goal[y], self.goal[z]
            self.goal_updated = False
        else:
            state = self.observe()
            state = tf.expand_dims(tf.convert_to_tensor(state), 0)
            theta = self.path_planning_model(state)
            phi = math.radians(90)
            self.orientation[0] = theta
            self.orientation[1] = phi
            self.speed[x] = round(self.max_speed * math.sin(phi) * math.cos(theta), 2)
            self.speed[y] = round(self.max_speed * math.sin(phi) * math.sin(theta), 2)
            self.speed[z] = round(self.max_speed * math.cos(phi), 2)
            # self.interpolate_trajectory()
            position_x = self.check_boundaries(self.position[x] + self.speed[x], 'x_coordinate')
            position_y = self.check_boundaries(self.position[y] + self.speed[y], 'y_coordinate')
            position_z = self.check_boundaries(self.position[z] + self.speed[z], 'z_coordinate')
            self.position = [position_x, position_y, position_z]
            self.index_position[x] = self.pos2index(self.position[x])
            self.index_position[y] = self.pos2index(self.position[y])
            self.index_position[z] = self.pos2index(self.position[z])
            # if self.oscillation_detection():
            #     self.local_minima_flag = True                           # oscillation detected -> switch to the other algorithm
            #     self.step_counter = 0

            self.position_history = np.vstack([self.position_history, self.position])
            if len(self.position_history) > self.position_list_security_length:
                self.position_history = np.delete(self.position_history, 0, axis=0)

    def coverage_random(self, z_position):
        location_found = False
        repeat_counter = 0
        # goal = np.array([np.random.random() * (self.map_dimension_x - 1), np.random.random() * (self.map_dimension_y - 1), np.random.random() * (self.map_dimension_z - 1)])
        goal = np.array([np.random.random() * (self.map_dimension_x - 1), np.random.random() * (self.map_dimension_y - 1), z_position])
        while not location_found:
            repeat_counter += 1
            goal = np.array([np.random.random() * (self.map_dimension_x - 1), np.random.random() * (self.map_dimension_y - 1), z_position])
            idx_x = int(self.pos2index(goal[0]))
            idx_y = int(self.pos2index(goal[1]))
            idx_z = int(self.pos2index(goal[2]))
            if self.potential_field_map[idx_z, idx_x, idx_y] < 50:
                location_found = True
            elif repeat_counter > 5:
                break
        new_goal = np.array(goal, dtype="f")
        self.old_goal = self.goal               # stores previous goal in memory
        self.goal = new_goal

    def coverage_voronoi(self, drone_list):
        unknown_points = np.vstack((self.x_cohordinate_matrix[self.covered_area_matrix==0], self.y_cohordinate_matrix[self.covered_area_matrix==0])).T
        kmeans = KMeans(n_clusters=self.n_drones, random_state=0).fit(unknown_points)
        exploration_area_centers = kmeans.cluster_centers_

        self_state = np.append(self.position, self.speed)
        self_state = np.append(self_state, self.drone_ID)
        drones_positions = np.vstack((self.other_drones_current_position, self_state))
        x_distances = np.array([drones_positions[:, 0]]).T - np.array([exploration_area_centers[:, 0]])
        y_distances = np.array([drones_positions[:, 1]]).T - np.array([exploration_area_centers[:, 1]])
        distances = np.hypot(x_distances, y_distances)
        for goal_idx in range(0, 4):
            selected_column = distances[:, goal_idx]
            idx = int(np.where(selected_column==np.min(selected_column))[0])
            drone_list[idx].old_goal = drone_list[idx].goal                           # stores previous goal in memory
            drone_list[idx].goal = exploration_area_centers[goal_idx, :]

    def oscillation_detection(self):
        if len(self.position_history) > 15:
            unique_positions = np.unique(np.floor(self.position_history[-16:-1] / self.map_resolution_x), axis=0)
            if len(unique_positions) < 10:                           # if the number of repeated position is higher than a treshold the oscillation flag is set true
                return True
            else:
                return False
        else:
            return False

    def detect_obstacles(self):
        x = 0
        y = 1
        z = 2
        cell_vision_range = self.vision_range / self.map_resolution_x
        previous_value = np.count_nonzero(self.covered_area_matrix)
        for theta_angle in np.linspace(-self.max_vision_angle, self.max_vision_angle, self.vision_resolution):
            theta_angle_rad = math.radians(theta_angle)
            theta = self.orientation[0] + theta_angle_rad
            for phi_angle in np.linspace(-self.max_vision_angle, self.max_vision_angle, self.vision_resolution):
                distance = 0
                phi_angle_rad = math.radians(phi_angle)
                phi = self.orientation[1] + phi_angle_rad
                while distance <= cell_vision_range:
                    idx_x = int(math.floor(self.index_position[x] + distance * math.sin(phi) * math.cos(theta)))
                    idx_y = int(math.floor(self.index_position[y] + distance * math.sin(phi) * math.sin(theta)))
                    idx_z = int(math.floor(self.index_position[z] + distance * math.cos(phi)))
                    self.covered_area_matrix[idx_z, idx_x, idx_y] = 1              # fill the coverage matrix in the explored area
                    if self.obstacle_map[idx_z, idx_x, idx_y] == 1:
                        self.obstacle_matrix[idx_z, idx_x, idx_y] = 1
                        self.covered_area_matrix[idx_z, idx_x, idx_y] = 1          # fill the coverage matrix in the obstacles
                        obstacle_position_x = idx_x * self.map_resolution_x
                        obstacle_position_y = idx_y * self.map_resolution_y
                        obstacle_position_z = idx_z * self.map_resolution_z
                        self.newfound_obstacle_list = np.vstack([self.newfound_obstacle_list, [obstacle_position_x, obstacle_position_y, obstacle_position_z]])
                        break
                    else:
                        distance += 1
        # self.detect_close_obstacles()
        next_value = np.count_nonzero(self.covered_area_matrix)
        previous_coverage_increment = self.single_coverage[-1] * self.total_explorable_elements / 100
        single_percentage = (previous_coverage_increment + (next_value - previous_value)) / self.total_explorable_elements * 100
        self.single_coverage.append(single_percentage)
        return

    def proximity_sensor(self):
        x = 0
        y = 1
        z = 2
        proximity_sensor_cells_range = self.proximity_sensor_range / self.map_resolution_x
        for theta_angle in np.linspace(-180, 180, self.proximity_sensor_resolution):
            theta_angle_rad = math.radians(theta_angle)
            theta = self.orientation[0] + theta_angle_rad
            for phi_angle in np.linspace(-180, 180, self.proximity_sensor_resolution):
                distance = 0
                phi_angle_rad = math.radians(phi_angle)
                phi = self.orientation[1] + phi_angle_rad
                while distance <= proximity_sensor_cells_range:
                    idx_x = int(math.floor(self.index_position[x] + distance * math.sin(phi) * math.cos(theta)))
                    idx_y = int(math.floor(self.index_position[y] + distance * math.sin(phi) * math.sin(theta)))
                    idx_z = int(math.floor(self.index_position[z] + distance * math.cos(phi)))
                    if self.obstacle_map[idx_z, idx_x, idx_y] == 1:
                        self.emergency_path_planning_flag = True
                        self.emergency_step_counter = 0
                        break
                    else:
                        distance += 1
        return

    def lidar_operation_mode(self):
        self.saved_vision_angle = self.max_vision_angle
        self.saved_vision_resolution = self.vision_resolution
        self.max_vision_angle = 180
        self.vision_resolution = 200
        return

    def normal_operation_mode(self):
        self.max_vision_angle = self.saved_vision_angle
        self.vision_resolution = self.saved_vision_resolution
        return

    def cartesian2spheric(self):
        x = 0
        y = 1
        z = 2
        distance_x = self.goal[x] - self.position[x]
        distance_y = self.goal[y] - self.position[y]
        distance_z = self.goal[z] - self.position[z]
        rho = math.sqrt(distance_x**2 + distance_y**2 + distance_z**2)
        phi = math.acos(distance_z/rho)
        # theta = math.atan(distance_y / (distance_x + 0.001))
        if distance_x > 0 and distance_y >= 0:
            theta = math.atan(distance_y / (distance_x + 0.001))
        elif (distance_x > 0 and distance_y < 0) or (distance_x < 0 and distance_y > 0):
            theta = math.atan(distance_y / (distance_x + 0.001)) + 2*math.pi
        else:
            theta = math.atan(distance_y / (distance_x + 0.001)) + math.pi
        if theta >= math.radians(360):
            theta -= math.radians(360)
        if phi > math.radians(180):
            phi = (math.radians(180) - (phi-math.radians(180)))
        elif phi <= 0:
            phi = abs(phi)
        return theta, phi

    def get_orientation(self, delta_x, delta_y):
        self.orientation = math.atan2(delta_y, delta_x)
        if self.orientation > 2*math.pi:
            self.orientation = self.orientation - 2*math.pi
        elif self.orientation < 0:
            self.orientation = self.orientation + 2*math.pi
        return

    def observe(self):
        x = 0
        y = 1
        z = 2

        observation = np.ones([self.observation_size, self.observation_size])*1000
        delta = (self.observation_size - 1) / 2

        min_x = int(max(self.index_position[x] - delta, self.min_index_x))  # min & max indexes indicating the portion of the big matrix in which M has to be pasted
        max_x = int(min(self.index_position[x] + delta + 1, self.max_index_x))
        min_y = int(max(self.index_position[y] - delta, self.min_index_y))
        max_y = int(min(self.index_position[y] + delta + 1, self.max_index_y))
        min_x_A = int(delta - (self.index_position[x] - min_x))
        max_x_A = int(delta + (max_x - self.index_position[x]))
        min_y_A = int(delta - (self.index_position[y] - min_y))
        max_y_A = int(delta + (max_y - self.index_position[y]))
        observation[min_x_A:max_x_A, min_y_A:max_y_A] = self.potential_field_map[self.index_position[z], min_x:max_x, min_y:max_y]
        observation = np.clip(observation, 0, 1000)
        observation = observation - observation[int(self.observation_size - delta - 1), int(self.observation_size - delta - 1)]
        observation /= 1000
        observation = np.round(observation, 2)
        return observation

    def predict(self, position_x, position_y):
        x = 0
        y = 1
        z = 2
        index_position_x = self.pos2index(position_x)
        index_position_y = self.pos2index(position_y)
        observation = np.ones([self.observation_size, self.observation_size])*1000
        delta = (self.observation_size - 1) / 2

        min_x = int(max(index_position_x - delta, self.min_index_x))  # min & max indexes indicating the portion of the big matrix in which M has to be pasted
        max_x = int(min(index_position_x + delta + 1, self.max_index_x))
        min_y = int(max(index_position_y - delta, self.min_index_y))
        max_y = int(min(index_position_y + delta + 1, self.max_index_y))
        min_x_A = int(delta - (index_position_x - min_x))
        max_x_A = int(delta + (max_x - index_position_x))
        min_y_A = int(delta - (index_position_y - min_y))
        max_y_A = int(delta + (max_y - index_position_y))
        observation[min_x_A:max_x_A, min_y_A:max_y_A] = self.potential_field_map[self.index_position[z], min_x:max_x, min_y:max_y]
        observation = np.clip(observation, 0, 1000)
        observation = observation - observation[int(self.observation_size - delta - 1), int(self.observation_size - delta - 1)]
        observation /= 1000
        observation = np.round(observation, 2)
        return observation

    def interpolate_trajectory(self):
        x = 0
        y = 1
        polynomial_degree = 2                           # polynomial degree of the interpolation
        steps = 3                                       # number of skipped points
        predict_increment = self.max_speed
        control_points_x = []
        control_points_y = []
        position_x = self.position[x]
        position_y = self.position[y]
        for ii in range(0, self.predict_length*steps):
            if ii % steps == 0:
                state = self.predict(position_x, position_y)
                state = tf.expand_dims(tf.convert_to_tensor(state), 0)
                goal_distance = np.round(abs(np.array([self.position[x], self.position[y]]) - np.array([self.goal[x], self.goal[y]])), 2)/[self.map_dimension_x, self.map_dimension_y]
                goal_distance_tf = tf.expand_dims(tf.convert_to_tensor(goal_distance), 0)
                angle = self.path_planning_model([state, goal_distance_tf])
                position_x = self.check_boundaries(position_x + predict_increment * math.cos(angle), 'x_coordinate')
                position_y = self.check_boundaries(position_y + predict_increment * math.sin(angle), 'y_coordinate')
                control_points_x.append(position_x)
                control_points_y.append(position_y)
            else:
                state = self.predict(position_x, position_y)
                state = tf.expand_dims(tf.convert_to_tensor(state), 0)
                goal_distance = np.round(abs(np.array([self.position[x], self.position[y]]) - np.array([self.goal[x], self.goal[y]])), 2)/[self.map_dimension_x, self.map_dimension_y]
                goal_distance_tf = tf.expand_dims(tf.convert_to_tensor(goal_distance), 0)
                angle = self.path_planning_model([state, goal_distance_tf])
                position_x = self.check_boundaries(position_x + predict_increment * math.cos(angle), 'x_coordinate')
                position_y = self.check_boundaries(position_y + predict_increment * math.sin(angle), 'y_coordinate')
        length = len(control_points_x)

        knots = np.linspace(0, 1, length - (polynomial_degree - 1), endpoint=True)
        knots = np.append(np.zeros(polynomial_degree), knots)
        knots = np.append(knots, np.ones(polynomial_degree))

        tck = [knots, [control_points_x, control_points_y], polynomial_degree]
        u3 = np.linspace(0, 1, 100, endpoint=True)
        interpolated_trajectory = interpolate.splev(u3, tck)
        self.smoothed_trajectory_points = interpolated_trajectory
        return

    def detect_close_obstacles(self):
        gray_matrix = self.obstacle_matrix.copy()
        gray_matrix[0, :] = 0
        gray_matrix[:, 0] = 0
        gray_matrix[-1, :] = 0
        gray_matrix[:, -1] = 0
        gray_matrix = gray_matrix.astype(np.uint8)
        gray_matrix *= 255
        blur_value = 3
        correction_index = blur_value - np.ceil(blur_value/2)
        gray_matrix = cv2.blur(gray_matrix, (blur_value, blur_value))
        gray_matrix[gray_matrix > 0] = 255
        contours = cv2.findContours(gray_matrix.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        for borders in contours:
            # M = cv2.moments(borders)                          # compute the center of the contour
            # cx = int(M["m10"] / (M["m00"] + 0.01))
            # cy = int(M["m01"] / (M["m00"] + 0.01))
            vertex = np.reshape(borders, [-1, 2])
            max_x, max_y = np.max(vertex[:, 0]), np.max(vertex[:, 1])
            min_x, min_y = np.min(vertex[:, 0]), np.min(vertex[:, 1])
            external_vertex = np.array([[min_x+correction_index, min_y+correction_index],[min_x+correction_index, max_y-correction_index],[max_x-correction_index, max_y-correction_index],[max_x-correction_index, min_y+correction_index]])
            if len(np.unique(external_vertex, axis=0)) >= 4:
                self.fill_rectangles(external_vertex)

    def fill_rectangles(self, points):
        x = np.arange(0, self.n_cell_y)             # TODO: to check
        y = np.arange(0, self.n_cell_x)             # TODO: to check
        x_coord, y_coord = zip(*points)
        if x_coord[1] >= x_coord[3]:
            mask_1 = (x[np.newaxis, :]) * (y_coord[0] - y_coord[1]) - (y[:, np.newaxis]) * (x_coord[0] - x_coord[1])\
                     - (x_coord[1] * (y_coord[0] - y_coord[1]) - y_coord[1] * (x_coord[0] - x_coord[1])) >= 0
            mask_2 = (x[np.newaxis, :]) * (y_coord[1] - y_coord[2]) - (y[:, np.newaxis]) * (x_coord[1] - x_coord[2])\
                     - (x_coord[2] * (y_coord[1] - y_coord[2]) - y_coord[2] * (x_coord[1] - x_coord[2])) >= 0
            mask_3 = (x[np.newaxis, :]) * (y_coord[2] - y_coord[3]) - (y[:, np.newaxis]) * (x_coord[2] - x_coord[3])\
                     - (x_coord[3] * (y_coord[2] - y_coord[3]) - y_coord[3] * (x_coord[2] - x_coord[3])) >= 0
            mask_4 = (x[np.newaxis, :]) * (y_coord[3] - y_coord[0]) - (y[:, np.newaxis]) * (x_coord[3] - x_coord[0])\
                     - (x_coord[0] * (y_coord[3] - y_coord[0]) - y_coord[0] * (x_coord[3] - x_coord[0])) >= 0
        else:
            mask_1 = (x[np.newaxis, :]) * (y_coord[0] - y_coord[1]) - (y[:, np.newaxis]) * (x_coord[0] - x_coord[1])\
                     - (x_coord[1] * (y_coord[0] - y_coord[1]) - y_coord[1] * (x_coord[0] - x_coord[1])) <= 0
            mask_2 = (x[np.newaxis, :]) * (y_coord[1] - y_coord[2]) - (y[:, np.newaxis]) * (x_coord[1] - x_coord[2])\
                     - (x_coord[2] * (y_coord[1] - y_coord[2]) - y_coord[2] * (x_coord[1] - x_coord[2])) <= 0
            mask_3 = (x[np.newaxis, :]) * (y_coord[2] - y_coord[3]) - (y[:, np.newaxis]) * (x_coord[2] - x_coord[3])\
                     - (x_coord[3] * (y_coord[2] - y_coord[3]) - y_coord[3] * (x_coord[2] - x_coord[3])) <= 0
            mask_4 = (x[np.newaxis, :]) * (y_coord[3] - y_coord[0]) - (y[:, np.newaxis]) * (x_coord[3] - x_coord[0])\
                     - (x_coord[0] * (y_coord[3] - y_coord[0]) - y_coord[0] * (x_coord[3] - x_coord[0])) <= 0

        final_mask = np.logical_and.reduce((mask_1, mask_2, mask_3, mask_4))
        final_mask = np.logical_xor((self.obstacle_matrix*final_mask), final_mask)
        obstacle_position_x = np.reshape(self.row_indexes[final_mask], [-1, 1]) * self.map_resolution_x
        obstacle_position_y = np.reshape(self.column_indexes[final_mask], [-1, 1]) * self.map_resolution_x
        obstacles_position = np.hstack([obstacle_position_x, obstacle_position_y])
        self.newfound_obstacle_list = np.vstack([self.newfound_obstacle_list, obstacles_position])
        self.obstacle_matrix[final_mask] = 1
        self.covered_area_matrix[final_mask] = 1

    def check_boundaries(self, position, direction):
        if direction == 'x_coordinate':
            if position >= self.map_dimension_x:
                position = self.map_dimension_x
            elif position <= 0:
                position = 0
            else:
                pass
        elif direction == 'y_coordinate':
            if position >= self.map_dimension_y:
                position = self.map_dimension_y
            elif position <= 0:
                position = 0
            else:
                pass
        elif direction == 'z_coordinate':
            if position >= self.map_dimension_z:
                position = self.map_dimension_z
            elif position <= 0:
                position = 0
            else:
                pass
        else:
            os.error('Wrong direction as 2nd argument in [Environment\check_boundaries]')
        return position

    def pos2index(self, pos):
        idx = int(math.floor(pos / self.map_resolution_x))
        return idx

    def index2pos(self, idx):
        pos = round(idx*self.map_resolution_x, 2)
        return pos

    def action(self, drone_list):
        self.detect_obstacles()
        self.share_covered_area(drone_list)                         # share information about coverage matrix
        if self.newfound_obstacle_list.any():                       # update fixed obstacle list
            self.update_fixed_obstacles(drone_list)
        self.proximity_sensor()
        # self.get_other_drones_positions(drone_list)                 # collect other drones position
        # self.update_mobile_obstacles()
        self.check_goal()                                           # check if the goals is inside a wall
        if not self.goal_updated:
            self.coverage_random(self.position[2])
            self.update_attractive_layer()
            # self.reset_local_minima_obstacles()                     # remove the previous local minima obstacles
        # if self.local_minima_flag:
        #     self.potential_field_path_planning_classic()
        if self.emergency_path_planning_flag:
            self.emergency_path_planning()
        else:
            self.reinforcement_learning_path_planning()
            # self.potential_field_path_planning()