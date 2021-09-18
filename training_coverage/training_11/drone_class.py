import numpy as np
import math
import scipy.stats as stats
# import scipy.ndimage.filters as filters
# import scipy.ndimage.morphology as morphology
# from scipy.spatial import ConvexHull
# from sklearn.cluster import KMeans
from scipy import interpolate
# import cv2
import os
import time
# import imutils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


class Drone:
    def __init__(self):
        self.drone_ID = 0                                       # unique ID identifying the drone
        self.position = np.empty(2, dtype='float16')            # current position of the drone
        self.index_position = np.array([0, 0], dtype='int16')
        self.orientation = 0                                    # orientation in radians
        self.position_history = np.empty([0,2], dtype='float16')# vector containing all the past values of the drone position
        self.goal_history = np.empty([0, 2], dtype='float16')
        self.radius = 0.2                                       # drone dimensions [m]
        self.speed = np.zeros([2])                              # current speed
        self.max_speed = None                                   # max abs of the speed vector
        self.vision_range = None                                # vision range of the drone [m]
        self.max_vision_angle = None                            # max angle of vision [degrees]
        self.vision_resolution = None                           # how many ray are used in the vision range
        self.map_dimension_x = None                             # initialization of map dimension. Precise number if the map is known, upper approx. if map is unknown
        self.map_dimension_y = None
        self.map_resolution_x = None                            # resolution of the potential field matrix (dimension of the cells)
        self.map_resolution_y = None
        self.n_cell_x = None                                    # number of cell along the x_coordinate dimension of the map (computed from the data above)
        self.n_cell_y = None
        self.min_index_x = None                                 # limit indexes of the internal map matrix
        self.max_index_x = None
        self.min_index_y = None
        self.max_index_y = None
        self.initialization_completed = False                   # only false at the beginning of the simulation to allow initialization of the map matrix
        self.potential_field_map = None
        self.attractive_constant = None                         # attractive potential gain
        self.layer_attractive = None
        self.x_cohordinate_matrix = None                        # used to speed up a lot the attractive pot. field computation
        self.y_cohordinate_matrix = None
        self.obstacle_map = None
        self.obstacle_matrix = None
        self.obstacle_list = np.empty([0,2], dtype='float16')   # stores all obstacles position as indexes of the corresponding cell
        self.newfound_obstacle_list = np.empty([0, 2], dtype='float16')# temporarily stores newfound obstacles position
        self.newfound_obstacle_share_list = np.empty([0, 2])
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
        self.mobile_obstacle_last_position = np.empty([0,3])    # 3-column matrix. 1st and 2nd columns are x_coordinate and y_coordinate of the mobile obstacle matrices.
                                                                # 3rd column indicates which circle is being used (big-medium-small). This matrix
                                                                # is used to delete mobile obstacle circles before drawing the new ones (after
                                                                # the other drone position and velocity has changed)
        self.mobile_matrix_peak_value = None                    # value of the potential in the middle of the 2D distribution for mobile obstacle matrix
        self.fixed_matrix_peak_value = None                     # value of the potential in the middle of the 2D distribution for fixed obstacle matrix
        self.minima_matrix_peak_value = None                    # value of the potential in the middle of the 2D distribution for local minima obstacle matrixes
        self.other_drones_current_position = None               # 4 columns: [x_coordinate, y_coordinate, v_x, v_y], 1 row for each drone
        self.layer_experience = None                            # matrix containing the "experience" component of the potential field
        self.goal_updated = False                               # signals that the goal has changed with respect to the previous one and that the pot. field layer must be updated
        self.goal = [0, 0]                                      # current goal
        self.old_goal = self.goal                               # last goal (used to correctly manage the potential field matrix update)
        self.min_goal_distance = 0.3                            # min distance to reach the goal
        self.path_planning_model = None                         # RL model for path planning
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
        self.covered_area_matrix_single_drone = None            # matrix that contains information about coverage of the single drone
        self.single_coverage = [0]                              # list that contains the % of the single exploration of the drone
        self.explored_cell_number = 0
        self.predict_length = None                              # length of the predicted trajectory
        self.smoothed_trajectory_points = [0, 0]                # list that will contain the points of the interpolated trajectory
        self.n_drones = None
        self.id_list = []
        self.goal_steps = 0
        self.max_steps_single_goal = None
        self.centers_position_index = np.empty([0,2])
        self.centers_position_index_memory = np.empty([0,2])
        self.column_indexes = None
        self.row_indexes = None
        self.motion_options = None
        self.drone_matrix = None

    def __str__(self):  # overwrites print. This way the command print(drone_obj) can be used to inspect the current position and objective of a certain drone
        x = 0
        y = 1
        return "drone %i is located in (%.2f, %.2f) and its goal is (%.2f, %.2f)" % (self.drone_ID, self.position[x], self.position[y], self.goal[x], self.goal[y])

    def set_drone_ID(self, ID):
        ID = int(round(ID))
        self.drone_ID = ID

    def set_initial_position(self, intial_position):
        x = 0
        y = 1
        intial_position = np.array(intial_position, dtype="f")
        self.position = intial_position
        self.index_position[x] = self.pos2index(self.position[x])
        self.index_position[y] = self.pos2index(self.position[y])

    def set_random_initial_position(self):
        # sets random x and y position
        x = 0
        y = 1
        safe_border_percentage_margin = 8   # percentage of the map size tht is taken as "safety margin" --> e.g. if this
                                            # number is 10 then the drone cannot be closer to the map edge than 10% of the map size
        initial_position = [np.random.random() * (self.map_dimension_x * (100 - 2*safe_border_percentage_margin)/100) + self.map_dimension_x * safe_border_percentage_margin/100,
                            np.random.random() * (self.map_dimension_y * (100 - 2*safe_border_percentage_margin)/100) + self.map_dimension_y * safe_border_percentage_margin/100]
        self.position = initial_position
        self.index_position[x] = self.pos2index(self.position[x])
        self.index_position[y] = self.pos2index(self.position[y])
        return

    def set_observation_size(self, N):
        self.observation_size = N

    def set_vision_settings(self, N, max_angle, resolution):
        self.vision_range = N
        self.max_vision_angle = max_angle
        self.vision_resolution = resolution

    def set_RL_path_planning_model(self, model):
        self.path_planning_model = model

    def set_goal(self, goal):
        new_goal = np.array(goal, dtype="f")
        self.old_goal = self.goal               # stores previous goal in memory
        self.goal_history = np.vstack([self.goal_history, new_goal])
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

    def set_matrix_peak_value(self, mobile_peak_value, fixed_peak_value):
        self.mobile_matrix_peak_value = mobile_peak_value
        self.fixed_matrix_peak_value = fixed_peak_value

    def max_steps_apf_descent_path_planning(self, N):
        self.max_step_apf_descent = N

    def set_attractive_constant(self, attractive_constant):
        self.attractive_constant = attractive_constant

    def set_max_speed(self, max_speed):
        self.max_speed = max_speed

    def import_map_properties(self, map_object):
        # collects data about the map (only for simulation purpose)
        self.map_dimension_x = map_object.map_dimension_x
        self.map_dimension_y = map_object.map_dimension_y
        self.map_resolution_x = map_object.map_resolution
        self.map_resolution_y = map_object.map_resolution
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

        self.min_index_x = 0
        self.max_index_x = int(math.floor(self.map_dimension_x / self.map_resolution_x)) # alternative: self.n_cell_x-1
        self.min_index_y = 0
        self.max_index_y = int(math.floor(self.map_dimension_y / self.map_resolution_x)) # alternative: self.n_cell_y-1
        #initialize position history vector
        self.position_history = np.zeros([1, 2])
        self.position_history[0, :] = self.position
        # initialize potential world as matrix of zeros
        self.potential_field_map = np.zeros((self.n_cell_x, self.n_cell_y))
        self.layer_attractive = np.zeros((self.n_cell_x, self.n_cell_y))
        self.layer_experience = np.zeros((self.n_cell_x, self.n_cell_y))
        self.covered_area_matrix = np.zeros((self.n_cell_x, self.n_cell_y))
        self.covered_area_matrix_single_drone = np.zeros((self.n_cell_x, self.n_cell_y))
        self.obstacle_matrix = np.zeros((self.n_cell_x, self.n_cell_y))
        self.total_explorable_elements = self.covered_area_matrix.size
        x = np.linspace(0, self.map_dimension_x, self.n_cell_x)
        y = np.linspace(0, self.map_dimension_y, self.n_cell_y)
        self.x_cohordinate_matrix, self.y_cohordinate_matrix = np.meshgrid(x, y)
        indexes = np.arange(0, self.n_cell_x)
        self.column_indexes = indexes * np.ones([self.n_cell_x, 1])
        self.row_indexes = self.column_indexes.T
        # initialize the motion option for classic algorithm
        self.motion_options = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1])).T.reshape(-1, 2)
        idx = int(np.where((self.motion_options == [0, 0]).all(axis = 1))[0])
        self.motion_options = np.delete(self.motion_options, idx, axis=0)    # find index of the (0, 0) row in the list of possible motions
        # initialize obstacle matrices
        self.compute_fixed_obstacle_matrix()    # for fixed obstacles
        self.compute_mobile_obstacle_matrices() # for mobile obstacles
        self.compute_drone_influence_matrix(25)
        self.generate_border_obstacles()
        self.n_drones = n_drones
        self.single_coverage.append((np.count_nonzero(self.covered_area_matrix) / n_drones)/ self.total_explorable_elements * 100)
        self.other_drones_current_position = np.empty((0, 5))   # initialize vector with other drones positions and velocities
        self.initialization_completed = True    # initialization completed

    def update_attractive_layer(self):
        x = 0 # only for readability
        y = 1
        self.potential_field_map -= self.layer_attractive   # removes contribute of the previous goal attractive field
        self.layer_attractive = self.attractive_constant * np.hypot(self.x_cohordinate_matrix - self.goal[x], self.y_cohordinate_matrix - self.goal[y]).T
        self.potential_field_map += self.layer_attractive
        self.goal_updated = True

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
        self.newfound_obstacle_share_list = np.empty([0, 2])
        return

    def share_covered_area(self, drone_list):
        for drone in drone_list:
            if drone.drone_ID != self.drone_ID:
                drone.covered_area_matrix += self.covered_area_matrix
                drone.covered_area_matrix = np.clip(drone.covered_area_matrix, 0, 1)
        return

    def matrix_placer(self, main_matrix, matrix, index_position_x, index_position_y, sign):
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
        main_matrix[min_x:max_x, min_y:max_y] += sign * matrix[min_x_A:max_x_A, min_y_A:max_y_A]


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

    def compute_drone_influence_matrix(self, matrix_dimension):
        # self.drone_matrix = np.ones([matrix_dimension, matrix_dimension])
        peak_value = 1
        mean = 0
        sigma = 1/4         # variance --> increase the denominator to obtain a smaller repulsive area around the obstacle
        n_points_x = int(matrix_dimension)      # computes dimension of the repulsive matrix along x_coordinate
        n_points_y = int(matrix_dimension)      # computes dimension of the repulsive matrix along y_coordinate
        n_points_x = math.floor(n_points_x/2)*2 + 1                                     # "trick" to make n_points_x to be odd every time
        n_points_y = math.floor(n_points_y/2)*2 + 1
        x_vec = np.linspace(-1, 1, n_points_x, endpoint=True).reshape(n_points_x, 1)    # column vector
        y_vec = np.linspace(-1, 1, n_points_y, endpoint=True).reshape(1, n_points_y)    # row vector
        dist_x = np.array(stats.norm.pdf(x_vec, mean, sigma))                           # normal distribution for x_coordinate direction
        dist_y = np.array(stats.norm.pdf(y_vec, mean, sigma))                           # normal distribution for y_coordinate direction
        dist_x = dist_x - min(dist_x)                                                   # to have the border of the distribution = 0
        dist_y = dist_y - min(dist_y.T)                                                 # same as line above
        dist_x = dist_x * math.sqrt(peak_value)/max(dist_x)           # rescale the distribution to match the desired max value
        dist_y = dist_y * math.sqrt(peak_value)/max(dist_y.T)         #  NOTE: max could be the wrong function to be used here (and also in the line above)
        self.drone_matrix = dist_x * dist_y

    def update_fixed_obstacles(self, drone_list):
        self.newfound_obstacle_list = np.unique(self.newfound_obstacle_list, axis=0)
        for obstacle_position_x, obstacle_position_y in self.newfound_obstacle_list:                # runs over the whole list of newly detected obstacles
            x_idx = int(math.floor(obstacle_position_x/self.map_resolution_x))    # index of x_coordinate-cell corresponding to obstacle position
            y_idx = int(math.floor(obstacle_position_y/self.map_resolution_y))    # index of y_coordinate-cell corresponding to obstacle position
            self.obstacle_matrix[x_idx, y_idx] = 1
            if self.check_fixed_obstacle_position(obstacle_position_x, obstacle_position_y):
                self.matrix_placer(self.potential_field_map, self.fixed_obstacle_matrix, x_idx, y_idx, +1)  # add the repulsive matrix to the global APF matrix
                self.matrix_placer(self.covered_area_matrix, self.drone_matrix, x_idx, y_idx, +1)
                self.obstacle_list = np.vstack([self.obstacle_list, np.array([obstacle_position_x, obstacle_position_y])])  # append the newfound obstacles to the complete obstacle list
                self.newfound_obstacle_share_list = np.vstack([self.newfound_obstacle_share_list, [obstacle_position_x, obstacle_position_y]])
            #     if len(self.obstacle_list) <= self.obstacle_list_security_length:
            #         self.obstacle_list = np.vstack([self.obstacle_list, np.array([x, y])])  # append the newfound obstacles to the complete obstacle list
            #         self.obstacle_list = np.unique(self.obstacle_list, axis=0)
            #     else:
            #         raise Warning("Memory limit reached for obstacle list - obstacle list can not be updated")
        self.share_new_obstacle_positions(drone_list)  # update the obstacles found in this iteration
        self.newfound_obstacle_list = np.empty([0, 2])    # empty newfound obstacle list -> it will be replenished during the next scanning phase

    def update_mobile_obstacles(self):
        # for x_idx, y_idx, ii in self.mobile_obstacle_last_position:
        #     self.matrix_placer(self.potential_field_map, self.mobile_obstacle_matrix[int(ii),:,:], x_idx, y_idx, -1)
        self.mobile_obstacle_last_position = np.empty([0,3])
        # d0 = 0.2 * self.drone_safe_distance                        
        for x, y, vx, vy, _ in self.other_drones_current_position:
            theta = math.atan2(vy, vx)
            x_idx = int(math.floor((x + self.max_speed * math.cos(theta)) / self.map_resolution_x))
            y_idx = int(math.floor((y + self.max_speed * math.sin(theta)) / self.map_resolution_y))
            self.matrix_placer(self.potential_field_map, self.mobile_obstacle_matrix[0, :, :], x_idx, y_idx, +1)
            self.mobile_obstacle_last_position = np.vstack([self.mobile_obstacle_last_position, [x_idx, y_idx, 0]])
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
        nx = int(round(self.map_dimension_x/self.map_resolution_x))         # n° of points along x_coordinate
        ny = int(round(self.map_dimension_y/self.map_resolution_x))         # n° of points along y_coordinate
        max_idx_x = nx - 1                                          # index of the max point along x_coordinate
        max_idx_y = ny - 1                                          # index of the max point along y_coordinate
        x_vec = np.linspace(0, max_idx_x*self.map_resolution_x, nx, endpoint=True).reshape(nx, 1)    # column vector of indexes of the obstacles along x_coordinate
        y_vec = np.linspace(0, max_idx_y*self.map_resolution_x, ny, endpoint=True).reshape(ny, 1)
        const_vec_x = np.ones([nx, 1])                      # constant column vector (vector of ones)
        const_vec_y = np.ones([ny, 1])
        border_obstacles = np.append(x_vec, const_vec_x*0, axis=1)
        border_obstacles = np.append(border_obstacles, np.append(x_vec, const_vec_x*max_idx_y*self.map_resolution_x, axis=1), axis=0)
        border_obstacles = np.append(border_obstacles, np.append(const_vec_y*0, y_vec, axis=1), axis=0)
        border_obstacles = np.append(border_obstacles, np.append(const_vec_y*max_idx_x*self.map_resolution_x, y_vec, axis=1), axis=0)
        self.obstacle_matrix[0, :] = 1
        self.obstacle_matrix[:, 0] = 1
        self.obstacle_matrix[-1, :] = 1
        self.obstacle_matrix[:, -1] = 1
        self.covered_area_matrix[0, :] = 1
        self.covered_area_matrix[:, 0] = 1
        self.covered_area_matrix[-1, :] = 1
        self.covered_area_matrix[:, -1] = 1
        self.newfound_obstacle_list = np.append(self.newfound_obstacle_list, border_obstacles, axis=0)

    def check_fixed_obstacle_position(self, x_idx, y_idx):
        found_idx = np.where((self.obstacle_list == [x_idx, y_idx]).all(axis=1))[0]          # check for obstacles in the same position along x_coordinate (1st column of the obstacle list array)
        if found_idx.size > 0:  # if there is any match i.e. there is already an obstacle in the same position, a "False" flag is returned and obstacle is ignored in "update_fixed_obstacles"
            return False
        else:
            return True
        # alternative: return found_idx_size > 0

    def potential_field_path_planning(self):
        """
        ADVANCED POTENTIAL FIELD PATH PLANNING
        The basic idea is to privilege the movement toward the objective rather than the movement in the direction of the lowest
        possible potential. To achieve this objective, the drone tries to move directly toward the objective. If the potential
        that direction is lower (minus an optional offset) than the current potential, then the drone moves without exploring
        any other movement possibility. Otherwise, it moves 1° to the right, then 1° to the left and so on until it finds a
        suitable movement direction.
        """
        x = 0  # only for readability
        y = 1  # same as above
        goal_distance_x = self.goal[x] - self.position[x]
        goal_distance_y = self.goal[y] - self.position[y]
        goal_direction = math.atan2(goal_distance_y, goal_distance_x)       # constant -> to remove
        if np.hypot(goal_distance_x, goal_distance_y) < self.min_goal_distance:                               # objective has been found (inside the same cell of the drone)
            self.speed[x] = 0
            self.speed[y] = 0
            self.position[x], self.position[y] = self.goal[x], self.goal[y]
            self.goal_updated = False
        else:
            sweep_angle = 0
            sign = -1
            current_cell_idx_x = int(math.floor(self.position[x] / self.map_resolution_x))
            current_cell_idx_y = int(math.floor(self.position[y] / self.map_resolution_y))
            current_potential = self.potential_field_map[current_cell_idx_x][current_cell_idx_y]
            potential_offset = 0
            while abs(sweep_angle) <= math.pi:
                angle = goal_direction + sign * sweep_angle
                new_position_x = self.position[x] + self.max_speed * math.cos(angle)
                new_position_y = self.position[y] + self.max_speed * math.sin(angle)
                # new_cell_idx_x = int(math.floor(new_x / self.map_resolution_x))
                # new_cell_idx_y = int(math.floor(new_y / self.map_resolution_y))
                new_cell_idx_x = self.pos2index(new_position_x)
                new_cell_idx_y = self.pos2index(new_position_y)
                new_potential = self.potential_field_map[new_cell_idx_x][new_cell_idx_y]
                if new_potential < (current_potential + potential_offset):
                    # selected direction is good!
                    self.orientation = angle
                    self.speed[x] = round(self.max_speed * math.cos(angle), 2)
                    self.speed[y] = round(self.max_speed * math.sin(angle), 2)
                    self.position[x] = new_position_x
                    self.position[y] = new_position_y
                    self.position_history = np.round(np.vstack([self.position_history, self.position]), 2)
                    self.index_position[x] = self.pos2index(self.position[x])
                    self.index_position[y] = self.pos2index(self.position[y])
                    return
                else:
                    # selected direction is not good. Another one will be tried
                    if sign == -1:
                        sweep_angle += math.pi / 30                                 # angle increased by 6
                    sign *= -1                                                      # sign of the sweep angle is flipped
            self.position[x], self.position[y] = self.position[x], self.position[y]         # if no suitable movement direction has been found, the drone stays still (see line below)
            self.speed[x], self.speed[y] = 0, 0
            self.local_minima_flag = True               # switch to the other algorithm
            self.step_counter = 0

            """ A possible alternative to returning the current position is to lower the potential offset (even to a negative 
                value) and recursively recall the "potential_field_path_planning_v2" in order to always find the best direction
                to move away from the current cell (this could lead the drone on a higher potential but avoids getting stucked)
            """
        self.position_history = np.vstack([self.position_history, self.position])
        return

    def reinforcement_learning_path_planning(self):
        x = 0
        y = 1
        goal_distance_x = self.goal[x] - self.position[x]
        goal_distance_y = self.goal[y] - self.position[y]
        if np.hypot(goal_distance_x, goal_distance_y) < self.min_goal_distance:                               # objective has been found (inside the same cell of the drone)
            self.speed[x] = 0
            self.speed[y] = 0
            self.position[x], self.position[y] = self.goal[x], self.goal[y]
            self.goal_updated = False
        else:
            state = self.observe()
            state = tf.expand_dims(tf.convert_to_tensor(state), 0)
            # angle = self.model.predict(state, batch_size=64, max_queue_size=10, workers=1, use_multiprocessing=True)
            angle = self.path_planning_model(state)
            self.orientation = angle
            self.speed[x] = self.max_speed * math.cos(angle)
            self.speed[y] = self.max_speed * math.sin(angle)
            self.interpolate_trajectory()
            position_x = self.check_boundaries(self.smoothed_trajectory_points[x][int(np.round(self.predict_length/2.5))], 'x_coordinate')
            position_y = self.check_boundaries(self.smoothed_trajectory_points[y][int(np.round(self.predict_length/2.5))], 'y_coordinate')
            self.position = [position_x, position_y]
            self.index_position[x] = self.pos2index(self.position[x])
            self.index_position[y] = self.pos2index(self.position[y])
            self.position_history = np.vstack([self.position_history, self.position])

    def detect_obstacles(self):
        x = 0
        y = 1
        cell_vision_range = self.vision_range / self.map_resolution_x
        previous_value = np.count_nonzero(self.covered_area_matrix)
        for sweep_angle in np.linspace(-self.max_vision_angle, self.max_vision_angle, self.vision_resolution):
            distance = 0
            sweep_angle_rad = math.radians(sweep_angle)
            angle = self.orientation + sweep_angle_rad
            while distance <= cell_vision_range:
                idx_x = int(math.floor(self.index_position[x] + distance * math.cos(angle)))
                idx_y = int(math.floor(self.index_position[y] + distance * math.sin(angle)))
                self.covered_area_matrix[idx_x, idx_y] = 1              # fill the coverage matrix in the explored area
                self.covered_area_matrix_single_drone[idx_x, idx_y] = 1
                if self.obstacle_map[idx_x, idx_y] == 1:
                    self.obstacle_matrix[idx_x, idx_y] = 1
                    self.covered_area_matrix_single_drone[idx_x, idx_y] = 1
                    self.covered_area_matrix[idx_x, idx_y] = 1          # fill the coverage matrix in the obstacles
                    obstacle_position_x = idx_x * self.map_resolution_x
                    obstacle_position_y = idx_y * self.map_resolution_y
                    # if self.check_fixed_obstacle_position(obstacle_position_x, obstacle_position_y):
                    self.newfound_obstacle_list = np.vstack([self.newfound_obstacle_list, [obstacle_position_x, obstacle_position_y]])
                    break
                else:
                    distance += 1
        # self.detect_close_obstacles()
        next_value = np.count_nonzero(self.covered_area_matrix)
        previous_coverage_increment = self.single_coverage[-1] * self.total_explorable_elements / 100
        single_percentage = (previous_coverage_increment + (next_value - previous_value)) / self.total_explorable_elements * 100
        self.explored_cell_number = next_value - previous_value
        self.single_coverage.append(single_percentage)
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
        observation[min_x_A:max_x_A, min_y_A:max_y_A] = self.potential_field_map[min_x:max_x, min_y:max_y]
        observation = observation - observation[int(self.observation_size - delta - 1), int(self.observation_size - delta - 1)]
        observation /= 1000
        observation = np.round(observation, 3)
        return observation

    def predict(self, position_x, position_y):
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
        observation[min_x_A:max_x_A, min_y_A:max_y_A] = self.potential_field_map[min_x:max_x, min_y:max_y]
        observation = observation - observation[int(self.observation_size - delta - 1), int(self.observation_size - delta - 1)]
        observation /= 1000
        return observation

    def interpolate_trajectory(self):
        x = 0
        y = 1
        degree = 3
        steps = 2
        control_points_x = [self.position[x]]
        control_points_y = [self.position[y]]
        position_x = self.position[x]
        position_y = self.position[y]
        # goal_distance_x = self.goal[x] - self.position[x]
        # goal_distance_y = self.goal[y] - self.position[y]
        # if np.hypot(goal_distance_x, goal_distance_y) < self.predict_length:
        #     prediction_steps = int(np.hypot(goal_distance_x, goal_distance_y))
        # else:
        #     prediction_steps = self.predict_length
        for ii in range(0, self.predict_length):
            for jj in range(0, steps):
                state = self.predict(position_x, position_y)
                state = tf.expand_dims(tf.convert_to_tensor(state), 0)
                angle = self.path_planning_model(state)
                position_x = self.check_boundaries(position_x + self.max_speed * math.cos(angle), 'x_coordinate')
                position_y = self.check_boundaries(position_y + self.max_speed * math.sin(angle), 'y_coordinate')
            control_points_x.append(position_x)
            control_points_y.append(position_y)

        length = len(control_points_x)

        knots = np.linspace(0, 1, length - (degree - 1), endpoint=True)
        knots = np.append(np.zeros(degree), knots)
        knots = np.append(knots, np.ones(degree))

        tck = [knots, [control_points_x, control_points_y], degree]
        u3 = np.linspace(0, 1, (max(length * 2, 100)), endpoint=True)
        interpolated_trajectory = interpolate.splev(u3, tck)
        self.smoothed_trajectory_points = interpolated_trajectory
        return

    # def detect_close_obstacles(self):
    #     self.centers_position_index = np.empty([0,4])
    #     gray_matrix = self.obstacle_matrix.copy()
    #     gray_matrix[0, :] = 0
    #     gray_matrix[:, 0] = 0
    #     gray_matrix[-1, :] = 0
    #     gray_matrix[:, -1] = 0
    #     gray_matrix = gray_matrix.astype(np.uint8)
    #     gray_matrix *= 255
    #     blur_value = 5
    #     correction_index = blur_value - 1
    #     gray_matrix = cv2.blur(gray_matrix, (blur_value, blur_value))
    #     gray_matrix[gray_matrix > 0] = 255
    #     contours = cv2.findContours(gray_matrix.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     contours = imutils.grab_contours(contours)
    #
    #     for borders in contours:
    #         # M = cv2.moments(borders)                          # compute the center of the contour
    #         # cx = int(M["m10"] / (M["m00"] + 0.01))
    #         # cy = int(M["m01"] / (M["m00"] + 0.01))
    #         vertex = np.reshape(borders, [-1, 2])
    #         max_x, max_y = np.max(vertex[:, 0]), np.max(vertex[:, 1])
    #         min_x, min_y = np.min(vertex[:, 0]), np.min(vertex[:, 1])
    #         external_vertex = np.array([[min_x+correction_index, min_y+correction_index],[min_x+correction_index, max_y-correction_index],[max_x-correction_index, max_y-correction_index],[max_x-correction_index, min_y+correction_index]])
    #         if len(np.unique(external_vertex)) == 4:
    #             self.fill_rectangles(external_vertex)

    # def fill_rectangles(self, points):
    #     x = np.arange(0, self.n_cell_y)             # TODO: to check
    #     y = np.arange(0, self.n_cell_x)             # TODO: to check
    #     x_coord, y_coord = zip(*points)
    #     if x_coord[1] >= x_coord[3]:
    #         mask_1 = (x[np.newaxis, :]) * (y_coord[0] - y_coord[1]) - (y[:, np.newaxis]) * (x_coord[0] - x_coord[1])\
    #                  - (x_coord[1] * (y_coord[0] - y_coord[1]) - y_coord[1] * (x_coord[0] - x_coord[1])) > 0
    #         mask_2 = (x[np.newaxis, :]) * (y_coord[1] - y_coord[2]) - (y[:, np.newaxis]) * (x_coord[1] - x_coord[2])\
    #                  - (x_coord[2] * (y_coord[1] - y_coord[2]) - y_coord[2] * (x_coord[1] - x_coord[2])) >= 0
    #         mask_3 = (x[np.newaxis, :]) * (y_coord[2] - y_coord[3]) - (y[:, np.newaxis]) * (x_coord[2] - x_coord[3])\
    #                  - (x_coord[3] * (y_coord[2] - y_coord[3]) - y_coord[3] * (x_coord[2] - x_coord[3])) >= 0
    #         mask_4 = (x[np.newaxis, :]) * (y_coord[3] - y_coord[0]) - (y[:, np.newaxis]) * (x_coord[3] - x_coord[0])\
    #                  - (x_coord[0] * (y_coord[3] - y_coord[0]) - y_coord[0] * (x_coord[3] - x_coord[0])) > 0
    #     else:
    #         mask_1 = (x[np.newaxis, :]) * (y_coord[0] - y_coord[1]) - (y[:, np.newaxis]) * (x_coord[0] - x_coord[1])\
    #                  - (x_coord[1] * (y_coord[0] - y_coord[1]) - y_coord[1] * (x_coord[0] - x_coord[1])) <= 0
    #         mask_2 = (x[np.newaxis, :]) * (y_coord[1] - y_coord[2]) - (y[:, np.newaxis]) * (x_coord[1] - x_coord[2])\
    #                  - (x_coord[2] * (y_coord[1] - y_coord[2]) - y_coord[2] * (x_coord[1] - x_coord[2])) <= 0
    #         mask_3 = (x[np.newaxis, :]) * (y_coord[2] - y_coord[3]) - (y[:, np.newaxis]) * (x_coord[2] - x_coord[3])\
    #                  - (x_coord[3] * (y_coord[2] - y_coord[3]) - y_coord[3] * (x_coord[2] - x_coord[3])) <= 0
    #         mask_4 = (x[np.newaxis, :]) * (y_coord[3] - y_coord[0]) - (y[:, np.newaxis]) * (x_coord[3] - x_coord[0])\
    #                  - (x_coord[0] * (y_coord[3] - y_coord[0]) - y_coord[0] * (x_coord[3] - x_coord[0])) <= 0
    #
    #     final_mask = np.logical_and.reduce((mask_1, mask_2, mask_3, mask_4))
    #     final_mask = np.logical_xor((self.obstacle_matrix*final_mask), final_mask)
    #     obstacle_position_x = np.reshape(self.row_indexes[final_mask], [-1, 1]) * self.map_resolution_x
    #     obstacle_position_y = np.reshape(self.column_indexes[final_mask], [-1, 1]) * self.map_resolution_x
    #     obstacles_position = np.hstack([obstacle_position_x, obstacle_position_y])
    #     self.newfound_obstacle_list = np.vstack([self.newfound_obstacle_list, obstacles_position])
    #     self.obstacle_matrix[final_mask] = 1
    #     self.covered_area_matrix[final_mask] = 1

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
        self.get_other_drones_positions(drone_list)                 # collect other drones position
        self.update_mobile_obstacles()
        # self.reinforcement_learning_path_planning()
        self.potential_field_path_planning()