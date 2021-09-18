import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import gym
from gym import spaces
import heatmap
import math
import scipy.stats as stats

gym.logger.set_level(40)

class Environment:

    def __init__(self):
        self.map_dimension_x = None
        self.map_dimension_y = None
        self.map_resolution = None
        self.N_cells_x = None
        self.N_cells_y  = None
        self.obstacle_map = None
        self.cost_map = None
        self.step_number = 0
        self.total_steps = 0
        self.episode_number = 0
        self.max_step_number = None
        self.map_counter = 0
        self.max_map_number = None
        self.map_folder_path = ''
        self.coverage_observation_size = None    # NOTE: this number is different from the one used in the drone class, even if they have the same name!
                                                 # Here observation_size is used by the coverage NN, while in the drone class is used for the path planning
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = None
        self.current_direction = None
        self.previous_direction = None
        self.obstacle_safe_distance = 1.5
        self.fixed_obstacle_matrix = None
        self.fixed_obstacle_matrix_dimensions = None
        self.fixed_matrix_peak_value = 200
        self.vision_range = None
        self.max_vision_angle = None
        self.vision_resolution = None
        self.layer_attractive = None
        self.attractive_constant = 75
        self.reached_goal_distance = 0.2
        self.min_index_x = None
        self.max_index_x = None
        self.min_index_y = None
        self.max_index_y = None
        self.drone_matrix = None
        self.last_observation = None
        self.prev_explored_cells = None
        self.reduced_observation = None

    def load_MATLAB_map(self, map_path):
        mat_file = sio.loadmat(map_path)
        map_structure = mat_file['map']

        self.map_dimension_x = map_structure['dimension_x'][0][0][0][0] - 0.01
        self.map_dimension_y = map_structure['dimension_y'][0][0][0][0] - 0.01
        if map_structure['resolution_x'][0][0][0][0] != map_structure['resolution_y'][0][0][0][0]:
            os.error('ERROR: resolution on x_coordinate and y_coordinate are different [Environment\load MATLAB map]')
        self.map_resolution = map_structure['resolution_x'][0][0][0][0]
        self.N_cells_x = map_structure['N_cells_x'][0][0][0][0]
        self.N_cells_y = map_structure['N_cells_y'][0][0][0][0]
        self.obstacle_map = map_structure['obstacle_map'][0][0]
        self.obstacle_map = self.obstacle_map.T
        self.min_index_x = 0
        self.max_index_x = self.N_cells_x - 1
        self.min_index_y = 0
        self.max_index_y =  self.N_cells_y - 1

    def reset(self, drone_list):
        # load next map
        self.map_counter += 1
        if self.map_counter > self.max_map_number:
            self.map_counter = 1
        # reset each drone
        n_drones = len(drone_list)
        self.last_observation = np.empty([n_drones, self.N_cells_x, self.N_cells_y])
        self.prev_explored_cells = np.zeros([n_drones])
        initial_state = np.empty([n_drones, self.coverage_observation_size, self.coverage_observation_size])
        for drone in drone_list:
            drone.orientation = 0
            drone.step_number = 0
            drone.set_initial_position([10, 10])
            drone.position_history = np.empty([0,2], dtype='float16')
            drone.goal_history = np.empty([0,2], dtype='float16')
            drone.obstacle_matrix = np.zeros((drone.n_cell_x, drone.n_cell_y))
            drone.potential_field_map = np.zeros((drone.n_cell_x, drone.n_cell_y))
            drone.layer_attractive = np.zeros((drone.n_cell_x, drone.n_cell_y))
            drone.covered_area_matrix = np.zeros((drone.n_cell_x, drone.n_cell_y))
            drone.covered_area_matrix_single_drone = np.zeros((drone.n_cell_x, drone.n_cell_y))
            drone.explored_cell_number = 0
            drone.single_coverage = [0]
            drone.obstacle_list = np.empty([0, 2], dtype='float16')
            drone.newfound_obstacle_list = np.empty([0, 2], dtype='float16')
            drone.newfound_obstacle_share_list = np.empty([0, 2])
            drone.mobile_obstacle_last_position = np.empty([0, 3])
            drone.other_drones_current_position = None
            drone.goal_updated = False  # signals that the goal has changed with respect to the previous one and that the pot. field layer must be updated
            drone.goal = [0, 0]
            drone.old_goal = drone.goal
            drone.import_map_properties(self)
            drone.generate_border_obstacles()
            drone.lidar_operation_mode()
            drone.detect_obstacles()
            drone.share_covered_area(drone_list)
            drone.update_fixed_obstacles(drone_list)
            drone.normal_operation_mode()

        for drone in drone_list:
            # compute initial state of each drone. This must be done_episode AFTER the initialization of each drone has been completed
            ID = drone.drone_ID
            initial_state[drone.drone_ID, :, :] = self.coverage_observe(drone_list, ID)
        return initial_state

    def compute_reward(self, drone, cumulative_explored_cells):
        if self.obstacle_map[drone.index_position[0]][drone.index_position[1]] == 1:
            r = - 5
        elif self.last_observation[drone.drone_ID, drone.pos2index(drone.goal[0]), drone.pos2index(drone.goal[1])] == 1:
            r = - 2
        elif self.last_observation[drone.drone_ID, drone.pos2index(drone.goal[0]), drone.pos2index(drone.goal[1])] > 0:
            r = - 0.5
        else:
            if cumulative_explored_cells[drone.drone_ID] - self.prev_explored_cells[drone.drone_ID] >= 200:
                r = (cumulative_explored_cells[drone.drone_ID] - self.prev_explored_cells[drone.drone_ID]) / 600 + 1
            else:
                r = -1
        self.prev_explored_cells[drone.drone_ID] = cumulative_explored_cells[drone.drone_ID]
        return r

    def isdone_step(self, drone):
        done = False
        # if obstacle is hit, end episode
        if self.obstacle_map[drone.index_position[0]][drone.index_position[1]] == 1:
            done = True
        # if goal is reached, end episode
        distance_x = drone.goal[0] - drone.position[0]
        distance_y = drone.goal[1] - drone.position[1]
        if np.hypot(distance_x, distance_y) <= self.reached_goal_distance:
            done = True
        # substep max number has NOT to be checked since substeps are executed as a "for" loop
        return done

    def coverage_observe(self, drone_list, ID):
        self.reduced_observation = np.ones([self.coverage_observation_size, self.coverage_observation_size])
        # observation = np.zeros([self.N_cells_x, self.N_cells_y])
        observation = np.zeros([self.coverage_observation_size, self.coverage_observation_size])
        observation += drone_list[ID].covered_area_matrix * 2
        observation += drone_list[ID].obstacle_matrix * 2
        for x_idx, y_idx, _ in drone_list[ID].mobile_obstacle_last_position:
                self.matrix_placer(observation, self.drone_matrix, x_idx, y_idx)
        observation = np.clip(observation, 0, 4)
        observation /= 4
        observation = np.round(observation, 2)
        self.last_observation[ID, :, :] = observation
        # self.matrix_placer(self.reduced_observation, observation, drone_list[ID].index_position[0], drone_list[ID].index_position[1])
        # return self.reduced_observation
        return observation

    def matrix_placer(self, main_matrix, matrix, index_position_x, index_position_y):
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
        main_matrix[min_x:max_x, min_y:max_y] += matrix[min_x_A:max_x_A, min_y_A:max_y_A]

    def set_drone_influence_matrix(self, matrix_dimension):
        # self.drone_matrix = np.ones([matrix_dimension, matrix_dimension])
        peak_value = 2
        mean = 0
        sigma = 1/1.5         # variance --> increase the denominator to obtain a smaller repulsive area around the obstacle
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

    def set_max_step_number(self, N):
        self.max_step_number = int(N)

    def set_map_max_number(self, bypass_flag, N):
        if bypass_flag:
            self.max_map_number = N
        else:
            _, _, files_list = next(os.walk(self.map_folder_path))
            self.max_map_number = len(files_list)
        print("training set contains " + str(self.max_map_number) + " maps")

    def set_map_folder(self, path):
        self.map_folder_path = path

    def set_coverage_observation_size(self, N):
        self.coverage_observation_size = N
        self.observation_space = spaces.Box(low=np.zeros([self.coverage_observation_size, self.coverage_observation_size]),
                                            high=np.ones([self.coverage_observation_size, self.coverage_observation_size]),
                                            dtype=np.float32)

    def visualize(self, episode_number, show_flag, save_flag, figure_path, drone_list, reward):
        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
        # fig, (ax0) = plt.subplots(1, 1)
        fig.dpi = 150
        fig.suptitle('episode N° ' + str(episode_number) + ' | ' + 'Average reward: ' + str(np.round(reward, 2)))

        ax0.set_xlim(0, self.map_dimension_x)
        ax0.set_ylim(0, self.map_dimension_y)
        ax0.set_aspect(1)
        ax0.plot(drone_list[0].position_history[:,0], drone_list[0].position_history[:,1], 'k-')
        ax0.plot(drone_list[0].goal[0], drone_list[0].goal[1], 'bx')
        ax0.plot(drone_list[0].position_history[0, 0], drone_list[0].position_history[0, 1], 'r.', markersize=3)
        ax0.plot(drone_list[0].goal_history[:, 0], drone_list[0].goal_history[:, 1], 'rx', markersize=3)
        heatmap.draw_heatmap(self.last_observation[0, :, :], self, ax0)
        # heatmap.draw_heatmap(drone_list[0].covered_area_matrix, self, ax0)

        ax1.set_xlim(0, self.map_dimension_x)
        ax1.set_ylim(0, self.map_dimension_y)
        ax1.set_aspect(1)
        ax1.plot(drone_list[1].position_history[:,0], drone_list[1].position_history[:,1], 'k-')
        ax1.plot(drone_list[1].goal[0], drone_list[1].goal[1], 'bx')
        ax1.plot(drone_list[1].position_history[0, 0], drone_list[1].position_history[0, 1], 'r.', markersize=3)
        ax1.plot(drone_list[1].goal_history[:, 0], drone_list[1].goal_history[:, 1], 'rx', markersize=3)
        heatmap.draw_heatmap(self.last_observation[1, :, :], self, ax1)
        # heatmap.draw_heatmap(drone_list[1].covered_area_matrix, self, ax1)
        #
        ax2.set_xlim(0, self.map_dimension_x)
        ax2.set_ylim(0, self.map_dimension_y)
        ax2.set_aspect(1)
        ax2.plot(drone_list[2].position_history[:,0], drone_list[2].position_history[:,1], 'k-')
        ax2.plot(drone_list[2].goal[0], drone_list[2].goal[1], 'bx')
        ax2.plot(drone_list[2].position_history[0, 0], drone_list[2].position_history[0, 1], 'r.', markersize=3)
        ax2.plot(drone_list[2].goal_history[:, 0], drone_list[2].goal_history[:, 1], 'rx', markersize=3)
        heatmap.draw_heatmap(self.last_observation[2, :, :], self, ax2)
        # heatmap.draw_heatmap(drone_list[2].covered_area_matrix, self, ax2)

        # ax3.set_xlim(0, self.map_dimension_x)
        # ax3.set_ylim(0, self.map_dimension_y)
        # ax3.set_aspect(1)
        # ax3.plot(drone_list[3].position_history[:, 0], drone_list[3].position_history[:, 1], 'k-')
        # ax3.plot(drone_list[3].goal[0], drone_list[3].goal[1], 'rx')
        # ax3.plot(drone_list[3].position_history[0, 0], drone_list[3].position_history[0, 1], 'r.', markersize=3)
        # # heatmap.draw_heatmap(self.last_observation[3, :, :], self, ax3)
        # heatmap.draw_heatmap(drone_list[3].covered_area_matrix_single_drone, self, ax3)

        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])

        if show_flag:
            plt.show(block=False)
            plt.pause(2)

        if save_flag:
            plt.savefig(figure_path)

        plt.close()
