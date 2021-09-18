import numpy as np
import scipy.io as sio
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import os
import gym
from gym import spaces
import heatmap

gym.logger.set_level(40)

class Environment:

    def __init__(self):
        self.position_x = None
        self.position_y = None
        self.index_x = None
        self.index_y = None
        self.map_dimension_x = None
        self.map_dimension_y = None
        self.map_resolution = None
        self.N_cells_x = None
        self.N_cells_y  = None
        self.obstacle_map = None
        self.cost_map = None
        self.goal_index_x = None
        self.goal_index_y = None
        self.goal_position_x = None
        self.goal_position_y = None
        self.starting_position_index_x = None
        self.starting_position_index_y = None
        self.starting_position_x = None
        self.starting_position_y = None
        self.position_history = np.empty([0, 2])
        self.obstacle_list = np.empty([0, 2])
        self.speed = None
        self.min_angle = 0
        self.max_angle = 359.99
        self.step_number = 0
        self.total_steps = 0
        self.episode_number = 0
        self.max_step_number = None
        self.map_counter = 0
        self.max_map_number = None
        self.map_folder_path = ''
        self.current_potential = 0
        self.previous_potential = 0
        self.obstacle_hit_penality = None
        self.goal_reached_prize = None
        self.time_penality = None
        self.potential_descent_prize = None
        self.angle_change_penality = None
        self.min_index_x = None
        self.max_index_x = None
        self.min_index_y = None
        self.max_index_y = None
        self.observation_size = None
        self.action_space = spaces.Box(low=math.radians(self.min_angle), high=math.radians(self.max_angle), shape=(1,), dtype=np.float32)
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
        self.newfound_obstacle_list = np.empty([0, 2])
        self.x_cohordinate_matrix = None
        self.y_cohordinate_matrix = None
        self.layer_attractive = None
        self.attractive_constant = 50


    def set_step_number(self, N):
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

    def set_observation_size(self, args):
        self.observation_size = args.observation_size
        self.observation_space = spaces.Box(low=np.ones([self.observation_size, self.observation_size])*args.lower_bound,
                                            high=np.ones([self.observation_size, self.observation_size]*args.upper_bound),
                                            dtype=np.float32)

    def set_speed(self, N):
        self.speed = N

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
        self.goal_index_x = map_structure['goal_position_index_x'][0][0][0][0]
        self.goal_index_y = map_structure['goal_position_index_y'][0][0][0][0]
        self.goal_position_x = map_structure['goal_position_x'][0][0][0][0]
        self.goal_position_y = map_structure['goal_position_y'][0][0][0][0]
        # self.goal_position_x = 10
        # self.goal_position_y = 10
        # goal_position = [1 + np.random.random() * (self.map_dimension_x - 2), 1 + np.random.random() * (self.map_dimension_y - 2)]
        # self.goal_position_x = goal_position[0]
        # self.goal_position_y = goal_position[1]
        self.starting_position_index_x = map_structure['starting_position_index_x'][0][0][0][0]
        self.starting_position_index_y = map_structure['starting_position_index_y'][0][0][0][0]
        self.starting_position_x = map_structure['starting_position_x'][0][0][0][0]
        self.starting_position_y = map_structure['starting_position_y'][0][0][0][0]
        self.obstacle_map = map_structure['obstacle_map'][0][0]
        # self.cost_map = map_structure['cost_map'][0][0]
        self.obstacle_map = self.obstacle_map.T
        # self.cost_map = self.cost_map.T

        self.min_index_x = 0
        self.max_index_x = self.N_cells_x - 1
        self.min_index_y = 0
        self.max_index_y = self.N_cells_y - 1

        x = np.linspace(0, self.map_dimension_x, self.N_cells_x)
        y = np.linspace(0, self.map_dimension_y, self.N_cells_y)
        self.x_cohordinate_matrix, self.y_cohordinate_matrix = np.meshgrid(x, y)

        # dim_x = len(self.obstacle_map)
        # dim_y = len(self.obstacle_map[1])
        #
        # for ii in range(1, dim_x):
        #     for jj in range(1, dim_y):
        #         if self.obstacle_map[ii][jj] == 1:
        #             np.vstack([self.obstacle_list, np.array([ii, jj])])

    def reset(self):
        # load next map
        self.map_counter += 1
        if self.map_counter > self.max_map_number:
            self.map_counter = 1
        current_map_path = os.path.join(self.map_folder_path, 'map_' + str(self.map_counter) + '.mat')
        self.load_MATLAB_map(current_map_path)
        self.compute_fixed_obstacle_matrix()
        self.obstacle_list = np.empty([0, 2])
        self.layer_attractive = np.zeros((self.N_cells_x, self.N_cells_y)).T
        self.cost_map = np.zeros((self.N_cells_x, self.N_cells_y))
        self.update_attractive_layer()
        # initialize stuff
        self.step_number = 0
        self.previous_direction = None
        # initial_position = [1 + np.random.random() * (self.map_dimension_x - 2),
        #                     1 + np.random.random() * (self.map_dimension_y - 2)]
        # self.starting_position_x = initial_position[0]
        # self.starting_position_y = initial_position[1]
        self.position_x = self.starting_position_x
        self.position_y = self.starting_position_y
        self.index_x = self.pos2index(self.position_x)
        self.index_y = self.pos2index(self.position_y)
        self.position_history = np.array([[self.position_x, self.position_y]])
        self.previous_potential = self.get_potential()
        self.detect_obstacles_LM()
        self.update_fixed_obstacles()
        initial_observation = self.observe()
        return initial_observation

    def step(self, action):
        # do action
        angle = np.clip(action, math.radians(self.min_angle), math.radians(self.max_angle))
        delta_x = self.speed * math.cos(angle)
        delta_y = self.speed * math.sin(angle)
        self.position_x = self.check_boundaries(self.position_x + delta_x, 'x_coordinate')
        self.position_y = self.check_boundaries(self.position_y + delta_y, 'y_coordinate')
        self.index_x = self.pos2index(self.position_x)
        self.index_y = self.pos2index(self.position_y)
        self.position_history = np.vstack([self.position_history, [self.position_x, self.position_y]])
        self.current_potential = self.get_potential()
        self.current_direction = angle
        # compute stuff and return it
        reward = self.compute_reward()
        self.detect_obstacles()
        self.update_fixed_obstacles()
        next_observation = self.observe()
        self.step_number += 1
        self.total_steps += 1
        done = self.isdone()
        self.previous_potential = self.current_potential
        self.previous_direction = self.current_direction
        return next_observation, reward, done, {}

    # def compute_reward(self):
    #     # r = 0
    #     self.obstacle_hit_penality = -10
    #     self.goal_reached_prize = 10
    #     self.time_penality = -0.2
    #     self.potential_descent_prize = -0.005
    #     distance_x = self.goal_position_x - self.position_x
    #     distance_y = self.goal_position_y - self.position_y
    #     if self.obstacle_map[self.index_x][self.index_y] == 1:
    #         r = self.obstacle_hit_penality
    #     elif np.hypot(distance_x, distance_y) <= 0.4:
    #         r = self.goal_reached_prize
    #     else:
    #         delta_pot = self.current_potential - self.previous_potential
    #         if delta_pot > 0:
    #             # r = delta_pot*self.potential_descent_prize
    #             r = -0.1
    #         else:
    #             r = 0
    #         # r -= self.step_number**2*0.0002
    #     return r

    # def compute_reward(self):                   # 2nd version, based only on APF
    #     self.goal_reached_prize = 20
    #     distance_x = self.goal_position_x - self.position_x
    #     distance_y = self.goal_position_y - self.position_y
    #     if np.hypot(distance_x, distance_y) <= 0.3:
    #         r = self.goal_reached_prize
    #     else:
    #         delta_pot = self.current_potential - self.previous_potential
    #         if delta_pot < 0:
    #             r = 1
    #         elif delta_pot > 70:
    #             r = -20
    #         elif delta_pot > 30:
    #             r = -10
    #         else:
    #             r = -0.5
    #         if self.previous_direction is not None:
    #             delta = abs(self.previous_direction - self.current_direction)
    #             delta_prime = abs(math.radians(360) - delta)
    #             delta_angle = min(delta, delta_prime)
    #             if delta_angle >= math.radians(90):
    #                 r -= 2
    #     return r

    def compute_reward(self):
        self.obstacle_hit_penality = -10
        self.goal_reached_prize = 20
        self.time_penality = -0.2
        self.potential_descent_prize = -20/200
        self.angle_change_penality = -1 / 4
        distance_x = self.goal_position_x - self.position_x
        distance_y = self.goal_position_y - self.position_y
        if self.obstacle_map[self.index_x][self.index_y] == 1:
            r = self.obstacle_hit_penality
        elif np.hypot(distance_x, distance_y) <= 0.3:
            r = self.goal_reached_prize
        else:
            delta_pot = self.current_potential - self.previous_potential
            if delta_pot < 0:
                r = self.time_penality + self.potential_descent_prize * delta_pot
            elif delta_pot > 100:
                r = self.time_penality + self.potential_descent_prize * delta_pot - 5
            else:
                r = -1
            if self.previous_direction is not None:
                delta = abs(self.previous_direction - self.current_direction)
                delta_prime = abs(math.radians(360) - delta)
                delta_angle = min(delta, delta_prime)
                # r += self.angle_change_penality * delta_angle**2
                if delta_angle >= math.radians(90):
                    r -= 2
        return r

    def isdone(self):
        done = False
        # check step against max_step_number
        if self.step_number  >= self.max_step_number:
            done = True
        # if obstacle is hit, end episode
        if self.obstacle_map[self.index_x][self.index_y] == 1:
            done = True
        # if goal is reached, end episode
        distance_x = self.goal_position_x - self.position_x
        distance_y = self.goal_position_y - self.position_y
        if np.hypot(distance_x, distance_y) <= 0.3:
            print('goal found')
            done = True
        return done

    def observe(self):
        observation = np.ones([self.observation_size, self.observation_size])*1000  # to create padding
        delta = (self.observation_size - 1) / 2

        min_x = int(max(self.index_x - delta, self.min_index_x))                # min & max indexes indicating the portion of the big matrix in which M has to be pasted
        max_x = int(min(self.index_x + delta + 1, self.max_index_x))
        min_y = int(max(self.index_y - delta, self.min_index_y))
        max_y = int(min(self.index_y + delta + 1, self.max_index_y))
        min_x_A = int(delta - (self.index_x - min_x))
        max_x_A = int(delta + (max_x - self.index_x))
        min_y_A = int(delta - (self.index_y - min_y))
        max_y_A = int(delta + (max_y - self.index_y))

        observation[min_x_A:max_x_A, min_y_A:max_y_A] = self.cost_map[min_x:max_x, min_y:max_y]
        observation = np.clip(observation, 0, 1000)         # clipping
        observation = observation - observation[int(self.observation_size - delta - 1), int(self.observation_size - delta - 1)] # differential state
        observation /= 1000                                 # normalization
        observation = np.round(observation, 2)              # rounding
        return observation

    def detect_obstacles(self):
        cell_vision_range = self.vision_range / self.map_resolution
        for sweep_angle in np.linspace(-self.max_vision_angle, self.max_vision_angle, self.vision_resolution):
            distance = 0
            sweep_angle_rad = math.radians(sweep_angle)
            angle = self.current_direction + sweep_angle_rad
            while distance <= cell_vision_range:
                idx_x = int(math.floor(self.index_x + distance * math.cos(angle)))
                idx_y = int(math.floor(self.index_y + distance * math.sin(angle)))
                if self.obstacle_map[idx_x, idx_y] == 1:
                    obstacle_position_x = idx_x * self.map_resolution
                    obstacle_position_y = idx_y * self.map_resolution
                    self.newfound_obstacle_list = np.vstack([self.newfound_obstacle_list, [obstacle_position_x, obstacle_position_y]])
                    break
                else:
                    distance += 1
        return

    def detect_obstacles_LM(self):
        cell_vision_range = self.vision_range / self.map_resolution
        for sweep_angle in np.linspace(-180, 180, 200):
            distance = 0
            sweep_angle_rad = math.radians(sweep_angle)
            angle = 0 + sweep_angle_rad
            while distance <= cell_vision_range:
                idx_x = int(math.floor(self.index_x + distance * math.cos(angle)))
                idx_y = int(math.floor(self.index_y + distance * math.sin(angle)))
                if self.obstacle_map[idx_x, idx_y] == 1:
                    obstacle_position_x = idx_x * self.map_resolution
                    obstacle_position_y = idx_y * self.map_resolution
                    if self.check_fixed_obstacle_position(obstacle_position_x, obstacle_position_y):
                        self.newfound_obstacle_list = np.vstack([self.newfound_obstacle_list, [obstacle_position_x, obstacle_position_y]])
                    break
                else:
                    distance += 1
        return

    def compute_fixed_obstacle_matrix(self):
        mean = 0
        sigma = 1/3         # variance --> increase the denominator to obtain a smaller repulsive area around the obstacle
        n_points_x = int(round(self.obstacle_safe_distance/self.map_resolution))      # computes dimension of the repulsive matrix along x_coordinate
        n_points_y = int(round(self.obstacle_safe_distance/self.map_resolution))      # computes dimension of the repulsive matrix along y_coordinate
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
        self.fixed_obstacle_matrix = dist_x * dist_y

    def update_fixed_obstacles(self):
        for x, y in self.newfound_obstacle_list:                # runs over the whole list of newly detected obstacles
            x_idx = int(math.floor(x/self.map_resolution))    # index of x_coordinate-cell corresponding to obstacle position
            y_idx = int(math.floor(y/self.map_resolution))    # index of y_coordinate-cell corresponding to obstacle position
            if self.check_fixed_obstacle_position(x, y):
                self.matrix_placer(self.cost_map, self.fixed_obstacle_matrix, x_idx, y_idx, +1)   # add the repulsive matrix to the global APF matrix
                self.obstacle_list = np.vstack([self.obstacle_list, np.array([x, y])])  # append the newfound obstacles to the complete obstacle list
        self.newfound_obstacle_list = np.empty([0, 2])    # empty newfound obstacle list -> it will be replenished during the next scanning phase

    def check_fixed_obstacle_position(self, x_idx, y_idx):
        found_idx = np.where((self.obstacle_list == [x_idx, y_idx]).all(axis=1))[0]          # check for obstacles in the same position along x_coordinate (1st column of the obstacle list array)
        if found_idx.size > 0:  # if there is any match i.e. there is already an obstacle in the same position, a "False" flag is returned and obstacle is ignored in "update_fixed_obstacles"
            return False
        else:
            return True

    def update_attractive_layer(self):
        self.cost_map -= self.layer_attractive.T   # removes contribute of the previous goal attractive field
        self.layer_attractive = self.attractive_constant * np.hypot(self.x_cohordinate_matrix - self.goal_position_x, self.y_cohordinate_matrix - self.goal_position_y)
        self.cost_map += self.layer_attractive.T

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

    def get_potential(self):
        potential = self.cost_map[self.index_x][self.index_y]
        return potential

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
        idx = int(math.floor(pos / self.map_resolution))
        return idx

    def index2pos(self, idx):
        pos = round(idx*self.map_resolution, 2)
        return pos

    def visualize(self, episode_number, show_flag, save_flag, figure_path, reward):
        fig = plt.figure()
        fig.dpi = 150
        ax = fig.add_subplot(111)
        ax.set_xlim(0, self.map_dimension_x)
        ax.set_ylim(0, self.map_dimension_y)
        ax.set_aspect(1)
        plt.title('episode N° ' + str(episode_number) + ' | ' + 'Episode steps: ' + str(self.step_number) + '|' + 'reward' + str(np.round(reward, 2)))
        ax.plot(self.position_history[:,0], self.position_history[:,1], 'k-')
        ax.plot(self.goal_position_x, self.goal_position_y, 'rx')
        ax.plot(self.starting_position_x, self.starting_position_y, 'r.', markersize=10)
        heatmap.draw_heatmap(self.cost_map, self)
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])

        if show_flag:
            plt.show(block=False)
            plt.pause(1.5)

        if save_flag:
            # print(figure_path)
            plt.savefig(figure_path)

        plt.close()
