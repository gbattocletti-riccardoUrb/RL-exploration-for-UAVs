import numpy as np
import warnings
import matplotlib.pyplot as plt
from operator import add
import scipy.io as sio
import math
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

""" 
The MAP class is a representation of the real world in which the drone will move. The drone, in the final simulation, 
should have no access to the map class informations. The drone moves and its perception ability is simulated by looking 
for nearby obstacles in the map class.
"""


class Map:
    def __init__(self, x_resolution = 1, y_resolution = 1):
        self.x_dimension = None          # NOTE: the dimensions must be given as 9.99 NOT 10
        self.y_dimension = None
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution
        self.obstacle_list = np.empty([0, 2])
        self.kmeans_center = np.empty([0, 2])
        self.kmeans_center_booked = np.empty([0, 2])
        self.obstacle_map = None



    def __str__(self):
        return "The map size is %.2fx%.2f [m] and the elementary cell dimension is %.2fx%.2f [m]. \nThere are currently %i obstacle points." % (self.x_dimension, self.y_dimension, self.x_resolution, self.y_resolution, self.obstacle_list.shape[0])

    def set_map_size(self, map_dimension_x, map_dimension_y):
        self.x_dimension = map_dimension_x
        self.y_dimension = map_dimension_y

    def set_map_resolution(self, map_resolution_x, map_resolution_y):
        self.x_resolution = map_resolution_x
        self.y_resolution = map_resolution_y

    def generate_obstacle_points(self, n_obstacles, **kwargs):
        if kwargs.items():
            for key, value in kwargs.items():
                if key == "positions":                  # "positions" must be a n*2 column array containing the x_coordinate, y_coordinate cohordinates of each obstacle
                    if len(value) > n_obstacles:
                        self.obstacle_list = np.append(self.obstacle_list, value[1: n_obstacles, :]) # take only the first lines
                    elif len(value) == n_obstacles:
                        self.obstacle_list = value
                    elif len(value) < n_obstacles:
                        self.obstacle_list = value
                        n_extra_obstacles = n_obstacles - len(value)
                        extra_obstacles_x = np.random.randint(0, self.x_dimension, [n_extra_obstacles, 1])
                        extra_obstacles_y = np.random.randint(0, self.y_dimension, [n_extra_obstacles, 1])
                        extra_obstacles = np.append(extra_obstacles_x, extra_obstacles_y, axis = 1)
                        self.obstacle_list = np.append(self.obstacle_list, extra_obstacles, axis = 0)       # column matrix with 2 columns (x_coordinate, y_coordinate)
                else:
                    warnings.warn("Invalid **kwarg in Map/generate_obstacle_circular")
        else:
            obstacles_x = np.random.randint(0, self.x_dimension, [n_obstacles, 1])
            obstacles_y = np.random.randint(0, self.y_dimension, [n_obstacles, 1])
            obstacles = np.append(obstacles_x, obstacles_y, axis = 1)
            self.obstacle_list = np.append(self.obstacle_list, obstacles, axis = 0)

    def generate_border_obstacles(self):
        nx = int(round(self.x_dimension/self.x_resolution))         # n° of points along x_coordinate
        ny = int(round(self.y_dimension/self.y_resolution))         # n° of points along y_coordinate
        max_idx_x = nx - 1                                          # index of the max point along x_coordinate
        max_idx_y = ny - 1                                          # index of the max point along y_coordinate
        x_vec = np.linspace(0, max_idx_x*self.x_resolution, nx, endpoint=True).reshape(nx, 1)    # column vector of indexes of the obstacles along x_coordinate
        y_vec = np.linspace(0, max_idx_y*self.y_resolution, ny, endpoint=True).reshape(ny, 1)
        const_vec_x = np.ones([nx, 1])                      # constant column vector (vector of ones)
        const_vec_y = np.ones([ny, 1])
        border_obstacles = np.append(x_vec, const_vec_x*0, axis=1)
        border_obstacles = np.append(border_obstacles, np.append(x_vec, const_vec_x*max_idx_y*self.y_resolution, axis=1), axis=0)
        border_obstacles = np.append(border_obstacles, np.append(const_vec_y*0, y_vec, axis=1), axis=0)
        border_obstacles = np.append(border_obstacles, np.append(const_vec_y*max_idx_x*self.x_resolution, y_vec, axis=1), axis=0)
        self.obstacle_list = np.append(self.obstacle_list, border_obstacles, axis=0)

    def generate_rectangular_obstacle(self, center_x, center_y, side_x, side_y):
        n_points_side_x = math.floor(int(round(side_x/self.x_resolution)) / 2) * 2 + 1
        n_points_side_y = math.floor(int(round(side_y/self.y_resolution)) / 2) * 2 + 1  # ensure that n° of points is odd
        step_x = side_x/n_points_side_x
        step_y = side_y/n_points_side_y
        # if Guido van Rossum saw the code below he would come and beat us up
        starting_point = [center_x - step_x * (n_points_side_x - 1) / 2, center_y - step_y * (n_points_side_y - 1) / 2]
        for ii in range(0, n_points_side_x):
            # print(list(map(add, starting_point, [step_x * ii, 0])))
            self.obstacle_list = np.vstack((self.obstacle_list, list(map(add, starting_point, [step_x * ii, 0]))))
        for ii in range(0, n_points_side_y):
            self.obstacle_list = np.vstack((self.obstacle_list, list(map(add, starting_point, [0, step_y * ii]))))
        starting_point = [center_x + step_x * (n_points_side_x - 1) / 2, center_y + step_y * (n_points_side_y - 1) / 2]
        for ii in range(0, n_points_side_x):
            self.obstacle_list = np.vstack((self.obstacle_list, list(map(add, starting_point, [-step_x * ii, 0]))))
        for ii in range(0, n_points_side_y):
            self.obstacle_list = np.vstack((self.obstacle_list, list(map(add, starting_point, [0, -step_y * ii]))))

    def plot_map(self):
        dpi = 100
        figure_size_pixel = 700, 500
        figure_size = figure_size_pixel[0] / dpi, figure_size_pixel[1] / dpi
        plt.figure(1, figure_size, dpi)
        nx = int(round(self.x_dimension / self.x_resolution))  # n° of points along x_coordinate
        ny = int(round(self.y_dimension / self.y_resolution))  # n° of points along y_coordinate
        # X = np.linspace(0, nx*self.x_resolution, nx+1, endpoint=True)
        # Y = np.linspace(0, ny*self.y_resolution, ny+1, endpoint=True)
        C = np.zeros([nx, ny])  # initialize color matrix
        for x, y in self.obstacle_list:
            # idx_x = int(math.floor(x_coordinate/self.x_resolution))
            # idx_y = int(math.floor(y_coordinate/self.y_resolution))
            idx_x = int(round(x/self.x_resolution))
            idx_y = int(round(y/self.y_resolution))
            C[idx_x, idx_y] = 1
        # plt.pcolormesh(X, Y, C.T, cmap=plt.cm.Greys, zorder=1)
        ax = plt.gca()
        C = np.flip(C,1)
        ax.imshow(C.T, extent=[0, self.x_dimension, 0, self.y_dimension])
        ax.set(xlim=(0, self.x_dimension), ylim=(0, self.y_dimension))
        aspect_ratio = 1
        # aspect_ratio = self.y_dimension/self.x_dimension
        # aspect_ratio = self.x_resolution/self.y_resolution
        ax.set_aspect(aspect_ratio)

    def convert_MATLAB_map(self, map_path):
        mat_file = sio.loadmat(map_path)
        map_structure = mat_file['map']
        self.obstacle_map = map_structure['obstacle_map'][0][0]
        self.obstacle_map = self.obstacle_map.T

        dim_x = len(self.obstacle_map)
        dim_y = len(self.obstacle_map[1])

        self.x_dimension = map_structure['dimension_x'][0][0][0][0] #- 0.01
        self.y_dimension = map_structure['dimension_y'][0][0][0][0] #- 0.01
        self.x_resolution = map_structure['resolution_x'][0][0][0][0]
        self.y_resolution = map_structure['resolution_y'][0][0][0][0]

        for ii in range(0, dim_x):
            for jj in range(0, dim_y):
                if self.obstacle_map[ii][jj] == 1:
                    index_x = ii * self.x_resolution
                    index_y = jj * self.y_resolution
                    self.obstacle_list = np.vstack([self.obstacle_list, np.array([index_x, index_y])])

    def coverage_voronoi(self, drone):
        self.kmeans_center = np.empty([0, 2])
        new_goals = np.empty([0, 2])
        # sse = []
        # silhouette_coefficients = []
        unknown_points = np.vstack((drone.y_cohordinate_matrix[drone.covered_area_matrix==0], drone.x_cohordinate_matrix[drone.covered_area_matrix==0])).T
        # for n_clusters in range(2, 10):
        #     kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++', n_init=10).fit(unknown_points)
        #     sse.append(kmeans.inertia_)
            # score = silhouette_score(unknown_points, kmeans.labels_)
            # silhouette_coefficients.append(score)
        # kl = KneeLocator(range(2, 10), sse, curve = "convex", direction = "decreasing")
        # n_clusters = kl.elbow
        # n_clusters = np.argmax(silhouette_coefficients) + 2
        # print(n_clusters)
        n_clusters = 10
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++', n_init=10).fit(unknown_points)
        exploration_area_centers = kmeans.cluster_centers_

        for new_goal_position in exploration_area_centers:
            if (np.where((np.round(self.kmeans_center) == np.round(new_goal_position)).all(axis=1))[0]).size == 0 and (np.where((np.round(self.kmeans_center_booked) == np.round(new_goal_position)).all(axis=1))[0]).size == 0 and drone.covered_area_matrix[drone.pos2index(new_goal_position[0]), drone.pos2index(new_goal_position[1])] == 0:    # if the goal is not already present or booked and the area is not explored
                new_goals = np.vstack([new_goals, new_goal_position])
            elif self.kmeans_center.size == 0  and (np.where((np.round(self.kmeans_center_booked) == np.round(new_goal_position)).all(axis=1))[0]).size == 0: #and drone.covered_area_matrix[drone.pos2index(new_goal_position[0]), drone.pos2index(new_goal_position[1])] == 0:  # if vector is empty and area is not explored
                new_goals = np.vstack([new_goals, new_goal_position])
            else:
                pass
        self.kmeans_center = np.vstack([self.kmeans_center, new_goals])                 # add the new goals to the possible goal
        for goal_position in self.kmeans_center:
            if drone.covered_area_matrix[drone.pos2index(goal_position[0]), drone.pos2index(goal_position[1])] != 0:
                goal_idx = np.where((self.kmeans_center == goal_position).all(axis=1))[0]
                self.kmeans_center = np.delete(self.kmeans_center, goal_idx, axis=0)

        # find the closer goal to the drone
        x_distances = np.ones(len(self.kmeans_center))*drone.position[0] - np.array([self.kmeans_center[:, 0]])
        y_distances = np.ones(len(self.kmeans_center))*drone.position[1] - np.array([self.kmeans_center[:, 1]])
        distances = np.hypot(x_distances, y_distances)[0]
        goal_idx = np.where(distances == np.min(distances))[0][0]
        goal = self.kmeans_center[goal_idx, :]
        print(goal)
        self.kmeans_center_booked = np.vstack([self.kmeans_center_booked, goal])
        self.kmeans_center = np.delete(self.kmeans_center, goal_idx, axis=0)
        return goal