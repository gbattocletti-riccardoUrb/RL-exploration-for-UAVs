import numpy as np
import matplotlib.pyplot as plt

import color_utils

my_colors = ['#04213B', '#1177B3', '#E7CC3C', '#DF9A12']
rgbs = color_utils.hex_to_rgb_color_list(my_colors)
my_cmap = color_utils.blended_cmap(rgbs)

def draw_heatmap(heatmap_data, map_obj):
    # x_axis = np.round(np.linspace(0, map_obj.x_dimension, int(map_obj.x_dimension / map_obj.x_resolution)+1, endpoint=True), 2)
    # y_axis = np.round(np.linspace(0, map_obj.y_dimension, int(map_obj.y_dimension / map_obj.y_resolution)+1, endpoint=True), 2)
    # heatmap_data = np.flip(heatmap_data, 0)
    nx = int(round(float(map_obj.map_dimension_x / map_obj.map_resolution)))  # n° of points along x_coordinate
    ny = int(round(float(map_obj.map_dimension_y / map_obj.map_resolution)))  # n° of points along y_coordinate
    X = np.linspace(0, nx * map_obj.map_resolution, nx + 1, endpoint=True)
    Y = np.linspace(0, ny * map_obj.map_resolution, ny + 1, endpoint=True)
    plt.register_cmap(cmap=plt.cm.winter)
    plt.set_cmap(plt.cm.winter)
    plt_color_map = plt.pcolormesh(X, Y, heatmap_data.T, shading='auto', zorder=0)
    return plt_color_map