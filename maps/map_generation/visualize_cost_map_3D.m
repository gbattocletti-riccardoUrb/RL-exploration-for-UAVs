%% POTENTIAL MAP 3D VISUALIZATION

clc;
close all;
clear variables;
format compact;
format bank;

% n = 2;
map_path = '..\0. map_archive\old_maps\small_maps_complexity_1\map_1.mat';
% map_path = '..\0. map_archive\training_set_5\map_1.mat';

map = load_map(map_path);

max_value = 1000;
[X, Y] = meshgrid(1:map.N_cells_x, 1:map.N_cells_y);

custom_colormap = [];
figure()
    hold on
    surf(X, Y, map.cost_map)
    colormap(custom_colormap)
    s.EdgeColor = 'none';
    plot3(map.goal_position_index_x, map.goal_position_index_y, max_value, 'o', 'MarkerFaceColor', [1,0,0], 'MarkerEdgeColor', [0,0,0] )
    plot3(map.starting_position_index_x, map.starting_position_index_y, max_value, 'o', 'MarkerFaceColor', [0,0,0], 'MarkerEdgeColor', [0,0,0] )