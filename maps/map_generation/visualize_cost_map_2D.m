%% POTENTIAL MAP 2D VISUALIZATION

clc;
close all;
clear variables;
format compact;
format bank;

% map_path = '..\0. map_archive\old_maps\small_maps_test_1\map_1.mat';
% map_path = '..\0. map_archive\case_study_map\map_layer_1_cone.mat';
% map_path = 'maps\map_1.mat';
[file_name, file_path] = uigetfile('C:\Giampo\università\PoliTO\Tesi\Dropbox (Politecnico Di Torino Studenti)\Tesi [Battocletti-Urban]\Maps');
map_path = fullfile(file_path, file_name);
map = load_map(map_path);

map.start_index_x = ceil(map.starting_position_x/map.resolution_x);
map.start_index_y = map.N_cells_y - ceil(map.starting_position_y/map.resolution_y) + 1;
map.goal_index_x = ceil(map.goal_position_x/map.resolution_x);
map.goal_index_y = map.N_cells_y - ceil(map.goal_position_y/map.resolution_y) + 1;

custom_color_map = [1 1 1; 
                    0 0 0];

figure()
    sgtitle(map_path)
    p1 = subplot(121);
        imagesc((1:map.N_cells_y)-0.5, (1:map.N_cells_x)-0.5, flip(map.obstacle_map))
        hold on
        plot(map.start_index_x, map.start_index_y, 'k.', 'MarkerSize', 20, 'linew', 1.4)
        plot(map.goal_index_x, map.goal_index_y, 'rx', 'MarkerSize', 10, 'linew', 1.4)
        axis square
    p2 = subplot(122);
        imagesc((1:map.N_cells_y)-0.5, (1:map.N_cells_x)-0.5, flip(map.cost_map))
        hold on
        plot(map.start_index_x, map.start_index_y, 'k.', 'MarkerSize', 20, 'linew', 1.4)
        plot(map.goal_index_x, map.goal_index_y, 'rx', 'MarkerSize', 10, 'linew', 1.4)
        axis square
    colormap(p1, custom_color_map)
    colormap(p2, 'parula')