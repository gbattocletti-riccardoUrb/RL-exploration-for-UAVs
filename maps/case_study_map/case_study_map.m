%% Case study map generation

clc;
close all;
clear variables;
format compact;
format bank;

%% general properties

map = struct();
map.dimension_x = 20.2;
map.dimension_y = 10.2;
map.resolution_x = 0.1;
map.resolution_y = 0.1;
map.N_cells_x = 202;
map.N_cells_y = 102;
map.starting_position_x = 1;
map.starting_position_y = 1;
map.starting_position_index_x = 1;
map.starting_position_index_y = 1;
map.goal_position_x = 1;
map.goal_position_y = 1;
map.goal_position_index_x = 1;
map.goal_position_index_y = 1;


%% obstacle map initialization

m = zeros(102, 202, 6);

% external borders
m(1, :, :) = 1;
m(end, :, :) = 1;
m(:, 1, :) = 1;
m(:, end, :) = 1;

%% obstacle generation

% ostacolo in basso a sinistra
m(87:101, 12:36, 1:2) = 1;
m(87:101, 20:27, 3:4) = 1;

% ostacolo al centro a sinistra
m(37:66, 19:58, 1:2) = 1;
m(47:56, 29:48, 1:2) = 0;
m(57:66, 19:58, 3:4) = 1;

% ostacolo in alto a sinistra
m(12:26, 32:71, 1:2) = 1;

% ostacolo in alto al centro (C)
m(2:21, 82:121, 1:4) = 1;
m(2:11, 92:111, 1:4) = 0;

% ostacolo a L in centro
m(37:56, 109:128, 1:2) = 1;
m(37:46, 109:118, 1:2) = 0;

% cubetto in basso al centro
m(77:91, 114:128, 1:2) = 1;

% ostacolo strambo al centr
m(62:91, 74:103, 1:2) = 1;
m(62:76, 84:93, 1:2) = 0;
m(62:81, 74:83, 3:4) = 1;
m(77:91, 84:93, 3:4) = 1;

% ostacolo lungo in alto verso destra
m(12:21, 132:161, 1:2) = 1;

% C grande al centro
m(32:71, 139:158, 1:4) = 1;
m(42:61, 149:158, 1:4) = 0;

% piramide weird (podio) in basso a destra
m(82:91, 139:168, 1) = 1;
m(82:91, 149:158, 2:3) = 1;
m(82:91, 159:168, 2) = 1;

% colonna in basso a destra
m(87:91, 187:191, 1:6) = 1;

% blocco in alto a destra
m(12:21, 172:191, 1:6) = 1;

% gazebo
m(32:36, 170:174, 1:5) = 1;
m(32:36, 187:191, 1:5) = 1;
m(67:71, 170:174, 1:5) = 1;
m(67:71, 187:191, 1:5) = 1;
m(32:71, 170:191,6) = 1;

%% save maps

for ii = 1:6
    map_name = sprintf('layer_%i', ii);
    save_path = sprintf('..\\0. map_archive\\case_study_map\\%s', map_name);
    map.obstacle_map = m(:, :, ii);
    map.cost_map = map.obstacle_map;
    save(save_path, 'map')
end

%% 2D plot

% map_path = '..\0. map_archive\case_study_map\layer_1.mat';
% map = load_map(map_path);
% 
% map.start_index_x = ceil(map.starting_position_x/map.resolution_x);
% map.start_index_y = map.N_cells_y - ceil(map.starting_position_y/map.resolution_y) + 1;
% map.goal_index_x = ceil(map.goal_position_x/map.resolution_x);
% map.goal_index_y = map.N_cells_y - ceil(map.goal_position_y/map.resolution_y) + 1;

c_map = [1 1 1;
         0 0 0];
figure()
    subplot(2,3,1)
        hold on
        colormap(c_map);
        imagesc((1:map.N_cells_x)-1.5, (1:map.N_cells_y)-1.5, flip(m(:,:,1)))
        axis([-1, 201, -1, 101])
        title('layer 1')
        xlabel('x')
        ylabel('y')
	subplot(2,3,2)
        hold on
        colormap(c_map);
        imagesc((1:map.N_cells_x)-1.5, (1:map.N_cells_y)-1.5, flip(m(:,:,2)))
        axis([-1, 201, -1, 101])
        title('layer 2')
        xlabel('x')
        ylabel('y')
    subplot(2,3,3)
        hold on
        colormap(c_map);
        imagesc((1:map.N_cells_x)-1.5, (1:map.N_cells_y)-1.5, flip(m(:,:,3)))
        axis([-1, 201, -1, 101])
        title('layer 3')
        xlabel('x')
        ylabel('y')
    subplot(2,3,4)
        hold on
        colormap(c_map);
        imagesc((1:map.N_cells_x)-1.5, (1:map.N_cells_y)-1.5, flip(m(:,:,4)))
        axis([-1, 201, -1, 101])
        title('layer 4')
        xlabel('x')
        ylabel('y')
    subplot(2,3,5)
        hold on
        colormap(c_map);
        imagesc((1:map.N_cells_x)-1.5, (1:map.N_cells_y)-1.5, flip(m(:,:,5)))
        axis([-1, 201, -1, 101])
        title('layer 5')
        xlabel('x')
        ylabel('y')
    subplot(2,3,6)
        hold on
        colormap(c_map);
        imagesc((1:map.N_cells_x)-1.5, (1:map.N_cells_y)-1.5, flip(m(:,:,6)))
        axis([-1, 201, -1, 101])
        title('layer 6')
        xlabel('x')
        ylabel('y')
        
%% 3D plot

[X, Y, Z] = ind2sub(size(m(2:end-1, 2:end-1, :)),find(m(2:end-1, 2:end-1, :) == 1));

figure()
    plot3(X, Y, Z, 'k.', 'MarkerSize', 30);