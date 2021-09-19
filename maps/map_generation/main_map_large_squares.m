clc;
close all;
clear variables;

%% Variables initialization
n_final_map = 1;
folder_name = 'square_maps';
prompt = {'Indicate the x position of the goal', 'Indicate the y position of the goal', 'Indicate the x position of the starting position', 'Indicate the y position of the starting position'};
if ~exist(folder_name, 'dir')
   mkdir(folder_name)
end

% Map variables initialization
map_number = 1;
map_base = map_class;
map_base.set_dimension(50, 50);
map_base.set_resolution(0.1, 0.1);
map_base.generate_maps();
map_base.generate_border_obstacles();
map_base.generate_matrix();
x = linspace(0, map_base.dimension_x, map_base.N_cells_x);
y = linspace(0, map_base.dimension_y, map_base.N_cells_y);
[X,Y] = meshgrid(x, y);

% Obstacle generation 
complexity = 2;
mean_dimension = mean([map_base.dimension_x, map_base.dimension_y]);
min_dimension = min([map_base.dimension_x, map_base.dimension_y]);
n_obstacles_value = [mean_dimension/8, mean_dimension/8, mean_dimension, mean_dimension, mean_dimension/2];
n_obstacles = round(n_obstacles_value(complexity));

obstacle_position_matrix = [3, 4.5, 2, 5
                 2, 10, 4, 2
                 4, 16, 4, 4
                 7, 8.5, 2, 5
                 10.5, 1.5, 7, 3
                 14.5, 9.5, 3, 3
                 12, 17, 6, 2
                 19, 10.5, 2, 5
                 18.5, 1.5, 3, 3];
             
diagonal_distance = 15;
             
obstacle_position_matrix_2 = obstacle_position_matrix;           
obstacle_position_matrix_2(:, 1:2) = obstacle_position_matrix(:, 1:2) + diagonal_distance;
obstacle_position_matrix_3 = obstacle_position_matrix_2;           
obstacle_position_matrix_3(:, 1:2) = obstacle_position_matrix_2(:, 1:2) + diagonal_distance;
obstacle_position_matrix_4 = obstacle_position_matrix_3;           
obstacle_position_matrix_4(:, 1:2) = obstacle_position_matrix_3(:, 1:2) + diagonal_distance;
obstacle_position_matrix_5 = obstacle_position_matrix_4;           
obstacle_position_matrix_5(:, 1:2) = obstacle_position_matrix_4(:, 1:2) + diagonal_distance;
obstacle_position_matrix_6 = obstacle_position_matrix_5;           
obstacle_position_matrix_6(:, 1:2) = obstacle_position_matrix_5(:, 1:2) + diagonal_distance;
obstacle_position_matrix_7 = obstacle_position_matrix_6;           
obstacle_position_matrix_7(:, 1:2) = obstacle_position_matrix_6(:, 1:2) + diagonal_distance;
obstacle_position_matrix_8 = obstacle_position_matrix_7;           
obstacle_position_matrix_8(:, 1:2) = obstacle_position_matrix_7(:, 1:2) + diagonal_distance;
obstacle_position_matrix_9 = obstacle_position_matrix_8;           
obstacle_position_matrix_9(:, 1:2) = obstacle_position_matrix_8(:, 1:2) + diagonal_distance;
obstacle_position_matrix_10 = obstacle_position_matrix_9;           
obstacle_position_matrix_10(:, 1:2) = obstacle_position_matrix_9(:, 1:2) + diagonal_distance;
             
square_points = [obstacle_position_matrix;
                obstacle_position_matrix_2;
                obstacle_position_matrix_3;
%                 obstacle_position_matrix_4;
%                 obstacle_position_matrix_5;
%                 obstacle_position_matrix_6;
%                 obstacle_position_matrix_7;
%                 obstacle_position_matrix_8;
%                 obstacle_position_matrix_9;
%                 obstacle_position_matrix_10;
                 ];
             
map_base.generate_manual_rectagle_obstacle(square_points)

% cost_map adjustment
min_distance = 20;              % cell distance
max_distance = 200;
max_value = 1000;

% map_base.reset_attractive_field()
% map_base.cost_map_conversion()
map_base.generate_goal([], [], 5.00, 6.50)              % position in meters
map_base.generate_starting_position([], [], 18, 18)
disp(map_base.goal_position_x)
disp(map_base.goal_position_y)
disp(map_base.starting_position_x)
disp(map_base.starting_position_y)
% map_base.cost_map = map_base.cost_map * 3e3;
% map_base.cost_map(map_base.cost_map>max_value) = max_value;

%% Main cycle for maps generation
while map_number <= n_final_map  
%    map_base.reset_attractive_field()
    map_base.cost_map_conversion()
%     selected_goal_and_sp  = inputdlg(prompt, 'goal and starting position', [1, 65]);
%     map_base.generate_goal([],[], selected_goal_and_sp{1}, selected_goal_and_sp{2})

%     map_base.generate_starting_position()               % starting position is generated randomly
%     map_base.generate_goal(min_distance, max_distance)  % goal position is generated randomly respecting min and max distance
    map_base.cost_map = map_base.cost_map * 3e3;
    map_base.cost_map(map_base.cost_map>max_value) = max_value;

    start_index_x = ceil(map_base.starting_position_x/map_base.resolution_x);
    start_index_y = map_base.N_cells_y - ceil(map_base.starting_position_y/map_base.resolution_y) + 1;
    goal_index_x = ceil(map_base.goal_position_x/map_base.resolution_x);
    goal_index_y = map_base.N_cells_y - ceil(map_base.goal_position_y/map_base.resolution_y) + 1;

    % Cost_map visualization 
    if map_base.goal_position_x && map_base.starting_position_x
        figure(map_number)
            imagesc((1:map_base.N_cells_y)-0.5, (1:map_base.N_cells_x)-0.5, flip(map_base.obstacle_map))
            hold on
            axis square
            plot(start_index_x, start_index_y, 'k.', 'MarkerSize', 20, 'linew', 1.4)
            plot(goal_index_x, goal_index_y, 'rx', 'MarkerSize', 10, 'linew', 1.4)
%             s = surf(X, Y, map.cost_map);
%             s.EdgeColor = 'none';
%             axis([-inf inf -inf inf -inf max_value])
%             axis vis3d
%             daspect([1 1 15])
            view(0, 90)
            xlabel("x")
            ylabel("y")
            zlabel("z")
            colormap winter

        disp(['map N°', num2str(map_number)])
        disp(['starting position located in: ', num2str(map_base.starting_position_x), '  ', num2str(map_base.starting_position_y)])
        disp(['goal located in: ', num2str(map_base.goal_position_x), '  ', num2str(map_base.goal_position_y)])
        m=input('Do you want to save the map? Y/N [Y]:','s');
        if m=='y'   % save map
            map = obj2struct(map_base);
            name = sprintf('%s%s%i%s', folder_name, '\map_', map_number, '.mat');
            save(name, 'map')
            map_number = map_number + 1;
            close      % Reset figure                       
        elseif m=='q'  % break the while cycle
            break                          
        else           % next map
           clf
        end
    end
end
