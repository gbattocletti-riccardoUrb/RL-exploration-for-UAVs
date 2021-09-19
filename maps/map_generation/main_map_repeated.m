clc;
close all;
clear variables;

%% Variables initialization
n_final_map = 200;
folder_name = 'maps';
prompt = {'Indicate the x position of the goal', 'Indicate the y position of the goal', 'Indicate the x position of the starting position', 'Indicate the y position of the starting position'};
if ~exist(folder_name, 'dir')
   mkdir(folder_name)
end

% Map variables initialization
map_number = 1;
map_base = map_class;
map_base.set_dimension(10, 10);
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

% max_length = min_dimension/8;
% 
% max_length_x = min_dimension/8;
% max_length_y = min_dimension/8;
% 
% c_obstacle_min_dimension = round(min_dimension/10);
% c_obstacle_max_dimension = round(min_dimension/5);
% 
% delta_matrix = [0 5  0  0  0;
%                 0 0  5  0  0;
%                 0  5  2  0  0;
%                 0  6  4  0  0;
%                 0  4  4  3  2];
% 
% for n = 1:n_obstacles
% 
%     delta_vector = delta_matrix(complexity, :);
%     delta_point = delta_vector(1);
%     delta_wall = delta_vector(2);
%     delta_rect = delta_vector(3);
%     delta_c_close = delta_vector(4);
%     delta_c_open = delta_vector(5);
% 
%     ii = randi(delta_point+delta_wall+delta_rect+delta_c_close+delta_c_open);
%     switch true
%         case any(ii==1 : delta_point)
%             map_base.generate_point_obstacle();
% 
%         case any(ii==1+delta_point : delta_point+delta_wall)        
%             map_base.generate_wall_obstacle(max_length);
% 
%         case any(ii==1+delta_point+delta_wall : delta_point+delta_wall+delta_rect)
%             map_base.generate_rectangular_obstacle(max_length_x, max_length_y);
% 
%         case any(ii==1+delta_point+delta_wall+delta_rect : delta_point+delta_wall+delta_rect+delta_c_open)
%             open = true;
%             map_base.generate_c_obstacle(open, c_obstacle_min_dimension, c_obstacle_max_dimension);
% 
%         case any(ii==1+delta_point+delta_wall+delta_rect+delta_c_open : delta_point+delta_wall+delta_rect+delta_c_close+delta_c_open)
%             open = close;
%             map_base.generate_c_obstacle(open, c_obstacle_min_dimension, c_obstacle_max_dimension);          
%     end
% end

% manual obstacles generator
% walls_geometric_dimension = [10, 5, 10, 1              % [center x position, center y position, length, orientation]
%                              10, 6, 10, 1;
%                              14 , 9, 5, 2;
%                              15 , 9, 8, 2;   ];
% map_base.generate_manual_wall_obstacle(walls_geometric_dimension);

% walls_points = [4, 5, 4, 11
%                 5, 6, 5, 10
%                 5, 10, 10, 10
%                 10, 10, 10, 15
%                 10, 15, 15, 15
%                 15, 15, 15, 18
%                 15, 18, 5, 18
%                 5, 18, 5, 15
%                 5, 15, 9, 15
%                 9, 15, 9, 11
%                 9, 11, 4, 11
%                 4, 5, 15, 5
%                 5, 6, 15, 6
%                 15, 5, 15, 1 
%                 15, 1, 18, 1
%                 18, 1, 18, 10
%                 18, 10, 15, 10
%                 15, 10, 15, 6
%                 ];
% 
% map_base.generate_wall_obstacle_from_points(walls_points)
walls_points = [6, 5, 6, 7
                6, 7, 8, 7
                8, 7, 8, 5
                8, 5, 6, 5];
map_base.generate_wall_obstacle_from_points(walls_points)

walls_points = [2, 3, 2, 5
                2, 5, 4, 5
                4, 5, 4, 3
                4, 3, 2, 3];
map_base.generate_wall_obstacle_from_points(walls_points)

% cost_map adjustment
min_distance = 40;              % cell distance
max_distance = 60;
max_value = 1000;

% map_base.reset_attractive_field()
% map_base.cost_map_conversion()
% map_base.generate_goal([], [], 7, 7)              % position in meters
% map_base.generate_starting_position()
% map_base.cost_map = map_base.cost_map * 3e3;
% map_base.cost_map(map_base.cost_map>max_value) = max_value;

%% Main cycle for maps generation
while map_number <= n_final_map  
    map_base.reset_attractive_field()
    map_base.cost_map_conversion()
    selected_goal_and_sp  = inputdlg(prompt, 'goal and starting position', [1, 65]);
    map_base.generate_goal([],[], selected_goal_and_sp{1}, selected_goal_and_sp{2})
    map_base.generate_starting_position([],[], (selected_goal_and_sp{3}), (selected_goal_and_sp{4}))
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
        if m=='n'
            clf                             % Reset figure
        elseif m=='q'
            break                           % break the while cycle
        else
            % Exporting maps
            map = obj2struct(map_base);
            name = sprintf('%s%s%i%s', folder_name, '\map_', map_number, '.mat');
            save(name, 'map')
            map_number = map_number + 1;
            close
        end
    end
end
