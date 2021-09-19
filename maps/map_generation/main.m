clc;
close all;
clear variables;

%% Variables initialization
n_final_map = 200;
folder_name = 'maps';
if ~exist(folder_name, 'dir')
   mkdir(folder_name)
end

%% Main cycle for maps generation
iteration = 1;
while iteration <= n_final_map
    
    % Map variables initialization
    map = map_class;
%     map.generate_random_properties('dimension_limits', [40, 100], 'same_resolution', true)
    map.set_dimension(10, 10);
    map.set_resolution(0.1, 0.1);
    map.generate_maps();
    map.generate_border_obstacles();
    map.generate_matrix();
    x = linspace(0, map.dimension_x, map.N_cells_x);
    y = linspace(0, map.dimension_y, map.N_cells_y);
    [X,Y] = meshgrid(x, y);
    % Obstacle generation 
    complexity = 3;
    mean_dimension = mean([map.dimension_x, map.dimension_y]);
    min_dimension = min([map.dimension_x, map.dimension_y]);
    n_obstacles_value = [mean_dimension, mean_dimension, mean_dimension/2, mean_dimension/2, mean_dimension/2];
    n_obstacles = round(n_obstacles_value(complexity));
    max_length = min_dimension/6;

    max_length_x = min_dimension/8;
    max_length_y = min_dimension/8;

    c_obstacle_min_dimension = round(min_dimension/8);
    c_obstacle_max_dimension = round(min_dimension/4);
    
    delta_matrix = [10 0  0  0  0;
                    10 5  0  0  0;
                    0  6  1  0  0;
                    6  5  4  2  0;
                    5  4  4  3  2];
    
    for n = 1:n_obstacles
        
        delta_vector = delta_matrix(complexity, :);
        delta_point = delta_vector(1);
        delta_wall = delta_vector(2);
        delta_rect = delta_vector(3);
        delta_c_close = delta_vector(4);
        delta_c_open = delta_vector(5);
        
        ii = randi(delta_point+delta_wall+delta_rect+delta_c_close+delta_c_open);
        switch true
            case any(ii==1 : delta_point)
                map.generate_point_obstacle();
                
            case any(ii==1+delta_point : delta_point+delta_wall)        
                map.generate_wall_obstacle(max_length);
                
            case any(ii==1+delta_point+delta_wall : delta_point+delta_wall+delta_rect)
                map.generate_rectangular_obstacle(max_length_x, max_length_y);
                
            case any(ii==1+delta_point+delta_wall+delta_rect : delta_point+delta_wall+delta_rect+delta_c_open)
                open = true;
                map.generate_c_obstacle(open, c_obstacle_min_dimension, c_obstacle_max_dimension);
                
            case any(ii==1+delta_point+delta_wall+delta_rect+delta_c_open : delta_point+delta_wall+delta_rect+delta_c_close+delta_c_open)
                open = close;
                map.generate_c_obstacle(open, c_obstacle_min_dimension, c_obstacle_max_dimension);          
        end
    end
       
    % Cost_map adjustment
    map.cost_map_conversion()
    min_distance = 30;
    max_distance = 70;
    map.generate_starting_position()
    map.generate_goal(min_distance, max_distance)
    map.cost_map = map.cost_map * 3e3;
    max_value = 1000;
    map.cost_map(map.cost_map>max_value) = max_value;
    
%   figure
%         hold on
%         imagesc(x, y, map.obstacle_map);
%         axis([x(1), x(end), y(1), y(end)])
%         colormap(flipud(gray))
    start_index_x = ceil(map.starting_position_x/map.resolution_x);
    start_index_y = map.N_cells_y - ceil(map.starting_position_y/map.resolution_y) + 1;
    goal_index_x = ceil(map.goal_position_x/map.resolution_x);
    goal_index_y = map.N_cells_y - ceil(map.goal_position_y/map.resolution_y) + 1;

    % Cost_map visualization 
    if map.goal_position_x & map.starting_position_x    %#ok<AND2>
        figure(iteration)
            imagesc((1:map.N_cells_y)-0.5, (1:map.N_cells_x)-0.5, flip(map.cost_map))
            hold on
            plot(start_index_x, start_index_y, 'k.', 'MarkerSize', 20, 'linew', 1.4)
            plot(goal_index_x, goal_index_y, 'rx', 'MarkerSize', 10, 'linew', 1.4)
%             s = surf(X, Y, map.cost_map);
%             s.EdgeColor = 'none';
%             plot3(map.goal_position_x, map.goal_position_y, max_value, 'o', 'MarkerFaceColor', [1,0,0], 'MarkerEdgeColor', [0,0,0] )
%             plot3(map.starting_position_x, map.starting_position_y, max_value, 'o', 'MarkerFaceColor', [0,0,0], 'MarkerEdgeColor', [0,0,0] )
            axis([-inf inf -inf inf -inf max_value])
%             axis vis3d
%             daspect([1 1 15])
            view(0, 90)
            xlabel("x")
            ylabel("y")
            zlabel("z")
            colormap winter

        disp(['map N°', num2str(iteration)])
        disp(['starting position located in: ', num2str(map.starting_position_x), '  ', num2str(map.starting_position_y)])
        disp(['goal located in: ', num2str(map.goal_position_x), '  ', num2str(map.goal_position_y)])
        m=input('Do you want to save the map? Y/N [Y]:','s');
        if m=='n'
            % Reset figure
            clf
        else
            % Exporting maps
            map = obj2struct(map);
            name = sprintf('%s%s%i%s', folder_name, '\map_', iteration, '.mat');
            save(name, 'map')
            iteration = iteration + 1;
            close
        end
    end
end
