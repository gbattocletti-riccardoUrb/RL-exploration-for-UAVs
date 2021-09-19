clc;
close all;
clear variables;

%%
map = map_class;

layer = 4;
map_archive_path = sprintf("..\\0. map_archive\\case_study_map\\layer_%i.mat", layer);
% map_path = sprintf("%smap_%i", map_archive_path, layer);
loaded_map = load_map(map_archive_path);
map.dimension_x = loaded_map.dimension_x;
map.dimension_y = loaded_map.dimension_y;
map.resolution_x = loaded_map.resolution_x;
map.resolution_y = loaded_map.resolution_y;
map.N_cells_x = loaded_map.N_cells_x;
map.N_cells_y = loaded_map.N_cells_y;
map.generate_maps();
map.obstacle_map = loaded_map.obstacle_map;

map.generate_matrix();
map.cost_map_conversion()    

min_distance = 60;
max_distance = 90;
map.generate_starting_position(4, 7.7)
map.generate_goal([], [], 18, 5)
map.cost_map = map.cost_map * 3e3;
max_value = 1000;
map.cost_map(map.cost_map>max_value) = max_value;

x = linspace(0, map.dimension_x, map.N_cells_x);
y = linspace(0, map.dimension_y, map.N_cells_y);
[X,Y] = meshgrid(x, y);

if map.goal_position_x && map.starting_position_x
    figure (1)
        hold on
        s = surf(X, Y, map.cost_map);
        s.EdgeColor = 'none';
        plot3(map.goal_position_x, map.goal_position_y, max_value, 'o', 'MarkerFaceColor', [1,0,0], 'MarkerEdgeColor', [0,0,0] )
        plot3(map.starting_position_x, map.starting_position_y, max_value, 'o', 'MarkerFaceColor', [0,0,0], 'MarkerEdgeColor', [0,0,0] )
        axis([-inf inf -inf inf -inf max_value])
        view(0, 90)
        xlabel("x")
        ylabel("y")
        zlabel("z")
        colormap winter
end
map = obj2struct(map);
name = sprintf('%s%s%i%s%s', 'maps', '\map_layer_', 1,'_par_1000', '.mat');
save(name, 'map')




