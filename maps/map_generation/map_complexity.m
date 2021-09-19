clc;
close all;
clear variables;

%%
complexity = [];
for i=1:300
    map_path = sprintf('training_set_4.1/map_%d', i);
    map = load_map(map_path);
    complexity(i) = length(nonzeros(map.obstacle_map))/(200*200);
    disp(i)
end

disp(sum(complexity)/300*100)