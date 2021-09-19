n_obstacles = zeros(300, 1);
for n_map = 1:300
    path = sprintf('training_set_3/map_%i.mat', n_map);
    map = load_map(path);
    a = 0;
    for ii = 1:400
        for jj = 1:400
            if map.obstacle_map(ii, jj) == 1
                a = a + 1;
            end
        end
    end
    n_obstacles(n_map) = a;
end

figure(1)
plot(n_obstacles)