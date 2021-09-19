classdef map_class < handle
        
    properties (Constant)
        max_iteration = 5
        security_coefficient = 1
        wall_security_coefficient = 3         % must be greater than security_coefficient
        steepness = 4;
        attractive_constant_cone = 9e-3;
        attractive_constant_parable = 1e-3;
        point_thickness = 5;
    end
    
    properties
		% independent properties
		dimension_x
		dimension_y
		resolution_x
		resolution_y
		N_cells_x
		N_cells_y
		% map matrices
		obstacle_map
		cost_map
        c_matrix
        gaussian_matrix
        iteration
        x_cohordinate_matrix
        y_cohordinate_matrix
        starting_position_x
        starting_position_y
        starting_position_index_x
        starting_position_index_y
        goal_position_x
        goal_position_y
        goal_position_index_x
        goal_position_index_y
        attractive_field
        security_interval
    end

    methods
        
        function generate_random_properties(self, varargin)
            % named optional arguments are given. Options available are:
            %   - dimension_limits = [min, max] --> min and max of the
            %                                       map side length [m]
            %   - same_resolution = BOOL --> true if resolution along x
            %								 and y is the same
            % for details about management of name/value pairs see the link below:
            % https://stackoverflow.com/questions/2775263/how-to-deal-with-name-value-pairs-of-function-arguments-in-matlab/60178631#60178631
            options = struct('dimension_limits', [0, 100], 'same_resolution', false);   % default arguments
            option_names = fieldnames(options);
            n_args = length(varargin);          % count arguments
            if round(n_args/2)~=n_args/2        % check that name/value pairs are correctly given
               error('generate_random_properties needs propertyName/propertyValue pairs')
            end
            for pair = reshape(varargin,2,[])   % pair is {propName;propValue}
                input_name = lower(pair{1});     % make case insensitive
                if any(strcmp(input_name, option_names))    % string comparison between option and argument names
                    options.(input_name) = pair{2};
                else
                    error('%s is not a recognized parameter name of generate_random_properties',input_name)
                end
            end
            self.dimension_x = randi(options.dimension_limits);
            self.dimension_y = randi(options.dimension_limits);
            resolution_values = [0.05, 0.1, 0.25, 0.5];
            selector = randi(length(resolution_values));
            self.resolution_x = resolution_values(selector);
            if options.same_resolution
                self.resolution_y = self.resolution_x; 
            else
                self.resolution_y  = resolution_values(randi(length(resolution_values)));
            end
        end
        
        function set_resolution(self, res_x, res_y)
            self.resolution_x = res_x;
            self.resolution_y = res_y;
        end
        
        function set_dimension(self, dim_x, dim_y)
            self.dimension_x = dim_x;
            self.dimension_y = dim_y;
        end
        
        function generate_maps(self)
           self.N_cells_x = round(self.dimension_x/self.resolution_x);
           self.N_cells_y = round(self.dimension_y/self.resolution_y);
           self.security_interval = round(mean([self.N_cells_x, self.N_cells_y])/50);
           self.attractive_field = zeros(self.N_cells_y, self.N_cells_x);
           self.cost_map = zeros(self.N_cells_y, self.N_cells_x);
           self.obstacle_map = self.cost_map;
           x = linspace(0, self.dimension_x, self.N_cells_x);
           y = linspace(0, self.dimension_y, self.N_cells_y);
           [self.x_cohordinate_matrix, self.y_cohordinate_matrix] = meshgrid(x, y);
%            p = polyfit([50, 100, 200, 400], [2e-3, 1e-3, 7e-4, 3e-4], 3);                                   % created with polyfit
%            self.attractive_constant = polyval(p, min([self.dimension_x, self.dimension_y]));
        end
                  
        function generate_matrix(self)
            n = 21;                                         % must be odd
            normal_prob_distribution = makedist('Normal');          
            x = linspace(-n/self.steepness, n/self.steepness, n);
            y = linspace(-n/self.steepness, n/self.steepness, n);
            x_distribution = pdf(normal_prob_distribution, x);
            y_distribution = pdf(normal_prob_distribution, y);
            self.gaussian_matrix = (y_distribution'*x_distribution);
        end
        
        function generate_c_matrix(self, open, min_dimension, max_dimension)
            dimension = randi([min_dimension,  max_dimension]);
            dimension = round(dimension / max([self.resolution_x, self.resolution_y]));
            self.c_matrix = zeros(dimension);
            orientation = randi([1, 4]);
            if open
                self.c_matrix(3, 3:end-2) = 1;
                self.c_matrix(end-2, 3:end-2) = 1;
                self.c_matrix(3:end-2, 3) = 1;
            else
                tick = round(dimension/5);
                self.c_matrix(3, 3:end-2) = 1;
                self.c_matrix(end-2, 3:end-2) = 1;
                self.c_matrix(3:end-2, 3) = 1;
                self.c_matrix(3:3+tick, end-2) = 1;             
                self.c_matrix(end-2-tick:end-2, end-2) = 1;                
            end
            switch orientation
                case 1
                    self.c_matrix = self.c_matrix;
                case 2
                    self.c_matrix = self.c_matrix';
                case 3
                    self.c_matrix = flip(self.c_matrix,2);
                case 4
                    self.c_matrix = flip(self.c_matrix');
            end  
        end
              
        function generate_border_obstacles(self)
        	self.obstacle_map(1, :) = 1;
        	self.obstacle_map(:, 1) = 1;
        	self.obstacle_map(end, :) = 1;
        	self.obstacle_map(:, end) = 1;
        end
       
        function generate_point_obstacle(self)
        	location_found = false;
            cost_map_flag = false;
            point_flag = true;
            self.iteration = 1;
            interval_x = round(self.N_cells_x/50);
            interval_y = round(self.N_cells_y/50);
        	while ~location_found
        		index_position_x = randi([1, self.N_cells_x]);
        		index_position_y = randi([1, self.N_cells_y]);
                [index_x, index_y] = self.check_proximity(index_position_x, index_position_y, interval_x, interval_y);
                if ~any(self.obstacle_map(index_y-interval_y:index_y+interval_y, index_x-interval_x:index_x+interval_x), "all")
                    self.matrix_placer(index_position_y, index_position_x, self.gaussian_matrix, cost_map_flag, point_flag)
        			self.obstacle_map(index_y, index_x) = 1;
        			location_found = true;
        		else
                    self.iteration = self.iteration + 1;
                end
                if self.iteration > self.max_iteration
                    break
                end
        	end
        end
        
        function generate_wall_obstacle(self, max_length)
            location_found = false;
            semi_length = randi(round(max_length/2));
            semi_length_x = round(semi_length/self.resolution_x);
            semi_length_y = round(semi_length/self.resolution_y);
            self.iteration = 1;
            while ~location_found
                orientation = randi(2,1);
                index_position_x = randi([1, self.N_cells_x]);
                index_position_y = randi([1, self.N_cells_y]);
                [index_x, index_y] = self.check_proximity(index_position_x, index_position_y, semi_length_x, semi_length_y);
                if ~any(self.obstacle_map(index_y-semi_length_y:index_y+semi_length_y, index_x-semi_length_x:index_x+semi_length_x), "all")
                    if orientation == 1
                        self.obstacle_map(index_y, index_x-semi_length_x:index_x+semi_length_x) = 1;
                    elseif orientation == 2
                        self.obstacle_map(index_y-semi_length_y:index_y+semi_length_y, index_x) = 1;
                    end
        			location_found = true;
                else
                    self.iteration = self.iteration + 1;
                end
                if self.iteration > self.max_iteration
                    break
                end
            end
        end
        
        function generate_manual_wall_obstacle(self, geometric_dimension)
            for ii = 1:size(geometric_dimension, 1)
                orientation = geometric_dimension(ii, 4);
                length = geometric_dimension(ii, 3);
                semi_length = length/2;
                semi_length_x = round(semi_length/self.resolution_x);
                semi_length_y = round(semi_length/self.resolution_y);
                position_x = geometric_dimension(ii, 1);
                position_y = geometric_dimension(ii, 2); 
                index_x = ceil(position_x / self.resolution_x);
                index_y = ceil(position_y / self.resolution_y);
                if orientation == 1
                    self.obstacle_map(index_y, index_x-semi_length_x:index_x+semi_length_x) = 1;
                elseif orientation == 2
                    self.obstacle_map(index_y-semi_length_y:index_y+semi_length_y, index_x) = 1;
                end
            end
        end
        
        function generate_wall_obstacle_from_points(self, points)
            for ii = 1:size(points, 1)
                flag_x = false;
                flag_y = false;
                point_x = round(points(ii, 1) / self.resolution_x);
                point_y = round(points(ii, 2) / self.resolution_y);
                while ~flag_x || ~flag_y
                    self.obstacle_map(point_y, point_x) = 1;
                    if point_x ~= round(points(ii, 3) / self.resolution_x)
                        if points(ii, 1) < points(ii, 3)
                            point_x = point_x + 1;
                        else
                            point_x = point_x - 1;
                        end
                    else
                        flag_x = true;
                    end
                    if point_y ~= round(points(ii, 4) / self.resolution_y)
                        if points(ii, 2) < points(ii, 4)
                            point_y = point_y + 1;
                        else
                            point_y = point_y - 1;
                        end
                    else
                        flag_y = true;
                    end
                end
                    
 
            end
        end
        
        function generate_rectangular_obstacle(self, max_length_x, max_length_y)
            location_found = false;
            semi_length_x = randi(round(max_length_x/2));
            semi_length_y = randi(round(max_length_y/2));    
            semi_length_x = round(semi_length_x/self.resolution_x);
            semi_length_y = round(semi_length_y/self.resolution_y);
            self.iteration = 1;
            while ~location_found
                index_position_x = randi([1, self.N_cells_x]);
                index_position_y = randi([1, self.N_cells_y]);
                [index_x, index_y] = self.check_proximity(index_position_x, index_position_y, semi_length_x, semi_length_y);
                if ~any(self.obstacle_map(index_y-semi_length_y:index_y+semi_length_y, index_x-semi_length_x:index_x+semi_length_x), "all")
                    self.obstacle_map(index_y-semi_length_y : index_y+semi_length_y, index_x-semi_length_x:index_x+semi_length_x) = 1;
        			location_found = true;
                else
                    self.iteration = self.iteration + 1;
                end
                if self.iteration > self.max_iteration
                    break
                end
            end
        end
        
        function generate_manual_rectagle_obstacle(self, geometric_dimension)
            for ii = 1:size(geometric_dimension, 1)
                length_x = geometric_dimension(ii, 3);
                length_y = geometric_dimension(ii, 4);
                semi_length_x = round(length_x/2 /self.resolution_x);
                semi_length_y = round(length_y/2 /self.resolution_y);
                position_x = geometric_dimension(ii, 1);
                position_y = geometric_dimension(ii, 2); 
                index_x = ceil(position_x / self.resolution_x);
                index_y = ceil(position_y / self.resolution_y);
                self.obstacle_map(index_y-semi_length_y+1:index_y+semi_length_y, index_x-semi_length_x+1:index_x+semi_length_x) = 1;
            end
        end
        
        function generate_c_obstacle(self, open, min_dimension, max_dimension)
            self.generate_c_matrix(open, min_dimension, max_dimension);
            matrix = self.c_matrix;
            half_dimension = length(matrix) / 2;
            half_dimension = floor(half_dimension/2)*2; 
            location_found = false; 
            cost_map_flag = false;
            point_flag = false;
            self.iteration = 1;
            while ~location_found
                index_position_x = randi([1, self.N_cells_x]);
                index_position_y = randi([1, self.N_cells_y]);
                [index_position_x, index_position_y] = self.check_proximity(index_position_x, index_position_y, half_dimension, half_dimension);
                half_dimension = round(half_dimension * self.security_coefficient);
                if ~any(self.obstacle_map(index_position_y-half_dimension:index_position_y+half_dimension, index_position_x-half_dimension:index_position_x+half_dimension), "all")
                    self.matrix_placer(index_position_y, index_position_x, matrix, cost_map_flag, point_flag)
        			location_found = true;
                else
                    self.iteration = self.iteration + 1;
                end
                if self.iteration > self.max_iteration
                    break
                end
            end
        end
        
        function generate_goal(self, varargin)
            n_args = length(varargin);          % count arguments
        	location_found = false;
            self.iteration = 1;
            
            if n_args == 0 
                while ~location_found
                    index_position_x = randi([1, self.N_cells_x]);
                    index_position_y = randi([1, self.N_cells_y]);
                    [index_x, index_y] = self.check_proximity(index_position_x, index_position_y, self.security_interval, self.security_interval);
                    if ~any(self.obstacle_map(index_y-self.security_interval : index_y+self.security_interval, index_x-self.security_interval : index_x+self.security_interval), "all")
                        self.goal_position_index_x = index_x;
                        self.goal_position_index_y = index_y;
                        self.goal_position_x = round(self.goal_position_index_x * self.resolution_x, 2);
                        self.goal_position_y = round(self.goal_position_index_y * self.resolution_y, 2);
                        self.generation_attractive_field();
                        location_found = true;
                    else
                        self.iteration = self.iteration + 1;
                    end
                    if self.iteration > self.max_iteration
                        break
                    end
                end
            elseif n_args == 2
                min_distance = varargin{1};
                max_distance = varargin{2};
                while ~location_found
                    index_position_x = randi([1, self.N_cells_x]);
                    index_position_y = randi([1, self.N_cells_y]);
                    [index_x, index_y] = self.check_proximity(index_position_x, index_position_y, self.security_interval, self.security_interval);
                    delta_x = index_x - ceil(self.starting_position_x / self.resolution_x);
                    delta_y = index_y - ceil(self.starting_position_y / self.resolution_y);
                    if ~any(self.obstacle_map(index_y-self.security_interval : index_y+self.security_interval, index_x-self.security_interval : index_x+self.security_interval), "all") & hypot(delta_x, delta_y) < max_distance & hypot(delta_x, delta_y) > min_distance  %#ok<AND2>
                        self.goal_position_index_x = index_x;
                        self.goal_position_index_y = index_y;
                        self.goal_position_x = round(self.goal_position_index_x * self.resolution_x, 2);
                        self.goal_position_y = round(self.goal_position_index_y * self.resolution_y, 2);
                        self.generation_attractive_field();
                        location_found = true;
                    else
                        self.iteration = self.iteration + 1;
                    end
                    if self.iteration > self.max_iteration
                        break
                    end
                end
            elseif n_args == 4 && isempty(varargin{3}) && isempty(varargin{4})
                while ~location_found
                    index_position_x = randi([1, self.N_cells_x]);
                    index_position_y = randi([1, self.N_cells_y]);
                    [index_x, index_y] = self.check_proximity(index_position_x, index_position_y, self.security_interval, self.security_interval);
                    if ~any(self.obstacle_map(index_y-self.security_interval : index_y+self.security_interval, index_x-self.security_interval : index_x+self.security_interval), "all")
                        self.goal_position_index_x = index_x;
                        self.goal_position_index_y = index_y;
                        self.goal_position_x = round(self.goal_position_index_x * self.resolution_x, 2);
                        self.goal_position_y = round(self.goal_position_index_y * self.resolution_y, 2);
                        self.generation_attractive_field();
                        location_found = true;
                    else
                        self.iteration = self.iteration + 1;
                    end
                    if self.iteration > self.max_iteration
                        break
                    end
                end
            elseif n_args == 4
%                 self.goal_position_x = round(str2double(varargin{3}), 2);
%                 self.goal_position_y = round(str2double(varargin{4}), 2); 
                self.goal_position_x = round(varargin{3}, 2);
                self.goal_position_y = round(varargin{4}, 2); 
                self.goal_position_index_x = self.goal_position_x / self.resolution_x;
                self.goal_position_index_y = self.goal_position_y / self.resolution_y;
                self.generation_attractive_field();
            else
                error('missing arguments: (1, 2) -> distance from starting position;  (3, 4) -> selected position ')
            end
        end
        
        function generate_starting_position(self, varargin)
        	location_found = false;
            self.iteration = 1;
            
            n_args = length(varargin);          % count arguments
            if n_args == 0
                while ~location_found
                    index_position_x = randi([1, self.N_cells_x]);
                    index_position_y = randi([1, self.N_cells_y]);
                    [index_x, index_y] = self.check_proximity(index_position_x, index_position_y, self.security_interval, self.security_interval);
                    if ~any(self.obstacle_map(index_y-self.security_interval : index_y+self.security_interval, index_x-self.security_interval : index_x+self.security_interval), "all")
                        self.starting_position_index_x = index_x;
                        self.starting_position_index_y = index_y;
                        self.starting_position_x = round(self.starting_position_index_x * self.resolution_x, 2);
                        self.starting_position_y = round(self.starting_position_index_y * self.resolution_y, 2); 
                        location_found = true;
                    else
                        self.iteration = self.iteration + 1;
                    end
                    if self.iteration > self.max_iteration
                        break
                    end
                end
            elseif n_args == 2
                min_distance = varargin{1};
                max_distance = varargin{2};
                while ~location_found
                    index_position_x = randi([1, self.N_cells_x]);
                    index_position_y = randi([1, self.N_cells_y]);
                    [index_x, index_y] = self.check_proximity(index_position_x, index_position_y, self.security_interval, self.security_interval);
                    delta_x = index_x - ceil(self.starting_position_x / self.resolution_x);
                    delta_y = index_y - ceil(self.starting_position_y / self.resolution_y);
                    if ~any(self.obstacle_map(index_y-self.security_interval : index_y+self.security_interval, index_x-self.security_interval : index_x+self.security_interval), "all") & hypot(delta_x, delta_y) < max_distance & hypot(delta_x, delta_y) > min_distance
                        self.starting_position_index_x = index_x;
                        self.starting_position_index_y = index_y;
                        self.starting_position_x = round(self.starting_position_index_x * self.resolution_x, 2);
                        self.starting_position_y = round(self.starting_position_index_y * self.resolution_y, 2); 
                        location_found = true;
                    else
                        self.iteration = self.iteration + 1;
                    end
                    if self.iteration > self.max_iteration
                        break
                    end
                end
            elseif n_args == 4 && isempty(varargin{3}) && isempty(varargin{4})
                while ~location_found
                    index_position_x = randi([1, self.N_cells_x]);
                    index_position_y = randi([1, self.N_cells_y]);
                    [index_x, index_y] = self.check_proximity(index_position_x, index_position_y, self.security_interval, self.security_interval);
                    if ~any(self.obstacle_map(index_y-self.security_interval : index_y+self.security_interval, index_x-self.security_interval : index_x+self.security_interval), "all")
                        self.starting_position_index_x = index_x;
                        self.starting_position_index_y = index_y;
                        self.starting_position_x = round(self.starting_position_index_x * self.resolution_x, 2);
                        self.starting_position_y = round(self.starting_position_index_y * self.resolution_y, 2); 
                        location_found = true;
                    else
                        self.iteration = self.iteration + 1;
                    end
                    if self.iteration > self.max_iteration
                        break
                    end
                end
                
            elseif n_args == 4
%                 self.starting_position_x = round(str2double(varargin{3}), 2);
%                 self.starting_position_y = round(str2double(varargin{4}), 2); 
                self.starting_position_x = round(varargin{3}, 2);
                self.starting_position_y = round(varargin{4}, 2); 
                self.starting_position_index_x = self.starting_position_x / self.resolution_x;
                self.starting_position_index_y = self.starting_position_y / self.resolution_y;
            else
                error('The position must be in two element -> x, y')
            end
        end
     
        function generation_attractive_field(self)
            for ii = 1 : self.N_cells_y
                for jj = 1 : self.N_cells_x
%                     distance = hypot(ii - self.goal_position_y / self.resolution_y, jj - self.goal_position_x / self.resolution_x);
%                     if distance >= 10
                    self.attractive_field(ii, jj) = (hypot(self.y_cohordinate_matrix(ii, jj) - self.goal_position_y, self.x_cohordinate_matrix(ii, jj) - self.goal_position_x));
%                     self.cost_map(ii, jj) = self.cost_map(ii, jj) + (self.attractive_constant_cone * self.attractive_field(ii, jj));
%                     else
%                     self.cost_map(ii, jj) = self.cost_map(ii, jj) + (self.attractive_constant_parable * (hypot(self.y_cohordinate_matrix(ii, jj) - self.goal_position_y, self.x_cohordinate_matrix(ii, jj) - self.goal_position_x)).^2);
%                     end 
                end
            end
            self.cost_map = self.cost_map + self.attractive_constant_cone * self.attractive_field;
        end
        
        function reset_attractive_field(self)
            self.cost_map = self.cost_map - (self.attractive_constant_cone * self.attractive_field);
            self.cost_map = zeros(self.N_cells_y, self.N_cells_x);
        end
            
        function cost_map_conversion(self)
            cost_map_flag = true;
            point_flag = false;
            for index_position_y = 1:self.N_cells_y
                for index_position_x = 1:self.N_cells_x
                    if self.obstacle_map(index_position_y, index_position_x)
                        self.matrix_placer(index_position_y, index_position_x, self.gaussian_matrix, cost_map_flag, point_flag)    
                    end
                end
            end
        end
        
        function [index_position_x, index_position_y] = check_proximity(self, index_position_x, index_position_y, semi_interval_x, semi_interval_y) 
            semi_interval_x = round(semi_interval_x * self.wall_security_coefficient);            % in order to guarantee a good distance from wall
            semi_interval_y = round(semi_interval_y * self.wall_security_coefficient);
            if index_position_x + semi_interval_x > self.N_cells_x
                index_position_x = self.N_cells_x - semi_interval_x;
            elseif index_position_x - semi_interval_x < 1
                index_position_x = 1 + semi_interval_x;
            end
                
            if index_position_y + semi_interval_y > self.N_cells_y
                index_position_y = self.N_cells_y - semi_interval_y;
            elseif index_position_y - semi_interval_y < 1
                index_position_y = 1 + semi_interval_y;
            end
        end
        
        function matrix_placer(self, index_position_y, index_position_x, matrix, cost_map_flag, point_flag)
            n = length(matrix);
            delta_x = (n - 1) / 2;
            delta_y = (n - 1) / 2;
            
            min_index_x = 1;
            max_index_x = self.N_cells_x;
            min_index_y = 1;
            max_index_y = self.N_cells_y;
            
            min_x = round(max(index_position_x - delta_x, min_index_x));
            max_x = round(min(index_position_x + delta_x, max_index_x));
            min_y = round(max(index_position_y - delta_y, min_index_y));
            max_y = round(min(index_position_y + delta_y, max_index_y));
                    
            min_x_A = round(delta_x - (index_position_x - min_x) + 1);
            max_x_A = round(delta_x + (max_x - index_position_x) + 1);
            min_y_A = round(delta_y - (index_position_y - min_y) + 1);
            max_y_A = round(delta_y + (max_y - index_position_y) + 1);
            
            if cost_map_flag
                self.cost_map(min_y:max_y, min_x:max_x) = self.cost_map(min_y:max_y, min_x:max_x) + matrix(min_y_A:max_y_A, min_x_A:max_x_A);
            elseif point_flag
                self.cost_map(min_y:max_y, min_x:max_x) = self.cost_map(min_y:max_y, min_x:max_x) + matrix(min_y_A:max_y_A, min_x_A:max_x_A)*self.point_thickness;
            elseif ~cost_map_flag
                self.obstacle_map(min_y:max_y, min_x:max_x) = self.obstacle_map(min_y:max_y, min_x:max_x) + matrix;
            end
        end
        
        
    end 
end