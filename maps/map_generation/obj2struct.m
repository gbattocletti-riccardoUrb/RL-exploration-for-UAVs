function output_struct = obj2struct(obj)
% Converts obj into a struct by examining the public properties of obj. If
% a property contains another object, this function recursively calls
% itself on that object. Else, it copies the property and its value to 
% output_struct. This function treats structs the same as objects.
%
% Note: This function skips over serial, visa and tcpip objects, which
% contain lots of information that is unnecessary (for us).
    string_properties = ['dimension_x', 'dimension_y', 'resolution_x', 'resolution_y', 'N_cells_x', 'N_cells_y', 'cost_map', 'obstacle_map', 'starting_position_x', 'starting_position_y', 'starting_position_index_x', 'starting_position_index_y', 'goal_position_index_x', 'goal_position_index_y', 'goal_position_x', 'goal_position_y'];
    properties = fieldnames(obj); % works on structs & classes (public properties)
    for i = 1:length(properties)
        if contains(string_properties, properties{i,1})
            val = obj.(properties{i});
            if ~isstruct(val) && ~isobject(val)
                output_struct.(properties{i}) = val; 
            else
                if isa(val, 'serial') || isa(val, 'visa') || isa(val, 'tcpip')
                    % don't convert communication objects
                    continue
                end
                temp = obj2struct(val);
                if ~isempty(temp)
                    output_struct.(properties{i}) = temp;
                end
            end
        end
    end
end