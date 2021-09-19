function loaded_map = load_map(map_path)
        
    load(map_path, 'map');
    loaded_map = map;	% passes out the "map" variable --> NOTE: it is a STRUCTURE

end