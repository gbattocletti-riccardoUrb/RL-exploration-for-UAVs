import time
import animation
import os
import numpy as np
from os.path import join
from map_class import Map
from drone_class import Drone
from keras.models import load_model


def make_action_cycle(drone_list, world):
    for drone_obj in drone_list:
        drone_obj.action(drone_list, world)

def get_exploration_percentage(drone_list):
    nonzero_elements = np.count_nonzero(drone_list[0].covered_area_matrix)
    total_elements = drone_list[0].covered_area_matrix.size
    percentage = nonzero_elements / total_elements
    return percentage

def main():
    begin_time = time.time()

    # setup for animation
    max_iterations = 100
    show_animation = True
    save_animation = False
    video_duration = 600         # [s]
    media_file_name = join(os.getcwd(), '..', 'media', 'animation_coverage.mp4')

    # RL model
    path_planning_model_number = '22'
    local_minima_path_planning_model_number = '12_2b'      #'12_2b'
    coverage_model_number = '21_4'                      # 11_5
    observation_size = 75
    local_minima_observation_size = 25
    coverage_observation_size = 200
    path_planning_model_dir = join(os.getcwd(), '..', 'models', 'actor_' + str(path_planning_model_number))
    local_minima_path_planning_model_dir = join(os.getcwd(), '..', 'models', 'actor_' + str(local_minima_path_planning_model_number))
    coverage_model_dir = join(os.getcwd(), '..', 'models', 'coverage_actor_' + str(coverage_model_number))

    # generate world
    map_number = 'B'
    # maps_dir = join(os.getcwd(), 'maps', 'training_set_4.1', 'map_' + str(map_number) + '.mat')
    maps_dir = join(os.getcwd(), '..', 'maps', 'validation maps', 'map_' + str(map_number) + '.mat')
    # maps_dir = join(os.getcwd(), 'maps', 'validation maps', 'map_layer_1_cone' + '.mat')
    # maps_dir = join(os.getcwd(), 'maps', 'validation maps', 'map_empty' + '.mat')
    world = Map()
    world.convert_MATLAB_map(maps_dir)
    world.plot_map()

    # set up drone swarm
    n_drones = 4
    drone_selection_index = 0  # chosen drone that will be visualize in the animation
    drone_id = 0
    drone_list = []
    drone_max_speed = 0.12
    drone_safe_distance = 2.5  # dimension of mobile obstacles for other drones
    fixed_obstacles_safe_distance = 1.2
    minima_osbtacle_dimension = 3.5
    mobile_peak_value, fixed_peak_value, minima_peak_value = 500, 300, 50  # case study
    attractive_constant = 35
    max_steps_apf_descent = 50
    vision_range = 2  # [m] max distance at the drone can detect obstacles
    max_vision_angle = 180  # half cone of vision
    angle_resolution = 200  # number of precision points for vision
    predict_length = 7     # number of predict steps for the interpolated trajectory
    interpolate_trajectory = True
    min_obstacle_distance = 0.3
    max_steps_with_single_goal = 200
    drone_matrix_dimension = 75         #85

    for i in range(0, n_drones):
        drone = Drone()
        drone.set_drone_ID(drone_id)
        drone_id += 1
        drone.import_map_properties(world)
        drone.set_drone_safe_distance(drone_safe_distance)
        drone.set_fixed_obstacles_safe_distance(fixed_obstacles_safe_distance)
        drone.set_minima_obstacles_dimension(minima_osbtacle_dimension)
        drone.set_matrix_peak_value(mobile_peak_value, fixed_peak_value, minima_peak_value)
        drone.set_attractive_constant(attractive_constant)
        drone.set_predict_length(predict_length)
        drone.max_steps_apf_descent_path_planning(max_steps_apf_descent)
        drone.set_max_speed(drone_max_speed)
        drone.min_obstacles_distance(min_obstacle_distance)
        drone.set_interpolation_flag(interpolate_trajectory)
        drone.set_vision_settings(vision_range, max_vision_angle, angle_resolution)
        drone.set_RL_path_planning_model(load_model(path_planning_model_dir, compile=False))
        drone.set_local_minima_path_planning_model(load_model(local_minima_path_planning_model_dir, compile=False))
        drone.set_RL_coverage_model(load_model(coverage_model_dir, compile=False))
        drone.set_max_steps_with_single_goal(max_steps_with_single_goal)
        drone.set_observation_size(observation_size)
        drone.set_local_minima_observation_size(local_minima_observation_size)
        drone.set_coverage_observation_size(coverage_observation_size)
        drone.set_drone_influence_matrix(drone_matrix_dimension)
        # drone.set_random_initial_position(world)
        drone.initialize(n_drones)
        # drone.coverage_random()
        drone.update_attractive_layer()
        drone_list.append(drone)

    # TEST 1
    # drone_list[0].set_initial_position([8.95, 5.12])
    drone_list[0].set_initial_position([18.95, 4.5])
    drone_list[1].set_initial_position([1.5, 18.5])
    drone_list[2].set_initial_position([18.5, 18.5])
    drone_list[3].set_initial_position([2, 1])


    # Test B
    drone_list[2].set_initial_goal([15, 15])
    # drone_list[3].set_initial_position([5, 10])
    # drone_list[0].set_initial_goal([16.1, 16.17])

    # Test C
    # drone_list[3].set_initial_position([7, 5])
    # drone_list[1].set_initial_goal([5, 11])

    # TEST 2
    # drone_list[0].set_initial_position([17, 5])
    # drone_list[1].set_initial_position([1.5, 18.5])
    # drone_list[2].set_initial_position([18.5, 18.5])
    # drone_list[3].set_initial_position([1.5, 1.5])

    # TEST 3
    # drone_list[0].set_initial_position([17, 2])
    # drone_list[1].set_initial_position([1.5, 18.5])
    # drone_list[2].set_initial_position([18.5, 18.5])
    # drone_list[3].set_initial_position([2, 5])

    for drone in drone_list:
        print(drone)

    # drone_list[1].set_initial_position([16, 15])
    # drone_list[2].set_initial_position([14, 1])
    # drone_list[3].set_initial_position([1, 17])

    # lidar mode for the first iteration of the simulation
    for drone_obj in drone_list:
        drone_obj.lidar_operation_mode()
        drone_obj.detect_obstacles()
        drone_obj.generate_border_obstacles()
        drone_obj.share_covered_area(drone_list)
        drone_obj.update_fixed_obstacles(drone_list)
        drone_obj.normal_operation_mode()

    # main cycle of animation
    if show_animation:
        animation.animate(drone_list, drone_selection_index, world, media_file_name, video_duration, save_animation, begin_time)
    else:
        simulation_steps = 0
        elapsed_time = round((time.time() - begin_time), 3)
        print('Time elapsed for initialization: ' + str(elapsed_time) + ' s')
        print('SIMULATION IS STARTED')
        while simulation_steps <= max_iterations:
            make_action_cycle(drone_list, world)
            exploration_percentage = get_exploration_percentage(drone_list)
            print("Explored map percentage = {:.2%} | Step N° {}".format(exploration_percentage, simulation_steps))
            simulation_steps += 1
        animation.plot_figures(drone_list)

if __name__ == "__main__":
    main()
