import time
import animation
import os
import numpy as np
from os.path import join
from map_class import Map
from drone_class import Drone
from keras.models import load_model
from A_star import main as astar_main
import timeit
from APF_analytical import main as apf_main

def make_action_cycle(drone_list, sim_step):
    for drone_obj in drone_list:
        drone_obj.action2(drone_list, sim_step)

def get_exploration_percentage(drone_list):
    nonzero_elements = np.count_nonzero(drone_list[0].covered_area_matrix)
    total_elements = drone_list[0].covered_area_matrix.size
    percentage = nonzero_elements / total_elements
    return percentage

def main():
    begin_time = time.time()

    # setup for animation
    # max_iterations = 1

    save_animation = False
    video_duration = 250         # [s]
    media_file_name = join(os.getcwd(), '..', 'media', 'pp_presentazione.mp4')

    # RL model
    path_planning_model_number = '22'
    local_minima_path_planning_model_number = '23'          # '12_2b'
    coverage_model_number = '16_2'
    observation_size = 75
    local_minima_observation_size = 75
    coverage_observation_size = 200
    path_planning_model_dir = join(os.getcwd(), '..', 'models', 'actor_' + str(path_planning_model_number))
    local_minima_path_planning_model_dir = join(os.getcwd(), '..', 'models', 'actor_' + str(local_minima_path_planning_model_number))
    coverage_model_dir = join(os.getcwd(), '..', 'models', 'coverage_actor_' + str(coverage_model_number))

    # generate world
    map_number = 260
    # maps_dir = join(os.getcwd(), '..', 'maps', 'training_set_4.1', 'map_' + str(map_number) + '.mat')
    maps_dir = join(os.getcwd(), '..', 'maps', 'validation maps', 'map_large.mat')
    # maps_dir = join(os.getcwd(), 'maps', 'validation maps', 'map_layer_1_cone' + '.mat')
    # maps_dir = join(os.getcwd(), 'maps', 'validation maps', 'map_empty' + '.mat')
    world = Map()
    world.convert_MATLAB_map(maps_dir)
    world.plot_map()

    # set up drone swarm
    n_drones = 1
    drone_selection_index = 0  # chosen drone that will be visualize in the animation
    drone_id = 0
    drone_list = []
    drone_max_speed = 0.12
    drone_safe_distance = 2  # dimension of mobile obstacles for other drones
    fixed_obstacles_safe_distance = 1.5
    minima_osbtacle_dimension = 3.5
    mobile_peak_value, fixed_peak_value, minima_peak_value = 500, 300, 50  # case study
    attractive_constant = 22
    max_steps_apf_descent = 45
    vision_range = 4  # [m] max distance at the drone can detect obstacles
    max_vision_angle = 180  # half cone of vision
    angle_resolution = 250  # number of precision points for vision
    interpolate_trajectory = True
    predict_length = 83    # number of predict steps for the interpolated trajectory
    min_obstacle_distance = 0.3
    max_steps_with_single_goal = 200
    drone_matrix_dimension = 25

    for i in range(0, n_drones):
        drone = Drone()
        drone_id += 1
        drone.set_drone_ID(drone_id)
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
        drone_list.append(drone)
        # print(drone)

    # TEST 1
    drone_list[0].set_initial_position([9, 7])
    # drone_list[0].set_initial_goal([25, 22])
    # drone_list[0].set_initial_position([25, 22])
    drone_list[0].set_initial_goal([35, 34])
    # drone_list[0].set_initial_goal([48, 48])

    # TEST 2
    # drone_list[0].set_initial_position([17, 4])
    # drone_list[0].set_initial_goal([16.4, 18])

    # TEST 3
    # drone_list[0].set_initial_position([17, 4])
    # drone_list[0].set_initial_goal([14.5, 13])

    # TEST 4
    # drone_list[0].set_initial_position([2, 13])
    # drone_list[0].set_initial_goal([14, 9])

    # TEST 5
    # drone_list[0].set_initial_position([2, 18.8])
    # drone_list[0].set_initial_goal([6, 17.2])

    # TEST 6
    # drone_list[0].set_initial_position([10, 10])
    # drone_list[0].set_initial_goal([5, 5])

    #TEST 7
    # drone_list[0].set_initial_position([14, 9])
    # drone_list[0].set_initial_goal([10, 16])

    # TEST 8
    # drone_list[0].set_initial_position([5, 1.5])
    # drone_list[0].set_initial_goal([18, 18])

    # TEST 9
    # drone_list[0].set_initial_position([4, 19])
    # drone_list[0].set_initial_goal([16.8, 5])

    # TEST 10
    # drone_list[0].set_initial_position([16, 18])
    # drone_list[0].set_initial_goal([4.8, 5.5])

    # drone_list[1].set_initial_position([16, 15])
    # drone_list[2].set_initial_position([14, 1])
    # drone_list[3].set_initial_position([1, 17])

    # lidar mode for the first iteration of the simulation
    for drone_obj in drone_list:
        drone_obj.lidar_operation_mode()
        drone_obj.detect_obstacles()
        drone_obj.generate_border_obstacles(world)
    #     drone_obj.share_covered_area(drone_list)
        drone_obj.update_fixed_obstacles(drone_list)
        drone_obj.normal_operation_mode()

    # main cycle of animation
    show_animation = False
    if show_animation:
        animation.animate(drone_list, drone_selection_index, world, media_file_name, video_duration, save_animation, begin_time)
    else:
        simulation_steps = 0
        elapsed_time = round((time.time() - begin_time), 3)
        print('Time elapsed for initialization: ' + str(elapsed_time) + ' s')
        print('SIMULATION IS STARTED')
        drone_list[0].begin_time = time.time()

        while simulation_steps < 1:
            astar_main(world.obstacle_list, world.x_resolution, drone_list[0].position, drone_list[0].goal, 0.5, 'sim' + ' a_star.csv', drone_list[0].obstacle_matrix)
            simulation_steps += 1
            print(simulation_steps)

        # simulation_steps = 0
        # while drone_list[0].goal_flag:
        #     drone_list[0].action(drone_list, simulation_steps)
        #     simulation_steps += 1
        #     print(simulation_steps)
        # np.savetxt(str(sim) + ' RL.csv', np.round(drone_list[0].position_history, 2))

        # simulation_steps = 0
        # while drone_list[0].goal_flag:
        #     drone_list[0].action2(drone_list, simulation_steps)
        #     simulation_steps += 1
        #     print(simulation_steps)
        # np.savetxt(str(sim) + ' apf.csv', np.round(drone_list[0].position_history, 2))

        # simulation_steps = 0
        # while simulation_steps < 1:
        #     apf_main(world,drone_list[0].position, drone_list[0].goal, '[test] apf.csv')
        #     simulation_steps += 1
        #     print(simulation_steps)

if __name__ == "__main__":
    main()
