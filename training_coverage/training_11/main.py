# Coverage training 11

import argparse
import os
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from environment import Environment
from noise_class import *
from buffer_class import *
from agent_functions import *
from drone_class import Drone

# from os.path import join
import tensorflow as tf
import wandb
import time

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], False)
#wandb.login(key = '')
#wandb.init(name='', project="",  entity='')
#config = wandb.config

# set object to collect all parameters and hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)                    # gamma
parser.add_argument('--actor_lr', type=float, default=0.001)                # actor learning rate
parser.add_argument('--critic_lr', type=float, default=0.001)               # critic learning rate
parser.add_argument('--buffer_capacity', type=int, default=10000)           # experience memory buffer size
parser.add_argument('--batch_size', type=int, default=16)                   # minibatch size
parser.add_argument('--tau', type=float, default=0.005)                     # tau (smoothed target update)
parser.add_argument('--state_size', type=int, default=1)                    # image size (input is NxN)
parser.add_argument('--lower_bound', type=float, default=0)                 # action lower bound
parser.add_argument('--upper_bound', type=float, default=1)                 # action upper bound
parser.add_argument('--std_dev', type=float, default=0.2)                   # exploration noise std
parser.add_argument('--episode_number', type=int, default=10000)            # number of training episodes
parser.add_argument('--episode_steps', type=int, default=150)               # number of steps per episode
parser.add_argument('--show_figure', type=bool, default=False)              # show figure each "save_figure_period" episodes
parser.add_argument('--save_figure', type=bool, default=False)              # save figure each "save_figure_period" episodes
parser.add_argument('--save_figure_period', type=int, default=500)          # how many episodes between
parser.add_argument('--model_number', type=int, default=1)                  # number of model (for model saving)
parser.add_argument('--average_window', type=int, default=10)               # n° of episodes on which to compute the reward average
parser.add_argument('--coverage_observation_size', type=int, default=100)    # n° of episodes on which to compute the reward average
parser.add_argument('--speed', type=float, default=0.1)                     # n° of episodes on which to compute the reward average
parser.add_argument('--min_output', type=float, default=0.0)                # min position output
parser.add_argument('--max_output_x', type=float, default=19.99)               # max position output
parser.add_argument('--max_output_y', type=float, default=19.99)               # max position output
parser.add_argument('--substeps', type=int, default=10)
parser.add_argument('--drone_matrix_dimension', type=int, default=25)       # dimension of the area of influence of the drone to past in observation - must be odd
args = parser.parse_args()

# set hyperparameters value
args.gamma = 0.95
args.tau = 0.005
args.std_dev = 0.3
args.critic_lr = 0.001
args.actor_lr = 0.0001
args.save_figure = True
args.show_figure = False
args.save_figure_period = 500
args.model_number = '11'
args.episode_number = 500001
args.episode_steps = 5
args.substeps = 30
args.coverage_observation_size = 200
args.speed = 0.12
args.min_output = 0
args.max_output_x = 19.99
args.max_output_y = 19.99
args.drone_matrix_dimension = 55

# save hyperparameters in wandb
# config.gamma = args.gamma
# config.buffer_capacity = args.buffer_capacity
# config.batch_size = args.batch_size
# config.lower_bound = args.lower_bound
# config.upper_bound = args.upper_bound
# config.std_dev = args.std_dev
# config.tau = args.tau
# config.critic_lr = args.critic_lr
# config.actor_lr = args.actor_lr
# config.episode_number = args.episode_number
# config.episode_steps = args.episode_steps
# config.observation_size = args.coverage_observation_size
# config.speed = args.speed
# config.average_window = args.average_window
# config.min_action_position = args.min_output
# config.max_action_position_x = args.max_output_x
# config.max_action_position_y = args.max_output_y
# config.drone_matrix_dimension = args.drone_matrix_dimension

# create folder and folder paths
map_path = os.path.join('maps', 'training_set_6')
if not os.path.exists('episode_images'):
    os.makedirs('episode_images')
if not os.path.exists('models'):
    os.makedirs('models')
training_path = os.path.join('episode_images', 'training_' + str(args.model_number))
if not os.path.exists(training_path):
    os.makedirs(training_path)
actor_name = os.path.join('models','coverage_actor_' + str(args.model_number))
critic_name = os.path.join('models', 'coverage_critic_' + str(args.model_number))

# create environment object
env = Environment()
env.set_map_folder(map_path)
env.map_counter = 0
env.set_map_max_number(False, 1)
env.set_max_step_number(args.episode_steps)
env.set_coverage_observation_size(args.coverage_observation_size)
env.set_drone_influence_matrix(args.drone_matrix_dimension)

args.state_size = env.observation_space.shape[0]
args.upper_bound = 1
args.lower_bound = 0
args.num_actions = env.action_space.shape[0]
print("Size of State Space ->  {}".format(args.state_size))
print("Size of Action Space ->  {}".format(args.num_actions))
print("Max Value of goal position x ->  {}".format(args.max_output_x))
print("Max Value of goal position y ->  {}".format(args.max_output_y))

# create agent nets
actor_model = create_actor(args)
critic_model = create_critic(args)
target_actor = create_actor(args)
target_critic = create_critic(args)

# initialize weights (initialized with same values for net and target couples)
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# initialize buffer
buffer = Buffer(args)
ep_mean_reward_list = []                                 # To store reward history of each episode
avg_reward_list = []                                        # To store average reward history of last few episodes
# reward_data_matrix = np.empty([args.episode_steps, args.episode_number])
# reward_data_matrix[:] = np.nan

# inizialize drones
# set up drone swarm
n_drones = 3
drone_id = 0
drone_list = []
drone_max_speed = 0.12
drone_safe_distance = 2.5  # dimension of mobile obstacles for other drones
fixed_obstacles_safe_distance = 1.3
mobile_peak_value, fixed_peak_value = 500, 200  # case study
attractive_constant = 40
vision_range = 1.3  # [m] max distance at the drone can detect obstacles
max_vision_angle = 180  # half cone of vision
angle_resolution = 180  # number of precision points for vision
drone_observation_size = 25

first_map_path = os.path.join(map_path, 'map_1.mat')    # load first map to allow initialization of drones
env.load_MATLAB_map(first_map_path)

for i in range(0, n_drones):
    drone = Drone()
    drone.set_drone_ID(drone_id)
    drone_id += 1
    drone.set_drone_safe_distance(drone_safe_distance)
    drone.set_fixed_obstacles_safe_distance(fixed_obstacles_safe_distance)
    drone.set_matrix_peak_value(mobile_peak_value, fixed_peak_value)
    drone.set_attractive_constant(attractive_constant)
    drone.set_max_speed(drone_max_speed)
    drone.set_vision_settings(vision_range, max_vision_angle, angle_resolution)
    drone.set_observation_size(drone_observation_size)
    # drone.set_RL_path_planning_model(load_model(path_planning_model_dir, compile=False))
    drone.import_map_properties(env)
    drone.initialize(n_drones)
    drone_list.append(drone)
    print(drone)

# create noise process
ou_noises = []
for drone in drone_list:
    ou_noise_i = OUActionNoise(np.zeros(1), args.std_dev * np.ones(1))
    ou_noises.append(ou_noise_i)

# TRAINING
total_steps = np.zeros(n_drones, dtype=int)

for ep in range(1, args.episode_number):
    state = env.reset(drone_list)   # compute initial state
    action = np.empty([n_drones, 2])
    goal_position = np.empty(2)
    done_episode = np.zeros(n_drones, dtype=bool)       # reset episode termination condition
    done_step = np.zeros(n_drones, dtype=bool)          # reset step end condition
    episode_reward = np.zeros([n_drones], dtype=float)                 # reset episode reward for each drone
    step = 0                                            # reset step counter
    drones_step = np.zeros(n_drones, dtype=int)
    cumulative_explored_cells = np.zeros([n_drones])
        # https: // www.youtube.com / watch?v = IuBQkdsajVE
    while True:
        for drone in drone_list:
            if not done_episode[drone.drone_ID]:
                done_step[drone.drone_ID] = False                                                    # reset step termination condition
                drones_step[drone.drone_ID] += 1
                state[drone.drone_ID] = env.coverage_observe(drone_list, drone.drone_ID)
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(state[drone.drone_ID]), 0)
                drone_position = (np.array(drone.position)) / [args.max_output_x, args.max_output_y]
                drone_position = tf.expand_dims(tf.convert_to_tensor(drone_position), 0)
                action[drone.drone_ID, :] = policy(actor_model, tf_prev_state, ou_noises[drone.drone_ID], drone_position)[0]    # action = [x, y]
                goal_position[0] = action[drone.drone_ID, 0] * args.max_output_x
                goal_position[1] = action[drone.drone_ID, 1] * args.max_output_y
                drone.set_goal(goal_position)
                for substep in range(args.substeps):
                    if not done_step[drone.drone_ID]:
                        drone.action(drone_list)
                        cumulative_explored_cells[drone.drone_ID] += drone.explored_cell_number
                        done_step[drone.drone_ID] = env.isdone_step(drone)
                if not done_episode[drone.drone_ID]:
                    reward = env.compute_reward(drone, cumulative_explored_cells)
                    episode_reward[drone.drone_ID] += reward
                    next_state = env.coverage_observe(drone_list, drone.drone_ID)  # perform observation at the end of the step
                    drone_position = np.array(drone.position) / [args.max_output_x, args.max_output_y]
                    buffer.record((state[drone.drone_ID], action[drone.drone_ID, :], reward, next_state, drone_position))  # add sample to experience buffer
                    state[drone.drone_ID] = next_state
                    if done_step[drone.drone_ID]:
                        if env.obstacle_map[drone.index_position[0]][drone.index_position[1]] == 1:  # check for collisions - update done_episode
                            done_episode[drone.drone_ID] = True

        # backpropagation & gradient descent
        buffer.learn(actor_model, critic_model, target_actor, target_critic, args)
        update_target(target_actor.variables, actor_model.variables, args.tau)
        update_target(target_critic.variables, critic_model.variables, args.tau)
        buffer.learn(actor_model, critic_model, target_actor, target_critic, args)
        update_target(target_actor.variables, actor_model.variables, args.tau)
        update_target(target_critic.variables, critic_model.variables, args.tau)

        # increase step counter
        step += 1

        # check for episode termination conditions
        if done_episode.all():  # end this episode when all elements of `done_episode` are True
            break
        elif step >= args.episode_steps:
            break
        else:
            continue
        # each drone that has not collided with obstacles (i.e. done_episode == False) does its substeps

        # for drone in drone_list:
        #     for substep in range(args.substeps):
        #         if not done_step[drone.drone_ID]:
        #             drone.action(drone_list)
        #             cumulative_explored_cells[drone.drone_ID] += drone.explored_cell_number
        #             done_step[drone.drone_ID] = env.isdone_step(drone)
        # for drone in drone_list:
        #     if not done_episode[drone.drone_ID]:
        #         reward = env.compute_reward(drone, cumulative_explored_cells)
        #         episode_reward[drone.drone_ID] += reward
        #         next_state = env.coverage_observe(drone_list, drone.drone_ID)                       # perform observation at the end of the step
        #         drone_position = np.array(drone.position) / [args.max_output_x, args.max_output_y]
        #         buffer.record((state[drone.drone_ID], action[drone.drone_ID, :], reward, next_state, drone_position))                  # add sample to experience buffer
        #         state[drone.drone_ID] = next_state
        #         if done_step[drone.drone_ID]:
        #             if env.obstacle_map[drone.index_position[0]][drone.index_position[1]] == 1:     # check for collisions - update done_episode
        #                 done_episode[drone.drone_ID] = True
        # backpropagation & gradient descent - finally!
        # buffer.learn(actor_model, critic_model, target_actor, target_critic, args)
        # update_target(target_actor.variables, actor_model.variables, args.tau)
        # update_target(target_critic.variables, critic_model.variables, args.tau)
        # reward_data_matrix[step, ep] = reward # da sistemare --> 4 matrici?
        # # increase step counter
        # step += 1
        #
        # # check for episode termination conditions
        # if done_episode.all():                                                                  # end this episode when all elements of `done_episode` are True
        #     break
        # elif step >= args.episode_steps:
        #     break
        # else:
        #     continue
    # end episode
    total_steps += drones_step
    ep_mean_reward_list.append(np.mean(episode_reward))
    avg_reward = np.mean(ep_mean_reward_list[-args.average_window:])                  # the average reward is the mean of the last 10 episodes
    avg_reward_list.append(avg_reward)
    print("Episode {} | Episode Reward = {} | Average Reward = {} | Episode steps = {} | Total steps = {} | Map N° {}".format(ep, np.round(episode_reward, 2), np.round(avg_reward, 2), drones_step, total_steps, env.map_counter))
    # wandb.log({'Average Reward': avg_reward, 'Reward drone 0': episode_reward[0], 'Reward drone 1': episode_reward[1], 'Reward drone 2': episode_reward[2], 'Reward drone 3': episode_reward[3],
    #            'Episode Steps drone 0': drones_step[0], 'Episode Steps drone 1': drones_step[1], 'Episode Steps drone 2': drones_step[2], 'Episode Steps drone 3': drones_step[3]})
    #wandb.log({'Average Reward': avg_reward, 'Reward drone 0': episode_reward[0], 'Reward drone 1': episode_reward[1], 'Reward drone 2': episode_reward[2],
    #           'Episode Steps drone 0': drones_step[0], 'Episode Steps drone 1': drones_step[1], 'Episode Steps drone 2': drones_step[2]})
    if ep % args.save_figure_period == 0:
        figure_name = os.path.join('episode_images', 'training_' + str(args.model_number), 'episode_' + str(ep))
        env.visualize(ep, args.show_figure, args.save_figure, figure_name, drone_list, avg_reward)
        actor_model.save(actor_name + '_ep_' + str(ep))
        critic_model.save(critic_name + '_ep_' + str(ep))
        # np.savetxt("prova.csv", reward_data_matrix, fmt='%0.4f', delimiter=",")

# save models
# np.savetxt("prova.csv", reward_data_matrix, fmt='%0.4f', delimiter=",")
actor_model.save(actor_name + '_final')
critic_model.save(critic_name + 'final')
print(actor_model.summary(0))
print(critic_model.summary(0))
