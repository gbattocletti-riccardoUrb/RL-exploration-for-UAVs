import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import main
import re
import pandas as pd
import time
# import pickle
from matplotlib.colors import ListedColormap

def plot_figures(drone_list):
    print('END OF SIMULATION')
    plt.close('all')
    single_area_covered = {}
    covered_percentage = np.zeros([1, len(drone_list[0].single_coverage[2: -1])])
    for drone in drone_list:
        single_area_covered['Drone ' + str(drone.drone_ID)] = drone.single_coverage[2: -1]
        covered_percentage += np.array(drone.single_coverage[2: -1])

    covered_percentage = covered_percentage[0]
    plt.rc('font', family='serif')
    dpi = 150
    figure_size_pixel = 1500, 1000
    figure_size = figure_size_pixel[0] / dpi, figure_size_pixel[1] / dpi
    final_figure = plt.figure(3, dpi=dpi, figsize=figure_size)
    ax = final_figure.add_subplot(121, autoscale_on=False, ylim=[0, 100], xlim=[0, len(covered_percentage)])
    ax.set_xlabel('step number', labelpad=8)
    ax.set_ylabel('% of' + '\n' + 'covered area')
    ax.plot(np.linspace(0, len(covered_percentage)-1, len(covered_percentage)), covered_percentage)

    ax_2 = final_figure.add_subplot(122, autoscale_on=False, ylim=[0, 100], xlim=[0, len(covered_percentage)])
    ax_2.set_xlabel('step number', labelpad=8)
    ax_2.set_ylabel('covered' + '\n' + 'area')
    df = pd.DataFrame(single_area_covered)
    df.plot.area(ax=ax_2, linewidth=0, colormap="viridis")

    plt.tight_layout()
    plt.show()
    exit(0)

def final_configuration(drone_list):
    single_area_covered = {}
    covered_percentage = np.zeros([1, len(drone_list[0].single_coverage[2: -1])])
    for drone in drone_list:
        single_area_covered['Drone ' + str(drone.drone_ID)] = drone.single_coverage[2: -1]
        covered_percentage += np.array(drone.single_coverage[2: -1])

    covered_percentage = covered_percentage[0]
    plt.rc('font', family='serif')
    dpi = 150
    figure_size_pixel = 1500, 1000
    figure_size = figure_size_pixel[0] / dpi, figure_size_pixel[1] / dpi
    final_figure = plt.figure(3, dpi=dpi, figsize=figure_size)
    ax = final_figure.add_subplot(121, autoscale_on=False, ylim=[0, 100], xlim=[0, len(covered_percentage)])
    ax.set_xlabel('step number', labelpad=8)
    ax.set_ylabel('% of' + '\n' + 'covered area')
    ax.plot(np.linspace(0, len(covered_percentage)-1, len(covered_percentage)), covered_percentage)

    ax_2 = final_figure.add_subplot(122, autoscale_on=False, ylim=[0, 100], xlim=[0, len(covered_percentage)])
    ax_2.set_xlabel('step number', labelpad=8)
    ax_2.set_ylabel('covered' + '\n' + 'area')
    df = pd.DataFrame(single_area_covered)
    df.plot.area(ax=ax_2, linewidth=0, colormap="viridis")

    plt.tight_layout()
    plt.show()
    exit(0)

def animate(drone_list, drone_selection_index, world, media_file_name, duration, save_animation, begin_time):

    def init_plot(map_object):
        # set up figure and animation
        # plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        dpi = 250
        figure_size_pixel = 1500, 500
        figure_size = figure_size_pixel[0] / dpi, figure_size_pixel[1] / dpi
        fig = plt.figure(2)
        fig.dpi = dpi
        fig.size = figure_size
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=[0, map_object.x_dimension], ylim=[0, map_object.y_dimension])
        ax.set_yticks([])
        ax.set_xticks([])
        # ax.set_xlabel('x [m]', labelpad=8)
        # ax.set_ylabel('y [m]', labelpad=20)
        ms = int(fig.dpi * drone_list[drone_selection_index].radius * fig.get_figwidth() / np.diff(ax.get_xbound())[0]) / 8  # dimension of points in the plot
        plt.tight_layout()

        # plot of the colorbar
        # p_map = heatmap.draw_heatmap(potential_map, map_object)
        # c_bar = fig.colorbar(p_map, orientation="horizontal", pad=0.15)
        # c_bar.set_label("Potential field intensity", labelpad=-50)

        # setting for the outer box of the plot
        # box = plt.Rectangle((-0.05, -0.05), map_object.x_dimension+0.2, map_object.y_dimension+0.2, ec='none', lw=2, fc='none')
        # ax.add_patch(box)
        # box.set_edgecolor('k')

        # setting of the plotted objects
        drones, = ax.plot([], [], 'bo', ms=6, zorder=10)
        objectives, = ax.plot([], [], 'bx', ms=6, zorder=15)
        selected_drone, = ax.plot([], [], 'ro', ms=6, zorder=20)
        selected_objective, = ax.plot([], [], 'rx', ms=6, zorder=20)
        obstacles, = ax.plot([], [], 'ko', ms=6, zorder=5)
        full_obstacles, = ax.plot([], [], 's', color='grey', ms=5, zorder=4, alpha=0.1)
        # trajectory, = ax.plot([], [], 'r-', linewidth=0.8, zorder=15)
        trajectory, = ax.plot([], [], 'r-', linewidth=0.8, zorder=15)
        local_minima, = ax.plot([], [], 'o',color='orange', ms=8, fillstyle='none', zorder=10)

        drone_0, = ax.plot([], [], 'ro', ms=6, zorder=21)
        drone_1, = ax.plot([], [], 'go', ms=6, zorder=21)
        drone_2, = ax.plot([], [], 'bo', ms=6, zorder=21)
        drone_3, = ax.plot([], [], 'mo', ms=6, zorder=21)
        # trajectory_0, = ax.plot([], [], 'r-', linewidth=0.8, zorder=16)
        trajectory_0, = ax.plot([], [], 'r-', linewidth=0.8, zorder=16)
        trajectory_1, = ax.plot([], [], 'g-', linewidth=0.8, zorder=16)
        trajectory_2, = ax.plot([], [], 'b-', linewidth=0.8, zorder=16)
        trajectory_3, = ax.plot([], [], 'm-', linewidth=0.8, zorder=16)
        objective_0, = ax.plot([], [], 'rx', ms=6, zorder=16)
        objective_1, = ax.plot([], [], 'gx', ms=6, zorder=16)
        objective_2, = ax.plot([], [], 'bx', ms=6, zorder=16)
        objective_3, = ax.plot([], [], 'mx', ms=6, zorder=16)

        # texts
        text_step_counter = ax.text(0.05, 0.95, '', fontsize=10, transform=ax.transAxes, bbox={'facecolor':'w', 'alpha':1, 'pad':5})

        return fig, ax, ms, drones, objectives, selected_drone, selected_objective, obstacles, full_obstacles, trajectory, local_minima, text_step_counter, \
                drone_0, drone_1, drone_2, drone_3, trajectory_0, trajectory_1, trajectory_2, trajectory_3, \
                objective_0, objective_1, objective_2, objective_3,

    def init_plot_coverage(map_object):
        # set up figure and animation
        ax = fig.add_subplot(122, aspect='equal', autoscale_on=False, xlim=[0, map_object.x_dimension], ylim=[0, map_object.y_dimension])
        ax.set_yticks([])
        ax.set_xticks([])
        # ax.set_xlabel('x [m]', labelpad=8)
        # ax.set_ylabel('y [m]', rotation=0, labelpad=20)
        ms = int(fig.dpi * drone_list[drone_selection_index].radius * fig.get_figwidth() / np.diff(ax.get_xbound())[0]) / 8  # dimension of points in the plot
        plt.tight_layout()

        # setting of the plotted objects
        drones_coverage, = ax.plot([], [], 'bo', ms=6, zorder=10)
        obstacles_coverage, = ax.plot([], [], 'ko', ms=6, zorder=5)
        trajectory_coverage, = ax.plot([], [], 'r:', ms=6, zorder=15)

        k_means_center, = ax.plot([], [], 'rx', ms=6, zorder=16)

        # texts
        coverage_percentage = ax.text(0.05, 0.95, '', fontsize=6, transform=ax.transAxes, bbox={'facecolor':'w', 'alpha':0.5, 'pad':5})

        return ax, ms, drones_coverage, obstacles_coverage, trajectory_coverage, coverage_percentage, k_means_center,

    def draw_heatmap(heatmap_data, map_obj, ax, v_max, c_map):
        # heatmap_data_mod = np.flip(np.array(heatmap_data), 1)
        # plt_color_map = plt.imshow(heatmap_data_mod.T, cmap=cmap, vmax=1000, zorder=0, extent=[0, map_obj.x_dimension-0.08, 0, map_obj.y_dimension-0.08])
        x_axis = np.round(np.linspace(0, map_obj.x_dimension, int(map_obj.x_dimension / map_obj.x_resolution)+1, endpoint=True), 2)
        y_axis = np.round(np.linspace(0, map_obj.y_dimension, int(map_obj.y_dimension / map_obj.y_resolution)+1, endpoint=True), 2)
        plt_color_map = ax.pcolormesh(x_axis, y_axis, heatmap_data.T, cmap=c_map, vmax=v_max, vmin=0, shading='auto', zorder=0)
        return plt_color_map

    def hex_to_rgb_color_list(colors):
        """
        Take color or list of hex code colors and convert them
        to RGB colors in the range [0,1].

        Parameters:
            - colors: Color or list of color strings of the format
                      '#FFF' or '#FFFFFF'

        Returns:
            The color or list of colors in RGB representation.
        """
        if isinstance(colors, str):
            colors = [colors]

        for i, color in enumerate([color.replace('#', '') for color in colors]):
            hex_length = len(color)

            if hex_length not in [3, 6]:
                raise ValueError('Colors must be of the form #FFFFFF or #FFF')

            regex = '.' * (hex_length // 3)
            colors[i] = [int(val * (6 // hex_length), 16) / 255for val in re.findall(regex, color)]

        return colors[0] if len(colors) == 1 else colors

    def blended_cmap(rgb_color_list, N):
        """
        Created a colormap blending from one color to the other.

        Parameters:
            - rgb_color_list: A list of colors represented as [R, G, B]
              values in the range [0, 1], like [[0, 0, 0], [1, 1, 1]],
              for black and white, respectively.

        Returns:
            A matplotlib `ListedColormap` object
        """
        if not isinstance(rgb_color_list, list):
            raise ValueError('Colors must be passed as a list.')
        elif len(rgb_color_list) < 2:
            raise ValueError('Must specify at least 2 colors.')
        elif (not isinstance(rgb_color_list[0], list) or not isinstance(rgb_color_list[1], list)) or (len(rgb_color_list[0]) != 3 or len(rgb_color_list[1]) != 3):
            raise ValueError('Each color should be represented as a list of size 3.')

        N, entries = N, 4  # red, green, blue, alpha
        rgbas = np.ones((N, entries))

        segment_count = len(rgb_color_list) - 1
        segment_size = N // segment_count
        remainder = N % segment_count  # need to add this back later

        for i in range(entries - 1):  # we don't alter alphas
            updates = []
            for seg in range(1, segment_count + 1):
                # determine how much needs to be added back to account for remainders
                offset = 0 if not remainder or seg > 1 else remainder
                updates.append(np.linspace(start=rgb_color_list[seg - 1][i], stop=rgb_color_list[seg][i], num=segment_size + offset))
            rgbas[:, i] = np.concatenate(updates)

        return ListedColormap(rgbas)

    def get_all_drones_positions(drone_list):
        # gets the position of all drones. Used for the plot (see main function)
        position_list = np.empty([0, 2])
        for drone in drone_list:
            position_list = np.vstack([position_list, drone.position])
        return position_list

    def get_all_goal_positions(drone_list):
        # gets the position of all drones goal. Used for the plot (see main function)
        goal_position_list = np.empty([0, 2])
        for drone in drone_list:
            goal_position_list = np.vstack([goal_position_list, drone.goal])
        return goal_position_list

    def get_obstacles_list(drone_list):
        obstacle_list = drone_list[0].obstacle_list
        return obstacle_list

    def get_full_obstacles_list():
        full_obstacle_list = world.obstacle_list
        return full_obstacle_list

    def get_exploration_percentage(drone_list):
        nonzero_elements = np.count_nonzero(drone_list[0].covered_area_matrix)
        total_elements = drone_list[0].covered_area_matrix.size
        percentage = nonzero_elements / total_elements * 100
        covered_perc.append(percentage)
        return percentage

    def init_animation():
        """ initialize animation """
        drones.set_data([], [])
        objectives.set_data([], [])
        obstacles.set_data([], [])
        full_obstacles.set_data([], [])
        trajectory.set_data([], [])
        local_minima.set_data([], [])
        text_step_counter.set_text('')
        # drones_coverage.set_data([], [])
        # obstacles_coverage.set_data([], [])
        # coverage_percentage.set_text('')
        drone_0.set_data([], [])
        drone_1.set_data([], [])
        drone_2.set_data([], [])
        drone_3.set_data([], [])
        trajectory_0.set_data([], [])
        trajectory_1.set_data([], [])
        trajectory_2.set_data([], [])
        trajectory_3.set_data([], [])
        objective_0.set_data([], [])
        objective_1.set_data([], [])
        objective_2.set_data([], [])
        objective_3.set_data([], [])
        # k_means_center.set_data([], [])

        return drones, objectives, obstacles, full_obstacles, local_minima, trajectory, text_step_counter, drone_0,\
               drone_1, drone_2, drone_3, trajectory_0, trajectory_1, trajectory_2, trajectory_3, \
               objective_0, objective_1, objective_2, objective_3

    def main_animation(iteration):
        """ Main cicle of animation"""
        main.make_action_cycle(drone_list, iteration)
        print(iteration)
        all_positions = get_all_drones_positions(drone_list)
        all_goal_positions = get_all_goal_positions(drone_list)
        obstacles_discovered = get_obstacles_list(drone_list)
        full_obstacle_list = get_full_obstacles_list()
        exploration_percentage = get_exploration_percentage(drone_list)
        local_minima_list, _ = drone_list[drone_selection_index].detect_local_minima()

        # plot of the heatmaps
        pf_map = drone_list[drone_selection_index].potential_field_map
        max_value = 1000
        plt_color_map = draw_heatmap(pf_map, world, ax, max_value, None)

        # covered_map = drone_list[drone_selection_index].covered_area_matrix
        # max_value = 1
        # covered_area = draw_heatmap(covered_map, world, ax_2, max_value, c_map_coverage)

        # update pieces of the animation
        selected_drone.set_data(all_positions[drone_selection_index, x], all_positions[drone_selection_index, y])
        selected_drone.set_markersize(ms*2)

        drones.set_data(all_positions[:, x], all_positions[:, y])
        drones.set_markersize(ms*2)

        selected_objective.set_data(all_goal_positions[drone_selection_index, x], all_goal_positions[drone_selection_index, y])
        selected_objective.set_markersize(ms*2)

        objectives.set_data(all_goal_positions[:, x], all_goal_positions[:, y])
        objectives.set_markersize(ms*2)

        obstacles.set_data(obstacles_discovered[:, x], obstacles_discovered[:, y])
        obstacles.set_markersize(ms)

        # full_obstacles.set_data(full_obstacle_list[:, x], full_obstacle_list[:, y])
        # full_obstacles.set_markersize(ms)

        # drones_coverage.set_data(all_positions[:, x], all_positions[:, y])
        # drones_coverage.set_markersize(ms)

        # obstacles_coverage.set_data(obstacles_discovered[:, x], obstacles_discovered[:, y])
        # obstacles_coverage.set_markersize(ms/2)

        trajectory.set_data(drone_list[drone_selection_index].smoothed_trajectory_points[x][:], drone_list[drone_selection_index].smoothed_trajectory_points[y][:])
        trajectory_0.set_data(drone_list[0].smoothed_trajectory_points[x][:],
                            drone_list[0].smoothed_trajectory_points[y][:])
        # trajectory_1.set_data(drone_list[1].smoothed_trajectory_points[x][:],
        #                     drone_list[1].smoothed_trajectory_points[y][:])
        # trajectory_2.set_data(drone_list[2].smoothed_trajectory_points[x][:],
        #                     drone_list[2].smoothed_trajectory_points[y][:])
        # trajectory_3.set_data(drone_list[3].smoothed_trajectory_points[x][:],
        #                     drone_list[3].smoothed_trajectory_points[y][:])

        drone_0.set_data(all_positions[0, x], all_positions[0, y])
        drone_0.set_markersize(ms)
        # drone_1.set_data(all_positions[1, x], all_positions[1, y])
        # drone_1.set_markersize(ms)
        # drone_2.set_data(all_positions[2, x], all_positions[2, y])
        # drone_2.set_markersize(ms)
        # drone_3.set_data(all_positions[3, x], all_positions[3, y])
        # drone_3.set_markersize(ms)

        objective_0.set_data(all_goal_positions[0, x], all_goal_positions[0, y])
        objective_0.set_markersize(ms*1.5)
        # objective_1.set_data(all_goal_positions[1, x], all_goal_positions[1, y])
        # objective_1.set_markersize(ms*1.5)
        # objective_2.set_data(all_goal_positions[2, x], all_goal_positions[2, y])
        # objective_2.set_markersize(ms*1.5)
        # objective_3.set_data(all_goal_positions[3, x], all_goal_positions[3, y])
        # objective_3.set_markersize(ms*1.5)
        # local_minima.set_data(local_minima_list[:, x], local_minima_list[:, y])
        # local_minima.set_markersize(ms*5)
        # k_means_center.set_data(drone_list[0].kmeans_center[:, 0], drone_list[0].kmeans_center[:, 1])
        # k_means_center.set_markersize(ms*1.5)
        # text
        text_step_counter.set_text("Step: {}".format(iteration))
        # coverage_percentage.set_text(u" Covered area = {:.2f} %".format(exploration_percentage))
        # save_frequency = 2
        # if iteration%save_frequency == 0:
        #     plt.savefig('../simulation_images/PP_sim' + str(int(iteration/save_frequency)) + '.png', dpi=250)

        # for stopping simulation or return the coverage plot
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [plot_figures(drone_list) if event.key == 'enter' else None])
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])

        return drones, objectives, selected_drone, selected_objective, obstacles, full_obstacles, local_minima,\
               text_step_counter, trajectory, \
               drone_0, drone_1, drone_2, drone_3, trajectory_0, trajectory_1, trajectory_2, trajectory_3, \
               objective_0, objective_1, objective_2, objective_3, plt_color_map,

    # initial settings
    x = 0
    y = 1
    plt.rcParams['animation.ffmpeg_path'] = r'C:\Program Files\ffmpeg_2020\bin\ffmpeg'
    FFwriter = anim.FFMpegWriter(fps=5, extra_args=['-vcodec', 'libx264'])
    coverage_colors = ['#FFFFFF','#35A4F3']
    rgbs_coverage = hex_to_rgb_color_list(coverage_colors)
    c_map_coverage = blended_cmap(rgbs_coverage, 256)
    covered_perc = []

    (fig, ax, ms, drones, objectives, selected_drone, selected_objective, obstacles, full_obstacles, trajectory, local_minima, text_step_counter,
     drone_0, drone_1, drone_2, drone_3,
     trajectory_0, trajectory_1, trajectory_2, trajectory_3,
     objective_0, objective_1, objective_2, objective_3) = init_plot(world)

    # (ax_2, ms, drones_coverage, obstacles_coverage, trajectory_coverage, coverage_percentage, k_means_center) = init_plot_coverage(world)

    # elapsed time for initialization
    elapsed_time = round((time.time() - begin_time), 3)
    print('Time elapsed for initialization: ' + str(elapsed_time) + ' s')
    print('SIMULATION IS STARTED')

    animation = anim.FuncAnimation(fig, main_animation, init_func=init_animation, frames=(duration * 5), interval=10, blit=True)

    if save_animation:
        animation.save(media_file_name, writer=FFwriter, dpi=250)
        plt.show()
    else:
        plt.show()
    # plot_figures(drone_list)
    return

# trajectory importation
# f = open('store_data.pckl', 'rb')
# pos_his = pickle.load(f)
# f.close()
# trajectory.set_data(pos_his[1:, x], pos_his[1:, y])

# Saving the trajectory of one drone
# f = open('store_data.pckl', 'wb')
# pickle.dump(drone_list[0].position_history, f)
# f.close()