"""
A* grid planning
author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)
See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)
Source: https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/AStar/a_star.py
"""
import timeit
import math
import matplotlib.pyplot as plt
import time
import numpy as np

show_animation = False


class AStarPlanner:

    def __init__(self, ox, oy, resolution, robot_radius, obstacle_matrix):
        """
        Initialize grid map for a star planning
        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.resolution = resolution
        self.rr = robot_radius
        self.min_x, self.min_y = 0, 0       # map limits, computed on the base of obstacle position
        self.max_x, self.max_y = 0, 0       # same as above but for the upper limit
        self.obstacle_map = None            # initialized as none, computed in calc_obstacle_map
        self.x_width, self.y_width = 0, 0   # n° of cells on x and y, computed as (max - min) / resolution
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy, obstacle_matrix)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search
        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]
        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),              # computes index of cell given the x/y coordinate: (position - min_pos)/resolution
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while 1:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(open_set, key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o]))
            current = open_set[c_id]

            # show graph

            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                node.x = int(node.x)
                node.y = int(node.y)
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)
        rx = np.array([rx])
        ry = np.array([ry])


        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx = [self.calc_grid_position(goal_node.x, self.min_x)]
        ry = [self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0                                                 # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)            # Euclidean heuristic
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position
        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy, obstacle_matrix):

        self.min_x = int(round(min(ox)))
        self.min_y = int(round(min(oy)))
        self.max_x = int(round(max(ox)))
        self.max_y = int(round(max(oy)))
        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)

        self.x_width = int(round((self.max_x - self.min_x) / self.resolution))
        self.y_width = int(round((self.max_y - self.min_y) / self.resolution))
        print("x_width:", self.x_width)
        print("y_width:", self.y_width)

        # obstacle map generation
        # self.obstacle_map = [[False for _ in range(self.y_width)]
        #                      for _ in range(self.x_width)]
        self.obstacle_map = obstacle_matrix.astype('bool')
        # for ix in range(self.x_width):
        #     print('row ', ix)
        #     x = self.calc_grid_position(ix, self.min_x)              #computed as: pos = index * self.resolution + min_position
        #     for iy in range(self.y_width):
        #         y = self.calc_grid_position(iy, self.min_y)
        #         for iox, ioy in zip(ox, oy):
        #             d = math.hypot(iox - x, ioy - y)
        #             if d <= self.rr:
        #                 self.obstacle_map[ix][iy] = True
        #                 break
        print(time.time())

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion


def main(obstacle_list, resolution, starting_position, goal_position, robot_radius, name_file, obstacle_matrix):
    print(__file__ + " start!!")

    # start and goal position
    sx = int(starting_position[0])  # [m]
    sy = int(starting_position[1]) # [m]
    gx = int(goal_position[0])  # [m]
    gy = int(goal_position[1])  # [m]
    grid_size = resolution  # [m]
    robot_radius = robot_radius

    # set obstacle positions
    ox = list(obstacle_list[:, 0])
    # ox = obstacle_list[:, 0].astype(int)
    oy = list(obstacle_list[:, 1])
    # oy = obstacle_list[:, 1].astype(int)

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    a_star = AStarPlanner(ox, oy, grid_size, robot_radius, obstacle_matrix)
    starttime = timeit.default_timer()

    rx, ry = a_star.planning(sx, sy, gx, gy)
    # print(timeit.repeat(stmt=a_star.planning, setup=import_module))
    print("The time elapsed is :", timeit.default_timer() - starttime)
    # elapsed_time = time.time() - begin_time
    # print('Elapsed time: ', elapsed_time, ' [s]')
    position_history = np.hstack([rx.T, ry.T])
    print(len(position_history))
    np.savetxt(name_file, np.round(position_history, 2))

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        plt.show()


# if __name__ == '__main__':
#     main()