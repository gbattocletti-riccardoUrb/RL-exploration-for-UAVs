import numpy as np
import time

def main(Map, starting_position, goal, name):
    # Map.obstacle_map = Map.obstacle_map.T
    rx = []
    ry = []
    iteration = 0
    delta = 2  # observation area size
    size = 200  # map size
    w1 = 0.5
    w2 = 10
    x = starting_position[0]       # starting position x
    y = starting_position[1]       # starting position y
    xi = round(x * 10)  # starting position index x
    yi = round(y * 10)  # starting position index y
    rx.append(x)
    ry.append(y)
    x_g = goal[0]
    y_g = goal[1]
    goal_dist = 0.2 # distance at which goal is considered reached
    begin_time = time.time()
    while True:
        vx = 0.0  # direction vector
        vy = 0.0
        for ii in range(-delta, delta):
            x_obs = x + ii
            if 0 < x_obs <= 20:
                for jj in range(-delta, delta):
                    y_obs = y + jj
                    if 0 < y_obs <= 20:
                        if Map.obstacle_map[int(x_obs*10), int(y_obs*10)] == 1:
                            vx = vx + w1 * 1/(x - x_obs)
                            vy = vy + w1 * 1/(y - y_obs)
        vx = vx + (w2 * (x_g - x))**2
        vy = vy + (w2 * (y_g - y))**2
        vx = vx/np.hypot(vx, vy) * 0.1
        vy = vy/np.hypot(vx, vy) * 0.1
        x = x + vx
        y = y + vy
        xi = round(x * 10)
        yi = round(y * 10)
        rx.append(x)
        ry.append(y)
        print(iteration)
        iteration += 1
        if iteration == 200:
            rx = np.array([rx])
            ry = np.array([ry])
            position_history = np.hstack([rx.T, ry.T])
            np.savetxt(name, np.round(position_history, 2))
        if np.hypot((x-x_g), (y-y_g)) <= goal_dist:
            print('goal found')
            end_time = time.time()-begin_time
            print(end_time)
            rx = np.array([rx])
            ry = np.array([ry])
            position_history = np.hstack([rx.T, ry.T])
            np.savetxt(name, np.round(position_history, 2))
            break