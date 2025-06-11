import matplotlib.pyplot as plt
import numpy as np
import random

width, height = 100, 100
num_obstacles = 4
buffer = 10
step_size = 2
goal_radius = 5
num_iterations = 1000
obstacle_area_fraction = 0.05
total_area = width * height
obstacle_area = total_area * obstacle_area_fraction
obstacle_size = int(np.sqrt(obstacle_area))
obstacles = []

for _ in range(num_obstacles):
    while True:
        x = random.randint(0, width - obstacle_size)
        y = random.randint(0, height - obstacle_size)
        new_obstacle = (x, y, obstacle_size, obstacle_size)
        if all(not (x <= obs_x <= x + obstacle_size or x <= obs_x + obs_w <= x + obstacle_size) or
               not (y <= obs_y <= y + obstacle_size or y <= obs_y + obs_h <= y + obstacle_size)
               for obs_x, obs_y, obs_w, obs_h in obstacles):
            obstacles.append(new_obstacle)
            break


def is_in_obstacle_with_buffer(point, buffer=10):
    px, py = point
    for ox, oy, ow, oh in obstacles:
        if (ox - buffer <= px <= ox + ow + buffer) and (oy - buffer <= py <= oy + oh + buffer):
            return True
    return False


def find_free_point(buffer=0):
    while True:
        point = (random.randint(0, width), random.randint(0, height))
        if not is_in_obstacle_with_buffer(point, buffer):
            print(f"DEBUG: returning point: {point}")
            return point


# Node class definition
class Node:
    def __init__(self, point, parent=None):
        self.point = point
        self.parent = parent


def distance(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])


def nearest_node(nodes, random_point):
    return min(nodes, key=lambda node: distance(node.point, random_point))


def steer(from_node, to_point, step_size=5):
    if distance(from_node.point, to_point) < step_size:
        return to_point
    else:
        theta = np.arctan2(to_point[1] - from_node.point[1], to_point[0] - from_node.point[0])
        return from_node.point[0] + step_size * np.cos(theta), from_node.point[1] + step_size * np.sin(theta)


def line_collision_check(p1, p2):
    steps = int(distance(p1, p2) / 0.5)
    for i in range(1, steps):
        interp = (p1[0] + i / steps * (p2[0] - p1[0]), p1[1] + i / steps * (p2[1] - p1[1]))
        if any(ox <= interp[0] <= ox + ow and oy <= interp[1] <= oy + oh for ox, oy, ow, oh in obstacles):
            return False
    return True


def smooth_path(path, obstacles, max_iterations=100):
    iteration = 0
    while iteration < max_iterations and len(path) > 2:
        path_length = len(path)
        # Randomly pick two indices to attempt to shortcut
        i1, i2 = sorted(random.sample(range(path_length), 2))
        if i2 - i1 > 1 and line_collision_check(path[i1].point, path[i2].point):
            # If direct path is possible, remove intermediate nodes
            path = path[:i1 + 1] + path[i2:]
        iteration += 1
    return path


def extract_path(goal_node):
    path = []
    current = goal_node
    while current:
        path.append(current)
        current = current.parent
    path.reverse()
    return path


def rrt(start, goal, use_greedy=False, use_buffer=False):
    nodes = [Node(start)]
    for i in range(num_iterations):
        if use_greedy and random.random() < 0.2:
            random_point = goal
        elif use_buffer:
            random_point = find_free_point(buffer)
        else:
            random_point = find_free_point()

        nearest = nearest_node(nodes, random_point)
        new_point = steer(nearest, random_point, step_size)
        if line_collision_check(nearest.point, new_point):
            new_node = Node(new_point, nearest)
            nodes.append(new_node)
            if distance(new_point, goal) <= goal_radius:
                return nodes, new_node  # Return path to goal
    return nodes, None  # Return the tree if goal not reached


def run_simulation(variant, num_trials=1000):
    results = []
    for _ in range(num_trials):
        start = find_free_point(buffer if variant in ["buffer", "greedy_buffer"] else 0)
        goal = find_free_point(buffer if variant in ["buffer", "greedy_buffer"] else 0)

        if variant == 'basic':
            nodes, _ = rrt(start, goal, use_greedy=False, use_buffer=False)
        elif variant == 'buffer':
            nodes, _ = rrt(start, goal, use_greedy=False, use_buffer=True)
        elif variant == 'greedy':
            nodes, _ = rrt(start, goal, use_greedy=True, use_buffer=False)
        elif variant == 'greedy_buffer':
            nodes, _ = rrt(start, goal, use_greedy=True, use_buffer=True)

        results.append(len(nodes))
    return np.mean(results), np.std(results)


def main_simulation():
    num_trials = 1000
    variants = ['basic', 'greedy', 'buffer', 'greedy_buffer']
    results = {variant: run_simulation(variant, num_trials) for variant in variants}

    # Plotting results
    means = [results[variant][0] for variant in variants]
    stds = [results[variant][1] for variant in variants]

    fig, ax = plt.subplots()
    ax.bar(variants, means, yerr=stds, capsize=5)
    ax.set_ylabel('Average Number of Nodes (± std dev)')
    ax.set_title('Comparison of RRT Variants Over {} Trials'.format(num_trials))
    ax.set_xlabel('RRT Variant')
    plt.show()


def doplot(starting, ending):
    variants = {
        'basic': {'use_greedy': False, 'use_buffer': False, 'smooth': False},
        'basic_buffer': {'use_greedy': False, 'use_buffer': True, 'smooth': False},
        'basic_buffer_greedy': {'use_greedy': True, 'use_buffer': True, 'smooth': False},
        'smooth_sampled_path': {'use_greedy': False, 'use_buffer': True, 'smooth': True}
    }
    color_map = {
        'basic': 'blue',
        'basic_buffer': 'green',
        'basic_buffer_greedy': 'red',
        'smooth_sampled_path': 'purple'
    }

    fig, ax = plt.subplots()
    # Draw the border with a different color and dashed line
    ax.plot([0, width, width, 0, 0], [0, 0, height, height, 0], color='darkgrey', linestyle='dashed')

    # Plotting obstacles with a darker shade
    for (x, y, w, h) in obstacles:
        ax.fill([x, x + w, x + w, x, x], [y, y, y + h, y + h, y], 'grey', alpha=0.8)

    start = starting
    goal = ending

    # Plot start and goal points with different shapes
    ax.plot(start[0], start[1], 'o', markersize=10, markeredgecolor='black', markerfacecolor='lime',
            label='Start')  # lime circle for start
    ax.plot(goal[0], goal[1], 'X', markersize=10, markerfacecolor='red', label='Goal')  # red 'X' for goal

    for name, settings in variants.items():
        nodes, goal_node = rrt(start, goal, settings['use_greedy'], settings['use_buffer'])
        path = extract_path(goal_node)
        if settings['smooth'] and path:
            path = smooth_path(path, obstacles)

        if path:
            path_points = [[node.point[0] for node in path], [node.point[1] for node in path]]
            ax.plot(path_points[0], path_points[1], label=name, color=color_map[name],
                    linestyle='-')  # Dashed lines for paths

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.set_title('Comparison of RRT Variants with Different Configurations')
    ax.grid(True)  # Enable grid for better alignment visibility
    ax.legend()
    plt.show()


def main():
    variants = {
        'basic': {'use_greedy': False, 'use_buffer': False, 'smooth': False},
        'basic_buffer': {'use_greedy': False, 'use_buffer': True, 'smooth': False},
        'basic_buffer_greedy': {'use_greedy': True, 'use_buffer': True, 'smooth': False},
        'smooth_sampled_path': {'use_greedy': False, 'use_buffer': True, 'smooth': True}
    }
    color_map = {
        'basic': 'blue',
        'basic_buffer': 'green',
        'basic_buffer_greedy': 'red',
        'smooth_sampled_path': 'purple'
    }

    start = find_free_point()
    goal = find_free_point()

    for name, settings in variants.items():
        fig, ax = plt.subplots()

        # Draw the border with a different color and dashed line
        ax.plot([0, width, width, 0, 0], [0, 0, height, height, 0], color='darkgrey', linestyle='dashed')

        # Plotting obstacles with a darker shade
        for (x, y, w, h) in obstacles:
            ax.fill([x, x + w, x + w, x, x], [y, y, y + h, y + h, y], 'grey', alpha=0.8)

        # Plot start and goal points with different shapes
        ax.plot(start[0], start[1], 'o', markersize=10, markeredgecolor='black', markerfacecolor='lime',
                label='Start')  # lime circle for start
        ax.plot(goal[0], goal[1], 'X', markersize=10, markerfacecolor='red', label='Goal')  # red 'X' for goal

        nodes, goal_node = rrt(start, goal, settings['use_greedy'], settings['use_buffer'])
        path = extract_path(goal_node)
        if settings['smooth'] and path:
            path = smooth_path(path, obstacles)

        if path:
            path_points = [[node.point[0] for node in path], [node.point[1] for node in path]]
            ax.plot(path_points[0], path_points[1], label=name, color=color_map[name],
                    linestyle='-')  # Dashed lines for paths

        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect('equal')
        ax.set_title(f'RRT Variant: {name}')
        ax.grid(True)  # Enable grid for better alignment visibility
        ax.legend()
        plt.show()
    doplot(start, goal)


def main_simulation():
    num_trials = 1000
    variants = ['basic', 'greedy', 'buffer', 'greedy_buffer']
    results = {variant: run_simulation(variant, num_trials) for variant in variants}

    # Plotting results
    means = [results[variant][0] for variant in variants]
    stds = [results[variant][1] for variant in variants]

    fig, ax = plt.subplots()
    ax.bar(variants, means, yerr=stds, capsize=5)
    ax.set_ylabel('Average Number of Nodes (± std dev)')
    ax.set_title('Comparison of RRT Variants Over {} Trials'.format(num_trials))
    ax.set_xlabel('RRT Variant')
    plt.show()


if __name__ == "__main__":
    main()
    main_simulation()


END
