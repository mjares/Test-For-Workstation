import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d


def set_fixed_square_path():
    """ Generates a fixed square path.
    """

    steps = 8  # Number of waypoints
    x_path = [0, 0, 5, 0, -5, 0, 5, 5]
    y_path = [0, 0, 0, 5, 0, -5, 0, 0]
    z_path = [0, 5, 5, 5, 5, 5, 5, 5]
    interval_steps = 20
    goals = []
    safe_region = []
    for i in range(steps):
        if i < steps-1:
            # Create linespace between waypoint i and i+1
            x_lin = np.linspace(x_path[i], x_path[i+1], interval_steps)
            y_lin = np.linspace(y_path[i], y_path[i+1], interval_steps)
            z_lin = np.linspace(z_path[i], z_path[i+1], interval_steps)
        else:
            x_lin = np.linspace(x_path[i], x_path[i], interval_steps)
            y_lin = np.linspace(y_path[i], y_path[i], interval_steps)
            z_lin = np.linspace(z_path[i], z_path[i], interval_steps)

        goals.append([x_path[i], y_path[i], z_path[i]])
        #for each pos in linespace append a goal
        safe_region.append([])
        for j in range(interval_steps):
            safe_region[i].append([x_lin[j], y_lin[j], z_lin[j]])

    return goals, safe_region, steps


def set_random_straight_line_path():
    """ Generates a random straight line path.
    """

    # Initializing
    goals = []
    safe_region = []
    # Generate Random destination
    limit = 10
    rng = default_rng()
    x_dest = rng.integers(-limit, limit)
    y_dest = rng.integers(-limit, limit)
    z_dest = rng.integers(5, limit)
    # Generate Path
    steps = 4  # Number of waypoints
    x_path = [0, 0, x_dest, x_dest]
    y_path = [0, 0, y_dest, y_dest]
    z_path = [5, 5, z_dest, z_dest]
    interval_steps = 50

    for i in range(steps):
        if i < steps-1:
            # Create linespace between waypoint i and i+1
            x_lin = np.linspace(x_path[i], x_path[i+1], interval_steps)
            y_lin = np.linspace(y_path[i], y_path[i+1], interval_steps)
            z_lin = np.linspace(z_path[i], z_path[i+1], interval_steps)
        else:
            x_lin = np.linspace(x_path[i], x_path[i], interval_steps)
            y_lin = np.linspace(y_path[i], y_path[i], interval_steps)
            z_lin = np.linspace(z_path[i], z_path[i], interval_steps)

        goals.append([x_path[i], y_path[i], z_path[i]])
        #for each pos in linespace append a goal
        safe_region.append([])
        for j in range(interval_steps):
            safe_region[i].append([x_lin[j], y_lin[j], z_lin[j]])

    return goals, safe_region, steps


def set_random_v_shaped_path():
    """ Generates a random v-shaped (two straight lines and a corner) path.
    """

    # Initializing
    goals = []
    safe_region = []
    # Generate Random destination
    limit = 10
    min_dist = 4
    min_angle = 70  # degrees
    p0 = [0, 0, 5]
    x0, y0, z0 = p0
    # p1
    p1, _ = _generate_point_minimum_distance(p0, min_dist, limit)
    x1, y1, z1 = p1
    # p2
    bad_p2 = True
    while bad_p2:
        p2, _ = _generate_point_minimum_distance(p1, min_dist, limit)
        x2, y2, z2 = p2
        turn_angle = _calculate_angle(p0, p1, p2)
        if turn_angle >= min_angle:
            bad_p2 = False
    x_dest = x2
    y_dest = y2
    z_dest = z2

    # Generate Path
    steps = 4  # Number of waypoints
    x_path = [x0, x1, x_dest, x_dest]
    y_path = [y0, y1, y_dest, y_dest]
    z_path = [z0, z1, z_dest, z_dest]
    interval_steps = 50

    for i in range(steps):
        if i < steps - 1:
            # Create linespace between waypoint i and i+1
            x_lin = np.linspace(x_path[i], x_path[i + 1], interval_steps)
            y_lin = np.linspace(y_path[i], y_path[i + 1], interval_steps)
            z_lin = np.linspace(z_path[i], z_path[i + 1], interval_steps)
        else:
            x_lin = np.linspace(x_path[i], x_path[i], interval_steps)
            y_lin = np.linspace(y_path[i], y_path[i], interval_steps)
            z_lin = np.linspace(z_path[i], z_path[i], interval_steps)

        goals.append([x_path[i], y_path[i], z_path[i]])
        # for each pos in linespace append a goal
        safe_region.append([])
        for j in range(interval_steps):
            safe_region[i].append([x_lin[j], y_lin[j], z_lin[j]])

    return goals, safe_region, steps


def plot_3d_trajectory(trajectory, safe_region, fault_type, limit=6, show=True, plot_safebounds=True):
    x_c = trajectory[:, 0]
    y_c = trajectory[:, 1]
    z_c = trajectory[:, 2]

    fig = plt.figure()
    ax = axes3d.Axes3D(fig)
    lim = limit
    ax.set_xlim3d([-lim, lim])
    ax.set_xlabel('X')
    ax.set_ylim3d([-lim, lim])
    ax.set_ylabel('Y')
    ax.set_zlim3d([0, lim])
    ax.set_zlabel('Z')

    ax.cla()
    ax.set_xlim3d([-lim, lim - 2])
    ax.set_xlabel('X')
    ax.set_ylim3d([-lim + 2, lim])
    ax.set_ylabel('Y')
    ax.set_zlim3d([0, lim])
    ax.set_zlabel('Z')

    ax.plot3D(x_c, y_c, z_c, linewidth=2, c="r", label=f'{fault_type}')
    ax.scatter(x_c[-1], y_c[-1], z_c[-1], c="r")

    # Plotting safe region
    if plot_safebounds:
        for j in range(len(safe_region)):
            for pos in safe_region[j]:
                ax.scatter(pos[0], pos[1], pos[2], alpha=0.02, linewidths=50, c="g")

    ax.legend()
    plt.draw()

    if show:
        plt.show()


def plot_features(obs, starttime):
    fig, ax = plt.subplots()
    ax = fig.add_subplot(3, 2, 1)
    ax.plot(obs[:, 0])
    ax.plot([starttime, starttime], [np.min(obs[:, 0]), np.max(obs[:, 0])], 'r')
    ax.set_ylabel('Position X')
    ax = fig.add_subplot(3, 2, 3)
    ax.plot(obs[:, 1])
    ax.plot([starttime, starttime], [np.min(obs[:, 1]), np.max(obs[:, 1])], 'r')
    ax.set_ylabel('Position Y')
    ax = fig.add_subplot(3, 2, 5)
    ax.plot(obs[:, 2])
    ax.plot([starttime, starttime], [np.min(obs[:, 2]), np.max(obs[:, 2])], 'r')
    ax.set_ylabel('Position Z')
    ax = fig.add_subplot(3, 2, 2)
    ax.plot(obs[:, 3])
    ax.plot([starttime, starttime], [np.min(obs[:, 3]), np.max(obs[:, 3])], 'r')
    ax.set_ylabel('Attitude tTheta')
    ax = fig.add_subplot(3, 2, 4)
    ax.plot(obs[:, 4])
    ax.plot([starttime, starttime], [np.min(obs[:, 4]), np.max(obs[:, 4])], 'r')
    ax.set_ylabel('Attitude Phi')
    ax = fig.add_subplot(3, 2, 6)
    ax.plot(obs[:, 5])
    ax.plot([starttime, starttime], [np.min(obs[:, 5]), np.max(obs[:, 5])], 'r')
    ax.set_ylabel('Attitude Gamma')

    plt.show()

def _generate_point_minimum_distance(p0, min_d, limit=np.inf):

    x0, y0, z0 = p0
    rng = default_rng()
    x1 = rng.integers(-limit, limit)
    y1 = rng.integers(-limit, limit)

    z1_min = z0 + max(0, min_d**2 - (x0 - x1)**2 - (y0 - y1)**2)  # Most likely 0. Need to fix

    z1 = rng.integers(min(z1_min, limit - 1), limit)  # What if z1_min is not large enough
    p1 = [x1, y1, z1]

    return p1, z1_min


def _calculate_angle(p0, p1, p2):
    p0 = np.array(p0)
    p1 = np.array(p1)
    p2 = np.array(p2)
    v1 = p1 - p0
    v2 = p2 - p1

    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = 180 - np.degrees(np.arccos(dot_product))

    return angle
