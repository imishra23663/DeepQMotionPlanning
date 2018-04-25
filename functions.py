import numpy as np
import matplotlib.pyplot as plt

"""
This file contains the utility helper functions that is used in the other classes
for performing sie basic computation

Author: Jeet
Date Modified
"""
import h5py

def write__to_h5_file(filename, **data):
    """
    This function writes the model data to an h5 file
    :param filename: name of the file to write
    :param data: A variable argument containing th data
    :return:
    """
    hf = h5py.File(filename)
    for key, value in data.items():
        hf.create_dataset(key, data=value)
    hf.close()


def get_random_Q_values(shape):
    """
    This method generates random numbers of given shape
    :param shape: shape of the the array of numbers to generate
    :return:
    """
    np.random.rand(shape=shape)


def get_predicted_Q_values(learning_model, state):
    # This  Method predicts the Q values for all the actions at a given state
    # using  the learning model
    Q = learning_model.predict(state)  # 1e-30 to avoid 0 values for this state always
    return Q


def epsilon_greedy_action(QValues, epsilon=0.01):
    # This  Method to choose action from a state using the
    # estimated Q values in epsilon-greedy fashion
    max_Q = np.max(QValues)

    equal_max = np.where(QValues == max_Q)[1]
    action = np.random.choice(equal_max)

    if np.random.random() < epsilon:
        random_action = np.random.randint(QValues.shape[1])
        action = random_action
        max_Q = QValues[0, action]

    return max_Q, action


def plot_Q_contour(agent, start, goal, terrain_limit):
    xs = np.round(np.arange(-terrain_limit[0], terrain_limit[0], 0.05).reshape(-1, 1), 3)
    ys = np.round(np.arange(-terrain_limit[1], terrain_limit[1], 0.05).reshape(-1, 1), 3)
    X, Y = np.meshgrid(xs, ys)

    # state the state matrix
    states = np.zeros((xs.shape[0]*ys.shape[0], 2))
    for i in range(xs.shape[0]):
        for j in range(ys.shape[0]):
            states[i*ys.shape[0]+j, 0] = xs[i, 0]
            states[i*ys.shape[0]+j, 1] = ys[j, 0]
    Q_values = get_predicted_Q_values(agent.learning_model, states)
    maxQ = np.max(Q_values, axis=1)
    # create A 2d matrix for denoting q value for each state
    Q = np.zeros((xs.shape[0], ys.shape[0]))
    for i in range(xs.shape[0]):
        for j in range(ys.shape[0]):
            Q[i, j] = maxQ[i*ys.shape[0]+j]

    plt.figure()
    cs = plt.contourf(X, Y, Q)
    plt.colorbar(cs)
    plt.text(goal[1], goal[0], 'G')
    plt.text(start[1], start[0], 'S')
    plt.ylabel("max Q")
    plt.show()


def plot_trajectory(path, terrain_limit):
    path_x = path[:, 0]
    path_y = path[:, 0]
    plt.figure()
    plt.plot(path_y, path_x, color='black')
    plt.xlim([-terrain_limit[0], terrain_limit[0]])
    plt.ylim([-terrain_limit[1], terrain_limit[1]])
    plt.show()

