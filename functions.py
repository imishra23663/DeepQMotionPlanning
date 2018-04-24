import numpy as np

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
