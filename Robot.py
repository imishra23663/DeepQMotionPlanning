import time
import numpy as np
from DNN import DNN
from functions import get_predicted_Q_values, epsilon_greedy_action


"""
# This class represents the reinforcement learning agent which
# moves in the environment seeking the goal

    Author: Jeet
Data Modified: 04/11/2018 
"""


class Robot:
    def __init__(self, simulation, gamma):
        self.replay_cs = np.zeros((0, 2), dtype=np.float32)
        self.replay_ca = np.zeros((0, 1), dtype=np.int32)
        self.replay_r = np.zeros((0, 1), dtype=np.float32)
        self.replay_ns = np.zeros((0, 2), dtype=np.float32)
        self.replay_cs_type = np.zeros((0, 1), dtype=np.int32)
        self.gamma = gamma
        self.simulation = simulation
        self.c = 10
        self.replay_size = 50000
        self.sample_size = 64
        self.learning_model = DNN()
        dense_layers = np.array([8, 64, 256, 64])
        activations = ["relu"]
        loss_functions = "mean_squared_error"
        optimizer_name = "Adam"
        input_size = (2,)
        output_size = self.simulation.actions.shape[0]
        self.learning_model.create(input_size, output_size, dense_layers, activations,loss_functions, optimizer_name)
        self.target_model = None
        self.epochs = 10  # default values
        self.decay = 0.99  # default values
        self.epsilon_threshold = 0.01  # default values
        self.max_step = 1000  # default values
        self.epsilon = 1  # default values
        self.start_pc = [0, 0, 0, 0, 0, 0]  # default values
        self.goal_pc = [0.5, 0.5, 0, 0, 0, 0]  # default values
        self.verbose = True  # default values
        self.verbose_iteration = 1  # default values

        self.epoch_count = 0  # to track the epoch since its triggered by simulator

    def set_run_args(self, epochs, decay, epsilon_threshold, max_step, epsilon, start_pc, goal_pc,
                     verbose, verbose_iteration):

        '''
        This method sets the running arguments for the model

        :param epochs: Number of epochs to train
        :param decay: favtor decay the randomness
        :param epsilon_threshold: minimum limit of randomness
        :param max_step: maximum steps for each attempt
        :param epsilon: randomness probability
        :param start: start position cooridinate
        :param verbose: boolean for print the message while training
        :param verbose_iteration: number of iteration after msg will be printed
        :return:
        '''

        self.epochs = epochs
        self.decay = decay
        self.epsilon_threshold = epsilon_threshold
        self.max_step = max_step
        self.epsilon = epsilon
        self.start_pc = start_pc
        self.goal_pc = goal_pc
        self.verbose = verbose
        self.verbose_iteration = verbose_iteration

    # This methods is used to get the Q values for all the states us the the learning model
    def get_Q_values(self, model, cs):
        cs_Q = get_predicted_Q_values(model, cs)
        return cs_Q

    # This is method for implementing the Q learning where the agent
    # tries to find the goal using the approximated Q values and update the
    # Q values
    def find_goal(self, epsilon, max_step):
        cs = self.simulation.get_current_state()
        step = 0
        rewards = 0
        goal_found = False
        while True:
            csQ = self.get_Q_values(self.learning_model, cs)
            cs_maxQ, ca = epsilon_greedy_action(csQ, epsilon)
            reward, ns, goal_reached = self.simulation.get_next_state(ca, step)

            self.replay_cs = np.vstack((self.replay_cs, cs))
            self.replay_ca = np.vstack((self.replay_ca, ca))
            self.replay_r = np.vstack((self.replay_r, np.array([[reward]])))
            self.replay_ns = np.vstack((self.replay_ns, ns))
            rewards += reward
            if goal_reached:
                print("Goal Found")
                self.replay_cs_type = np.vstack((self.replay_cs_type, np.array([[1]])))
                goal_found
                break
            elif step >= max_step-1:
                self.replay_cs_type = np.vstack((self.replay_cs_type, np.array([[0]])))
                break
            else:
                self.replay_cs_type = np.vstack((self.replay_cs_type, np.array([[0]])))
                step += 1
                cs = ns

            if self.replay_cs.shape[0] < self.sample_size:
                sample_size = self.replay_cs.shape[0]
            else:
                sample_size = self.sample_size
            idx = np.random.choice(self.replay_cs.shape[0], size=sample_size, replace=False)
            cs_mb = self.replay_cs[idx]
            ca_mb = self.replay_ca[idx]
            r_mb = self.replay_r[idx]
            ns_mb = self.replay_ns[idx]
            cs_type_mb = self.replay_cs_type[idx]
            target_Q = self.get_Q_values(self.learning_model, cs_mb)
            for i in range(ca_mb.shape[0]):
                if cs_type_mb[i] == 1:
                    target_Q[i, ca_mb[i]] = r_mb[i]
                else:

                    target_Q[i, ca_mb[i]] = r_mb[i] + self.gamma * np.max(
                        self.get_Q_values(self.target_model, ns_mb[i].reshape(1, -1)))

            # select mini batch
            self.learning_model.train(cs_mb, target_Q, epochs=1, batch_size=64, verbose=0)

        step += 1
        # return the state space created in this training
        return step, rewards, goal_found

    ''' This method refers to the learning phase of the agent
        Use verbose to print training message and verbose_iteration to print after verbose_iteration'''

    def learn(self):
        steps = []
        rewards = []
        self.simulation.robot.setConfig(self.start_pc)
        for epoch in range(1, self.epochs + 1):
            self.simulation.reset(self.start_pc)
            diff = self.replay_cs.shape[0] - self.replay_size
            if diff > 0:
                self.replay_cs = np.delete(self.replay_cs, (range(0, diff)), axis=0)
                self.replay_ca = np.delete(self.replay_ca, (range(0, diff)), axis=0)
                self.replay_r = np.delete(self.replay_r, (range(0, diff)), axis=0)
                self.replay_ns = np.delete(self.replay_ns, (range(0, diff)), axis=0)

            if epoch % self.c == 1:
                self.target_model = self.learning_model.copy()

            step, reward, goal_found = self.find_goal(self.epsilon, self.max_step)
            rewards.append(reward)
            steps.append(step)

            # To display training messages
            if self.verbose and epoch % self.verbose_iteration == 0:
                print("Epoch:", epoch, "Epsilon:", np.round(self.epsilon, 4), "Steps: ", step, "Rewards: ", reward)

            if self.epsilon > self.epsilon_threshold:
                self.epsilon *= self.decay
            else:
                self.epsilon = 0
        return steps, rewards

    # This method refers to the phase where the agent finds the goal
    # using the learned policy
    def get_path(self, start_pc, epsilon=0):
        self.simulation.reset(start_pc)
        trajectory = np.zeros((0, 2), dtype=np.float32)
        cs = self.simulation.get_current_state()
        step = 0
        rewards = 0
        while True:
            trajectory = np.vstack((trajectory, cs))
            csQ = self.get_Q_values(self.learning_model, cs)
            cs_maxQ, ca = epsilon_greedy_action(csQ, epsilon)
            reward, ns, goal_reached = self.simulation.get_next_state(ca, step, query=True)
            rewards += rewards
            step += 1
            if goal_reached:
                trajectory = np.vstack((trajectory, ns))
                break
            cs = ns

        return step, rewards, trajectory