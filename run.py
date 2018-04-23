from GridWorld import GridWorld
from Robot import Robot

env = GridWorld("grid-small.txt")
env.print_map()
gamma = 0.9

start = [0, 0]
agent = Robot(env, gamma)
epochs = 500
decay = 0.99
rvm_max_iter = 500
max_step = 1000
epsilon = 1
epsilon_threshold = 0.001
verbose = True
verbose_iteration = 1
steps, rewards = agent.learn(epochs, decay, rvm_max_iter, max_step, epsilon,  start, verbose, verbose_iteration)
path = agent.get_path(start)
print(path)
