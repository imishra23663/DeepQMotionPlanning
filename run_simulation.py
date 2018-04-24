import sys
import time
import numpy as np
from Robot import Robot
from klampt import WorldModel
from Simmulation import Simulation
from threading import Thread
from Builder import Builder

if __name__ == "__main__":
    if len(sys.argv)<=1:
        print "USikdAGE: kinematicSim.py [world_file]"
        exit()

    world = WorldModel()
    for fn in sys.argv[1:]:
        res = world.readFile(fn)
        if not res:
            raise RuntimeError("Unable to load model " + fn)

    # np.random.seed(11)
    np.random.seed(41)
    simulation = Simulation(world)
    start_pc = [0, 0, 0.0, 0.02, 0, 0]
    goal_pc = [-0.8, -0.9, 0.02, 0, 0, 0]
    simulation.create(start_pc, goal_pc)
    gamma = 0.99
    agent = Robot(simulation, gamma)

    epochs = 10000
    decay = 0.993
    max_step = 2000
    epsilon = 1
    epsilon_threshold = 0.0001
    verbose = True
    verbose_iteration = 1
    agent.set_run_args(epochs, decay, epsilon_threshold, max_step, epsilon, start_pc, goal_pc, verbose,
                       verbose_iteration)

    # thread = Thread(target=simulation.run)
    # thread.start()
    agent.learn()

    simulation.vis.kill()
    # Save the model
    agent.learning_model.save("model.h5")
    # To suppress scientific notation
    np.set_printoptions(suppress=True)
    print(agent.get_path(start_pc))
    #thread.join()
