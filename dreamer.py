import numpy as np 

import torch.nn as nn

from dm_control import suite
from dm_control.suite.wrappers import pixels
from networks import Encoder



class Dreamer:
    def __init__(self, input_size: int, output_size: int, action_size: int, config):
        self.config = config

        # ----- NN Creation -----
        self.encorder = Encoder()


























def train(env, nb_seeds: int = 5, ):

    # ------------- Generate and store sample of (action, state, reward) from random action -------------
    time_step = env.reset()
    random_state_dataset = []
    for seed in range(nb_seeds):
        print(f"Genereting 1000 sample with seed : {seed}")
        while not time_step.last():
            np.random.seed(seed)
            random_action = np.random.uniform(low=action_space_spec.minimum, high=action_space_spec.maximum, size=action_space_spec.shape)
            time_step = env.step(random_action)
            sample = (random_action, time_step.observation["pixels"], time_step.reward)
            random_state_dataset.append(sample)
        time_step = env.reset()

    # NN's size parameters
    input_size = random_state_dataset[0][1].flatten()
    action_dim = 1

    DreamerModel = Dreamer()

    return random_state_dataset


if __name__ == "__main__":

    print("Environment creation ...")
    env = suite.load(domain_name="cartpole", task_name="swingup")
    env = pixels.Wrapper(env=env, render_kwargs={"height": 64, "width": 64, "camera_id": 0})
    print("Environment created\n")


    test = train(env=env)
    print(len(test))
    print(test[0])
