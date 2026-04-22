import numpy as np 

import torch.nn as nn

from dm_control import suite
from dm_control.suite.wrappers import pixels
from networks import Encoder, Decoder, RecurrentModel, Prior, Posterior, RewardModel, ContinueModel, Actor, Critic



class Dreamer:
    def __init__(self, observation_shape, action_size, config):
        self.config = config
        self.observation_shape = observation_shape
        self.action_size = action_size

        self.recurrent_size = config.recurrent_size
        self.latent_size = config.latent_size
        self.encoded_obs_size = config.encoded_obs_size
        self.full_state_size = self.recurrent_size + self.latent_size

        # ----- NN Creation -----
        self.encorder        = Encoder(observation_shape, self.config.encoded_obs_size,                  config.encorder)
        self.decoder         = Decoder(self.encoded_obs_size, self.observation_shape,                    config.decoder)
        self.recurrent_model = RecurrentModel(self.config.recurrent_size, self.latent_size, action_size, config.recurrent_model)
        self.prior           = Prior(self.recurrent_size, self.latent_size,                              config.prior)
        self.posterior       = Posterior(self.recurrent_size+self.encoded_obs_size, self.latent_size,    config.posterior)
        self.reward_model    = RewardModel(self.full_state_size,                                         config.reward_model)
        self.continue_model  = ContinueModel(self.full_state_size,                                       config.continuation)
        self.actor           = Actor(self.full_state_size, self.action_size,                             config.actor)
        self.critic          = Critic(self.full_state_size,                                              config.critic)

    
    def world_model_training(self, data):
        encorded_observation = 
        return
    

    def behavior_traning(self):
        return






















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
