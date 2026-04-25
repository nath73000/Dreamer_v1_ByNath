import numpy as np 

import torch
import torch.nn as nn

from dm_control import suite
from dm_control.suite.wrappers import pixels
from networks import Encoder, Decoder, RecurrentModel, Prior, Posterior, RewardModel, ContinueModel, Actor, Critic
from buffer import ReplayBuffer



class Dreamer:
    def __init__(self, observation_shape, action_size, config, device):
        self.config = config
        self.device = device
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

        self.buffer = ReplayBuffer(self.observation_shape, self.action_size, config.buffer, device)

        self.world_model_parameters = (list(self.encorder.parameters()) + 
                                       list(self.decoder.parameters()) + 
                                       list(self.recurrent_model.parameters()) +
                                       list(self.prior.parameters()) + 
                                       list(self.posterior.parameters()) + 
                                       list(self.reward_model.parameters()) + 
                                       list(self.continue_model.parameters()))
        
        self.world_model_optimizer = torch.optim.Adam(self.world_model_parameters, lr=self.config.world_model_learning_rate)
        self.actor_optimiwer       = torch.optim.Adam(self.actor.parameters(),     lr=self.config.actor_learning_rate)
        self.critic_optimiwer      = torch.optim.Adam(self.critic.parameters(),    lr=self.config.critic_learning_rate)

        self.total_episodes       = 0
        self.total_env_steps      = 0
        self.total_gradient_steps = 0




    
    def world_model_training(self, data):
        return
    

    def behavior_traning(self):
        return


    @torch.no_grad()
    def environement_interaction(self, env, nb_episodes, seed=None, evaluation=False):
        scores = []
        for episod in range(nb_episodes):
            recurrent_state = torch.zeros(1, self.recurrent_size, device=self.device)
            latent_state    = torch.zeros(1, self.latent_size,    device=self.device)
            action          = torch.zeros(1, self.action_size,    device=self.device)

            _, _, _, observation = env.reset()
            encoded_observation = self.encorder(torch.from_numpy(observation["pixels"]).float().unsqueeze(0).to(self.device))

            current_score, step_count, done, frames = 0, 0, False, []
            while not done:
                recurrent_state = self.recurrent_model(recurrent_state, latent_state, action)
                latent_state, _ = self.posterior(torch.cat((encoded_observation.view(1, -1), recurrent_state), -1))

                action = self.actor(torch.cat((latent_state, recurrent_state), -1))
                action_numpy = action.cpu().numpy().reshape(-1)

                time_step = env.step(action_numpy)
                next_observation, reward, done = time_step.observation["pixels"], time_step.reward, time_step.last()

                if not evaluation:
                    self.buffer.add(observation, action, reward, next_observation, done)

                encoded_observation = self.encorder(torch.from_numpy(next_observation).float().unsqueeze(0).to(self.device))

                observation    = next_observation
                current_score += reward
                step_count    += 1

                if done:
                    scores.append(current_score)
                    if not evaluation:
                        self.total_episodes  += 1
                        self.total_env_steps += step_count
        
        return sum(scores)/nb_episodes if nb_episodes else None












# def train(env, nb_seeds: int = 5, ):

#     # ------------- Generate and store sample of (action, state, reward) from random action -------------
#     time_step = env.reset()
#     random_state_dataset = []
#     for seed in range(nb_seeds):
#         print(f"Genereting 1000 sample with seed : {seed}")
#         while not time_step.last():
#             np.random.seed(seed)
#             random_action = np.random.uniform(low=action_space_spec.minimum, high=action_space_spec.maximum, size=action_space_spec.shape)
#             time_step = env.step(random_action)
#             sample = (random_action, time_step.observation["pixels"], time_step.reward)
#             random_state_dataset.append(sample)
#         time_step = env.reset()

#     # NN's size parameters
#     input_size = random_state_dataset[0][1].flatten()
#     action_dim = 1

#     DreamerModel = Dreamer()

#     return random_state_dataset


if __name__ == "__main__":

    print("Environment creation ...")
    env = suite.load(domain_name="cartpole", task_name="swingup")
    env = pixels.Wrapper(env=env, render_kwargs={"height": 64, "width": 64, "camera_id": 0})
    print("Environment created\n")


    test = train(env=env)
    print(len(test))
    print(test[0])
