import numpy as np 

import torch
import torch.nn as nn
from torch.distributions import kl_divergence, Independent, OneHotCategoricalStraightThrough, Normal

from dm_control import suite
from dm_control.suite.wrappers import pixels
from networks import Encoder, Decoder, RecurrentModel, Prior, Posterior, RewardModel, ContinueModel, Actor, Critic
from utils import computeLambdaValues, Moments
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
        self.actor_optimizer       = torch.optim.Adam(self.actor.parameters(),     lr=self.config.actor_learning_rate)
        self.critic_optimizer      = torch.optim.Adam(self.critic.parameters(),    lr=self.config.critic_learning_rate)

        self.valueMoments = Moments(device)

        self.total_episodes       = 0
        self.total_env_steps      = 0
        self.total_gradient_steps = 0


    def world_model_training(self, data):
        encoded_observation = self.encorder(data.observations.view(-1, *self.observation_shape)).view(self.config.batch_size, self.config.batch_length, -1)
        previous_recurrent_state = torch.zeros(self.config.batch_size, self.recurrent_size,  device=self.device)
        previous_latent_state    = torch.zeros(self.config.batch_size, self.latent_size,     device=self.device)

        recurrent_states, priors_logits, posteriors, posteriors_logits = [], [], [], []
        for t in range(1, self.config.batch_length):
            recurrent_state             = self.recurrent_model(previous_recurrent_state, previous_latent_state, data.actions[:, t-1])
            _, prior_logits             = self.prior(recurrent_state)
            posterior, posterior_logits = self.posterior(torch.cat(recurrent_state, encoded_observation[:, t], -1))

            recurrent_states.append(recurrent_state)
            priors_logits.append(prior_logits)
            posteriors.append(posterior)
            posteriors_logits.append(posterior_logits)

            previous_recurrent_state = recurrent_state
            previous_latent_state    = posterior

        recurrent_states = torch.stack(recurrent_state,              dim=1)
        prior_logits =     torch.stack(prior_logits,                 dim=1)
        posteriors =       torch.stack(posteriors,                   dim=1)
        posterior_logits = torch.stack(posterior_logits,             dim=1)
        full_states =       torch.cat((recurrent_states, posteriors), dim=-1)

        reconstruction_means         =  self.decoder(full_states.view(-1, self.full_state_size)).view(self.config.batch_size, self.config.batch_length-1, *self.observation_shape)
        reconstruction_distribution  =  Independent(Normal(reconstruction_means, 1), len(self.observation_shape))
        reconstruction_loss          = -reconstruction_distribution.log_prob(data.observations[:, 1:]).mean()

        reward_distribution  =  self.rewardPredictor(full_states)
        reward_loss          = -reward_distribution.log_prob(data.rewards[:, 1:].squeeze(-1)).mean()

        prior_distribution        = Independent(OneHotCategoricalStraightThrough(logits=priors_logits              ), 1)
        prior_distribution_sg     = Independent(OneHotCategoricalStraightThrough(logits=priors_logits.detach()     ), 1)
        posterior_distribution    = Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits          ), 1)
        posterior_distribution_sg = Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits.detach() ), 1)

        prior_loss       = kl_divergence(posterior_distribution_sg, prior_distribution  )
        posterior_loss   = kl_divergence(posterior_distribution  , prior_distribution_sg)
        free_nats        = torch.full_like(prior_loss, self.config.free_nats)

        prior_loss       = self.config.beta_prior*torch.maximum(prior_loss, free_nats)
        posterior_loss   = self.config.beta_posterior*torch.maximum(posterior_loss, free_nats)
        kl_loss          = (prior_loss + posterior_loss).mean()

        world_model_loss =  reconstruction_loss + reward_loss + kl_loss # I think that the reconstruction loss is relatively a bit too high (11k) 

        if self.config.use_continuation_prediction:
            continue_distribution = self.continuePredictor(full_states)
            continue_loss         = nn.BCELoss(continue_distribution.probs, 1 - data.dones[:, 1:])
            world_model_loss     += continue_loss.mean()

        self.world_model_optimizer.zero_grad()
        world_model_loss.backward()
        nn.utils.clip_grad_norm_(self.world_model_parameters, self.config.gradient_clip, norm_type=self.config.gradient_norm_type)
        self.world_model_optimizer.step()

        klLoss_shift_for_graphing = (self.config.beta_prior + self.config.beta_posterior)*self.config.free_nats
        metrics = {
            "world_model_loss"        : world_model_loss.item() - klLoss_shift_for_graphing,
            "reconstruction_loss"     : reconstruction_loss.item(),
            "reward_predictor_loss"   : reward_loss.item(),
            "kl_loss"                 : kl_loss.item() - klLoss_shift_for_graphing}
        return full_states.view(-1, self.full_state_size).detach(), metrics
    

    def behavior_training(self, full_state):
        recurrent_state, latent_state = torch.split(full_state, (self.recurrent_size, self.latent_size), -1)
        full_states, logprobs, entropies = [], [], []
        for _ in range(self.config.imagination_horizon):
            action, logprob, entropy = self.actor(full_state.detach(), training=True)
            recurrent_state = self.recurrent_model(recurrent_state, latent_state, action)
            latent_state, _ = self.prior(recurrent_state)

            full_state = torch.cat((recurrent_state, latent_state), -1)
            full_states.append(full_state)
            logprobs.append(logprob)
            entropies.append(entropy)
        full_states  = torch.stack(full_states,    dim=1) # (batchSize*batchLength, imaginationHorizon, recurrentSize + latentLength*latentClasses)
        logprobs     = torch.stack(logprobs[1:],   dim=1) # (batchSize*batchLength, imaginationHorizon-1)
        entropies    = torch.stack(entropies[1:],  dim=1) # (batchSize*batchLength, imaginationHorizon-1)
        
        predicted_rewards = self.reward_model(full_states[:, :-1]).mean
        values            = self.critic(full_states).mean
        continues         = self.continue_model(full_states).mean if self.config.use_continuation_prediction else torch.full_like(predicted_rewards, self.config.discount)
        lambda_values     = computeLambdaValues(predicted_rewards, values, continues, self.config.lambda_)

        _, inverse_scale = self.valueMoments(lambda_values)
        advantages       = (lambda_values - values[:, :-1])/inverse_scale

        actor_loss = -torch.mean(advantages.detach()*logprobs + self.config.entropy_scale*entropies)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip, norm_type=self.config.gradient_normType)
        self.actor_optimizer.step()

        value_distributions  =  self.critic(full_states[:, :-1].detach())
        critic_loss          = -torch.mean(value_distributions.log_prob(lambda_values.detach()))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradient_clip, norm_type=self.config.gradient_normType)
        self.critic_optimizer.step()

        metrics = {
            "actor_loss"     : actor_loss.item(),
            "critic_loss"    : critic_loss.item(),
            "entropies"     : entropies.mean().item(),
            "logprobs"      : logprobs.mean().item(),
            "advantages"    : advantages.mean().item(),
            "criticValues"  : values.mean().item()}
        return metrics


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
