import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Independent, kl_divergence, Normal

from networks import Encoder, Decoder, RecurrentModel, Prior, Posterior, RewardModel, ContinueModel, Actor, Critic
from utils import computeLambdaValues, Moments
from buffer import ReplayBuffer


class Dreamer:
    def __init__(self, observation_shape, action_size, config, device="cpu"):
        self.config = config
        self.device = torch.device(device)
        self.observation_shape = observation_shape
        self.action_size = int(action_size)

        self.recurrent_size = config.recurrent_size
        self.latent_size = config.latent_size
        self.encoded_obs_size = config.encoded_obs_size
        self.full_state_size = self.recurrent_size + self.latent_size

        # ----- NN Creation -----
        self.encoder         = Encoder(observation_shape, self.config.encoded_obs_size,                  config.encoder)
        self.encorder        = self.encoder
        self.decoder         = Decoder(self.full_state_size, self.observation_shape,                     config.decoder)
        self.recurrent_model = RecurrentModel(self.config.recurrent_size, self.latent_size, self.action_size, config.recurrent_model)
        self.prior           = Prior(self.recurrent_size, self.latent_size,                              config.prior)
        self.posterior       = Posterior(self.recurrent_size+self.encoded_obs_size, self.latent_size,    config.posterior)
        self.reward_model    = RewardModel(self.full_state_size,                                         config.reward_model)
        self.continue_model  = ContinueModel(self.full_state_size,                                       config.continuation)
        self.actor           = Actor(self.full_state_size, self.action_size,                             config.actor)
        self.critic          = Critic(self.full_state_size,                                              config.critic)

        for module in self.modules:
            module.to(self.device)

        self.buffer = ReplayBuffer(self.observation_shape, self.action_size, config.buffer, self.device)

        self.world_model_parameters = (list(self.encoder.parameters()) +
                                       list(self.decoder.parameters()) + 
                                       list(self.recurrent_model.parameters()) +
                                       list(self.prior.parameters()) + 
                                       list(self.posterior.parameters()) + 
                                       list(self.reward_model.parameters()) + 
                                       list(self.continue_model.parameters()))
        
        self.world_model_optimizer = torch.optim.Adam(self.world_model_parameters, lr=self.config.world_model_learning_rate)
        self.actor_optimizer       = torch.optim.Adam(self.actor.parameters(),     lr=self.config.actor_learning_rate)
        self.critic_optimizer      = torch.optim.Adam(self.critic.parameters(),    lr=self.config.critic_learning_rate)

        self.valueMoments = Moments(self.device)

        self.total_episodes       = 0
        self.total_env_steps      = 0
        self.total_gradient_steps = 0


    @property
    def modules(self):
        return [
            self.encoder,
            self.decoder,
            self.recurrent_model,
            self.prior,
            self.posterior,
            self.reward_model,
            self.continue_model,
            self.actor,
            self.critic,
        ]


    def _observation_tensor(self, observation):
        observation = np.ascontiguousarray(observation)
        return torch.as_tensor(observation, device=self.device).unsqueeze(0)


    def world_model_training(self, data):
        batch_size, batch_length = data.observations.shape[:2]
        encoded_observation = self.encoder(data.observations.reshape(-1, *self.observation_shape)).reshape(batch_size, batch_length, -1)
        previous_recurrent_state = torch.zeros(batch_size, self.recurrent_size,  device=self.device)
        previous_latent_state    = torch.zeros(batch_size, self.latent_size,     device=self.device)

        recurrent_states, prior_distributions, posterior_distributions, posteriors = [], [], [], []
        for t in range(1, batch_length):
            recurrent_state             = self.recurrent_model(previous_recurrent_state, previous_latent_state, data.actions[:, t-1])
            prior_distribution, _       = self.prior(recurrent_state)
            posterior_distribution, posterior = self.posterior(torch.cat((recurrent_state, encoded_observation[:, t]), dim=-1))

            recurrent_states.append(recurrent_state)
            prior_distributions.append(prior_distribution)
            posterior_distributions.append(posterior_distribution)
            posteriors.append(posterior)

            previous_recurrent_state = recurrent_state
            previous_latent_state    = posterior

        recurrent_states = torch.stack(recurrent_states, dim=1)
        posteriors       = torch.stack(posteriors,       dim=1)
        full_states      = torch.cat((recurrent_states, posteriors), dim=-1)

        reconstruction_means         =  self.decoder(full_states.reshape(-1, self.full_state_size)).reshape(batch_size, batch_length-1, *self.observation_shape)
        reconstruction_targets       =  data.observations[:, 1:].float()
        reconstruction_targets       =  reconstruction_targets / 255.0 if reconstruction_targets.max() > 1.0 else reconstruction_targets
        reconstruction_distribution  =  Independent(Normal(reconstruction_means, 1), len(self.observation_shape))
        reconstruction_loss          = -reconstruction_distribution.log_prob(reconstruction_targets).mean()

        reward_distribution  =  self.reward_model(full_states)
        reward_loss          = -reward_distribution.log_prob(data.rewards[:, 1:].squeeze(-1)).mean()

        prior_mean        = torch.stack([dist.base_dist.loc for dist in prior_distributions], dim=1)
        prior_std         = torch.stack([dist.base_dist.scale for dist in prior_distributions], dim=1)
        posterior_mean    = torch.stack([dist.base_dist.loc for dist in posterior_distributions], dim=1)
        posterior_std     = torch.stack([dist.base_dist.scale for dist in posterior_distributions], dim=1)

        prior_distribution        = torch.distributions.Independent(Normal(prior_mean, prior_std), 1)
        prior_distribution_sg     = torch.distributions.Independent(Normal(prior_mean.detach(), prior_std.detach()), 1)
        posterior_distribution    = torch.distributions.Independent(Normal(posterior_mean, posterior_std), 1)
        posterior_distribution_sg = torch.distributions.Independent(Normal(posterior_mean.detach(), posterior_std.detach()), 1)

        prior_loss       = kl_divergence(posterior_distribution_sg, prior_distribution  )
        posterior_loss   = kl_divergence(posterior_distribution  , prior_distribution_sg)
        free_nats        = torch.full_like(prior_loss, self.config.free_nats)

        prior_loss       = self.config.beta_prior*torch.maximum(prior_loss, free_nats)
        posterior_loss   = self.config.beta_posterior*torch.maximum(posterior_loss, free_nats)
        kl_loss          = (prior_loss + posterior_loss).mean()

        world_model_loss =  reconstruction_loss + reward_loss + kl_loss # I think that the reconstruction loss is relatively a bit too high (11k) 

        if self.config.use_continuation_prediction:
            continue_distribution = self.continue_model(full_states)
            continue_targets      = 1 - data.dones[:, 1:].squeeze(-1)
            continue_loss         = -continue_distribution.log_prob(continue_targets).mean()
            world_model_loss     += continue_loss

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
        return full_states.reshape(-1, self.full_state_size).detach(), metrics
    

    def behavior_training(self, full_state):
        recurrent_state, latent_state = torch.split(full_state, (self.recurrent_size, self.latent_size), -1)
        full_states, logprobs, entropies = [], [], []
        for _ in range(self.config.imagination_horizon):
            action, logprob, entropy = self.actor(full_state.detach(), training=True)
            recurrent_state = self.recurrent_model(recurrent_state, latent_state, action)
            _, latent_state = self.prior(recurrent_state)

            full_state = torch.cat((recurrent_state, latent_state), -1)
            full_states.append(full_state)
            logprobs.append(logprob)
            entropies.append(entropy)
        full_states  = torch.stack(full_states,    dim=1) # (batchSize*batchLength, imaginationHorizon, recurrentSize + latentLength*latentClasses)
        logprobs     = torch.stack(logprobs[1:],   dim=1) # (batchSize*batchLength, imaginationHorizon-1)
        entropies    = torch.stack(entropies[1:],  dim=1) # (batchSize*batchLength, imaginationHorizon-1)
        
        predicted_rewards = self.reward_model(full_states[:, :-1]).mean
        values            = self.critic(full_states).mean
        continues         = self.continue_model(full_states[:, :-1]).mean if self.config.use_continuation_prediction else torch.full_like(predicted_rewards, self.config.discount)
        lambda_values     = computeLambdaValues(predicted_rewards, values, continues, self.config.lambda_)

        _, inverse_scale = self.valueMoments(lambda_values)
        advantages       = (lambda_values - values[:, :-1])/inverse_scale

        actor_loss = -torch.mean(advantages.detach()*logprobs + self.config.entropy_scale*entropies)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip, norm_type=self.config.gradient_norm_type)
        self.actor_optimizer.step()

        value_distributions  =  self.critic(full_states[:, :-1].detach())
        critic_loss          = -torch.mean(value_distributions.log_prob(lambda_values.detach()))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradient_clip, norm_type=self.config.gradient_norm_type)
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
    def environment_interaction(self, env, nb_episodes, seed=None, evaluation=False, save_video=False, filename=None):
        scores = []
        for episod in range(nb_episodes):
            recurrent_state = torch.zeros(1, self.recurrent_size, device=self.device)
            latent_state    = torch.zeros(1, self.latent_size,    device=self.device)
            action          = torch.zeros(1, self.action_size,    device=self.device)

            time_step = env.reset()
            observation = time_step.observation["pixels"]
            encoded_observation = self.encoder(self._observation_tensor(observation))

            current_score, step_count, done, frames = 0, 0, False, []
            while not done:
                recurrent_state = self.recurrent_model(recurrent_state, latent_state, action)
                _, latent_state = self.posterior(torch.cat((recurrent_state, encoded_observation.reshape(1, -1)), -1))

                action = self.actor(torch.cat((recurrent_state, latent_state), -1), deterministic=evaluation)
                action_numpy = action.cpu().numpy().reshape(-1)

                time_step = env.step(action_numpy)
                next_observation = time_step.observation["pixels"]
                reward = 0.0 if time_step.reward is None else time_step.reward
                done = time_step.last()

                if not evaluation:
                    self.buffer.add(observation, action_numpy, reward, next_observation, done)

                if save_video:
                    frames.append(np.ascontiguousarray(next_observation))

                encoded_observation = self.encoder(self._observation_tensor(next_observation))

                observation    = next_observation
                current_score += reward
                step_count    += 1

                if done:
                    scores.append(current_score)
                    if not evaluation:
                        self.total_episodes  += 1
                        self.total_env_steps += step_count
        
            if save_video and filename is not None and frames:
                self._save_video(frames, filename)

        return sum(scores)/nb_episodes if nb_episodes else None

    environement_interaction = environment_interaction

    def save_checkpoint(self, filename):
        if not filename.endswith(".pt"):
            filename += ".pt"
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)

        checkpoint = {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "recurrent_model": self.recurrent_model.state_dict(),
            "prior": self.prior.state_dict(),
            "posterior": self.posterior.state_dict(),
            "reward_model": self.reward_model.state_dict(),
            "continue_model": self.continue_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "world_model_optimizer": self.world_model_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_episodes": self.total_episodes,
            "total_env_steps": self.total_env_steps,
            "total_gradient_steps": self.total_gradient_steps,
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        if not filename.endswith(".pt"):
            filename += ".pt"
        checkpoint = torch.load(filename, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.recurrent_model.load_state_dict(checkpoint["recurrent_model"])
        self.prior.load_state_dict(checkpoint["prior"])
        self.posterior.load_state_dict(checkpoint["posterior"])
        self.reward_model.load_state_dict(checkpoint["reward_model"])
        self.continue_model.load_state_dict(checkpoint["continue_model"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.world_model_optimizer.load_state_dict(checkpoint["world_model_optimizer"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.total_episodes = checkpoint.get("total_episodes", 0)
        self.total_env_steps = checkpoint.get("total_env_steps", 0)
        self.total_gradient_steps = checkpoint.get("total_gradient_steps", 0)

    def _save_video(self, frames, filename):
        try:
            import imageio.v2 as imageio
        except ImportError:
            print("imageio is not installed; skipping video save.")
            return

        if not filename.endswith(".mp4"):
            filename += ".mp4"
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        imageio.mimsave(filename, frames, fps=30)
