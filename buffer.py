import numpy as np 
import torch

from utils import AttrDict


class ReplayBuffer(object):
    def __init__(self, observation_shape, action_size, config, device):
        self.config = config
        self.device = device
        self.capacity = int(self.config.capacity)
        self.action_size = int(action_size)

        self.observations        = np.empty((self.capacity, *observation_shape), dtype=np.float32)
        self.next_observations   = np.empty((self.capacity, *observation_shape), dtype=np.float32)
        self.actions             = np.empty((self.capacity, self.action_size),   dtype=np.float32)
        self.rewards             = np.empty((self.capacity, 1),                  dtype=np.float32)
        self.dones               = np.empty((self.capacity, 1),                  dtype=np.float32)

        self.buffer_index = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.buffer_index

    def add(self, observation, action, reward, next_observation, done):
        self.observations[self.buffer_index]      = observation
        self.next_observations[self.buffer_index] = next_observation
        self.actions[self.buffer_index]           = np.asarray(action, dtype=np.float32).reshape(self.action_size)
        self.rewards[self.buffer_index]           = reward
        self.dones[self.buffer_index]             = done

        self.buffer_index = (self.buffer_index + 1) % self.capacity
        self.full = self.full or self.buffer_index == 0

    def sample(self, batch_size, batch_length):
        current_size = len(self)
        if current_size < batch_length:
            raise ValueError(f"Replay buffer has {current_size} transitions, but batch_length={batch_length}.")

        chronological_indices = np.arange(current_size)
        if self.full:
            chronological_indices = (chronological_indices + self.buffer_index) % self.capacity

        max_start_index = current_size - batch_length + 1
        replace = max_start_index < batch_size
        start_indices = np.random.choice(max_start_index, batch_size, replace=replace)

        sample_indices = np.empty((batch_size, batch_length), dtype=np.int64)
        for i in range(batch_size):
            sample_indices[i] = chronological_indices[start_indices[i] + np.arange(batch_length)]

        observations =      torch.as_tensor(self.observations[sample_indices],      device=self.device)
        next_observations = torch.as_tensor(self.next_observations[sample_indices], device=self.device)
        actions =           torch.as_tensor(self.actions[sample_indices],           device=self.device)
        rewards =           torch.as_tensor(self.rewards[sample_indices],           device=self.device)
        dones =             torch.as_tensor(self.dones[sample_indices],             device=self.device)

        sample = AttrDict({
            "observations"       : observations,
            "next_observations"  : next_observations,
            "actions"            : actions,
            "rewards"            : rewards,
            "dones"              : dones
        })

        return sample 
