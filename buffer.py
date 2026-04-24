import numpy as np 
import attridict
import torch


class ReplayBuffer(object):
    def __init__(self, observation_shape, action_size, config, device):
        self.config = config
        self.device = device
        self.capacity = int(self.config.capacity)

        self.observations        = np.empty((self.capacity, *observation_shape), dtype=np.float32)
        self.next_observations   = np.empty((self.capacity, *observation_shape), dtype=np.float32)
        self.actions             = np.empty((self.capacity, action_size),        dtype=np.float32)
        self.rewards             = np.empty((self.capacity, 1),                  dtype=np.float32)
        self.dones               = np.empty((self.capacity, 1),                  dtype=np.float32)

        self.buffer_index = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.buffer_index

    def add(self, observation, action, reward, next_observation, done):
        self.observations[self.buffer_index]      = observation
        self.next_observations[self.buffer_index] = next_observation
        self.actions[self.buffer_index]           = action
        self.rewards[self.buffer_index]           = reward
        self.dones[self.buffer_index]             = done

        self.buffer_index = (self.buffer_index + 1) % self.capacity
        self.full = self.full or self.buffer_index == 0

    def sample(self, batch_size, batch_lenght):

        max_start_indice = self.buffer_index - batch_lenght + 1
        start_indices = np.random.choice(max_start_indice, batch_size, replace=False)

        sample_indices = np.empty((batch_size, batch_lenght), dtype=np.int64)
        for i in range(batch_size):
            sample_indices[i] = start_indices[i] + np.arange(batch_lenght)

        observations =      torch.as_tensor(self.observations[sample_indices])
        next_observations = torch.as_tensor(self.next_observations[sample_indices])
        actions =           torch.as_tensor(self.actions[sample_indices])
        rewards =           torch.as_tensor(self.rewards[sample_indices])
        dones =             torch.as_tensor(self.dones[sample_indices])

        sample = attridict({
            "observations"       : observations,
            "next_observations"  : next_observations,
            "actions"            : actions,
            "rewards"            : rewards,
            "dones"              : dones
        })

        return sample 