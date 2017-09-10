import numpy as np


class ReplayBuffer:
    '''
    Memory efficient implementation of replay buffer, storing each state only once.
    Example: Typical use for atari, with each frame being a 84x84 grayscale
             image (uint8), storing 1M frames should use about 7GiB of RAM
             (8 * 64 * 64 * 1M bits)

    Args:
        maxlen: Maximum number of transitions stored
        history_length: Number of sequential states stacked when sampling
        batch_size: Mini-batch size created by sample
    '''
    def __init__(self, maxlen, history_length=1, batch_size=32):
        self.initialized = False
        self.maxlen = maxlen
        self.history_length = history_length
        self.batch_size = batch_size
        self.current_idx = 0
        self.current_len = 0

    def add(self, state, action, reward, done):
        if not self.initialized:
            self.initialized = True
            state_shape = np.squeeze(state).shape
            # Allocate memory
            self.states = np.empty((self.maxlen,) + state_shape,
                                   dtype=state.dtype)
            self.actions = np.empty(self.maxlen, dtype=np.int32)
            self.rewards = np.empty(self.maxlen, dtype=np.float32)
            self.dones = np.empty(self.maxlen, dtype=np.bool)

        # Store transition
        self.states[self.current_idx] = np.squeeze(state)
        self.actions[self.current_idx] = action
        self.rewards[self.current_idx] = reward
        self.dones[self.current_idx] = done

        # Update current position
        self.current_idx = (self.current_idx + 1) % self.maxlen
        self.current_len = min(self.current_len + 1, self.maxlen)

    def sample(self):
        start_idxs, end_idxs = self._generate_idxs()
        # Get states
        b_states_t = np.array([self.states[start_idx:end_idx] for
                              start_idx, end_idx in zip(start_idxs, end_idxs)],
                              copy=False)
        b_states_tp1 = np.array([self.states[start_idx + 1: end_idx + 1] for
                                start_idx, end_idx in zip(start_idxs, end_idxs)],
                                copy=False)
        # Remember that when slicing the end_idx is not included
        actions = self.actions[end_idxs - 1]
        rewards = self.rewards[end_idxs - 1]
        dones = self.dones[end_idxs - 1]

        return (b_states_t.swapaxes(1, -1),
                b_states_tp1.swapaxes(1, -1),
                actions, rewards, dones)

    def _generate_idxs(self):
        start_idxs = []
        end_idxs = []
        while len(start_idxs) < self.batch_size:
            start_idx = np.random.randint(0, self.current_len - self.history_length)
            end_idx = start_idx + self.history_length

            # Check if idx was already picked
            if start_idx in start_idxs:
                continue
            # Only the last frame can have done == True
            if np.any(self.dones[start_idx: end_idx - 1]):
                continue

            # Valid idx!!
            start_idxs.append(start_idx)
            end_idxs.append(end_idx)

        return np.array(start_idxs), np.array(end_idxs)
