from model_init_study.initializers.initializer import Initializer

import numpy as np

import colorednoise

class ColoredNoiseMotion(Initializer):
    def __init__(self, params):
        ## Call super init
        super().__init__(params)
        self.noise_beta = params['noise_beta']

        samples = sample_action_sequences()

        self.actions = []
        for i in range(samples):
            self.actions.append(samples[i])
        
    def sample_action_sequences(self):
        """
        :param num_traj: number of trajectories
        :param obs: current observation
        :type time_slice: slice
        """
        # colored noise
        if self.noise_beta > 0:
            # assert (self.mean.ndim == 2)
            # Important improvement
            # self.mean has shape h,d: we need to swap d and h because temporal correlations are in last axis)
            # noinspection PyUnresolvedReferences
            samples = colorednoise.powerlaw_psd_gaussian(self.noise_beta,
                                                         size=(self._n_init_episodes+
                                                               self._n_test_episodes,
                                                               self._env_max_h,
                                                               self._act_dim)).transpose(
                                                                   [0, 2, 1])
        else:
            samples = np.random.randn(num_traj, self._env_max_h, self._act_dim)
        
        samples = np.clip(samples + self._action_init, self.env.action_space.low, self.env.action_space.high)
        return samples
