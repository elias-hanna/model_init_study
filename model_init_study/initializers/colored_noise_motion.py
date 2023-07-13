## Local Imports 
from model_init_study.initializers.formalized_initializer import FormalizedInitializer

## Utils imports
import numpy as np
import colorednoise

class ColoredNoiseMotion(FormalizedInitializer):
    def __init__(self, params):
        self.noise_beta = params['noise_beta']
        ## Call super init
        super().__init__(params)
        
    def _alpha(self, i, t):
        return 1

    def get_z(self):
        """
        :param num_traj: number of trajectories
        :param obs: current observation
        :type time_slice: slice
        """
        ## Note, step size used as stddev
        sigma = self.step_size
        var = sigma**2
        # colored noise
        # if self.noise_beta > 0:
            # assert (self.mean.ndim == 2)
            # Important improvement
            # self.mean has shape h,d: we need to swap d and h because temporal correlations are in last axis)
            # noinspection PyUnresolvedReferences
        samples = sigma * colorednoise.powerlaw_psd_gaussian(self.noise_beta, #10)
                                                             size=(self._act_dim,
                                                                   self._env_max_h),
                                                             random_state=None).transpose(
                                                                 [1, 0])
        # else:
            # samples = sigma * np.random.randn(self._env_max_h, self._act_dim)

        # samples = samples*0.5
        # samples = np.clip(samples + self._action_init, self._env.action_space.low, self._env.action_space.high)
        return samples
