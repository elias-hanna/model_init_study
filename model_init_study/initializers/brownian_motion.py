from model_init_study.initializers.initializer import Initializer

import numpy as np

class BrownianMotion(FormalizedInitializer):
    def __init__(self, params):
        ## Call super init
        super().__init__(params)
        
    def _alpha(self, i, t):
        return 1

    def get_alpha(self, i, t):
        return self._alpha

    def get_z(self, i, t):
        return np.random.uniform(low=-self.step_size,
                                 high=self.step_size,
                                 size=self._max_env_h)




## Below is sample code from iCEM

# def sample_action_sequences(self, obs, num_traj, time_slice=None):
#         """
#         :param num_traj: number of trajectories
#         :param obs: current observation
#         :type time_slice: slice
#         """
#         # colored noise
#         if self.noise_beta > 0:
#             assert (self.mean.ndim == 2)
#             # Important improvement
#             # self.mean has shape h,d: we need to swap d and h because temporal correlations are in last axis)
#             # noinspection PyUnresolvedReferences
#             samples = colorednoise.powerlaw_psd_gaussian(self.noise_beta, size=(num_traj, self.mean.shape[1],
#                                                                                 self.mean.shape[0])).transpose(
#                 [0, 2, 1])
#         else:
#             samples = np.random.randn(num_traj, *self.mean.shape)

#         samples = np.clip(samples * self.std + self.mean, self.env.action_space.low, self.env.action_space.high)
#         if time_slice is not None:
#             samples = samples[:, time_slice]
#         return samples
