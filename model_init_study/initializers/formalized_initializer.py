## Abstract layer imports
from abc import abstractmethod

## Local imports
from model_init_study.initializers.initializer import Initializer

## Utils imports
import numpy as np

class FormalizedInitializer(Initializer):
    def __init__(self, params):
        ## Call super init
        super().__init__(params)

        self.step_size = params['step_size']

        ## /!\ alpha is a matrix ##
        self.alpha = self.get_alpha() ## alphas are immutable over iterations

        ## Initialize the random action policies
        self.actions = []
        for _ in range(self._n_init_episodes + self._n_test_episodes):
            ## /!\ z is a tab ##
            self.z = self.get_z() ## zs are mutable over iterations
            self.z = np.clip(self.z, params['action_min'], params['action_max'])
            loc_actions = []
            for t in range(self._env_max_h):
                loc_alpha_z = [self._action_init + self.alpha[i,t]*self.z[i]
                               for i in range(self._env_max_h)]
                action = sum(loc_alpha_z[:t+1])
                loc_actions.append(action)
            loc_actions = np.clip(loc_actions, params['action_min'], params['action_max'])
            self.actions.append(loc_actions)

    def _get_action(self, idx, obs, t):
        return self.actions[idx][t]

    def get_actions(self):
        return self.actions

    @abstractmethod
    def get_z(self):
        raise NotImplementedError

    @abstractmethod
    def _alpha(self, i, t):
        raise NotImplementedError
    
    def get_alpha(self):
        ## Alpha is a square matrix of all possibles value combination for a i and a t
        alpha = np.zeros((self._env_max_h, self._env_max_h))
        for i in range(self._env_max_h):
            for t in range(self._env_max_h):
                alpha[i, t] = self._alpha(i,t)
        return alpha


## Code test and plot for random walks
if __name__ == '__main__':
    from brownian_motion import BrownianMotion
    from levy_flight import LevyFlight
    from colored_noise_motion import ColoredNoiseMotion
    
    import matplotlib.pyplot as plt
    import mb_ge
    import gym
    env = gym.make('BallInCup3d-v0')


    ## Dummy vals for testing
    obs_dim = 1
    act_dim = 1

    ss_min = -0.4
    ss_max = 0.4

    max_step = 300
    
    params = \
    {
        'obs_dim': obs_dim,
        'action_dim': act_dim,

        'n_init_episodes': 1,
        # 'n_test_episodes': int(.2*args.init_episodes), # 20% of n_init_episodes
        'n_test_episodes': 1,
        
        # 'controller_type': NeuralNetworkController,
        # 'controller_params': controller_params,

        # 'dynamics_model_params': dynamics_model_params,

        'action_min': -1,
        'action_max': 1,
        'action_init': 0, 
        'action_lasting_steps': 5,

        'state_min': ss_min,
        'state_max': ss_max,

        'step_size': 0.1,
        'noise_beta': 0,
        
        'policy_param_init_min': -5,
        'policy_param_init_max': 5,
        
        # 'dump_path': 'default_dump/',
        # 'path_to_test_trajectories': 'examples/'+args.environment+'_example_trajectories.npz',
        # 'path_to_test_trajectories': path_to_examples,

        'env': env,
        'env_max_h': max_step,
    }

    ## Brownian Motion
    bm = BrownianMotion(params)

    plt.figure()

    plt.plot(range(max_step), bm.actions[0])

    plt.title('Action value across time for Brownian Motion')

    ## Levy Flight
    lf = LevyFlight(params)

    plt.figure()

    plt.plot(range(max_step), lf.actions[0])

    plt.title('Action value across time for Levy Flight')

    ## Colored Noise motion
    cnm = ColoredNoiseMotion(params)

    plt.figure()

    plt.plot(range(max_step), cnm.actions[0])

    plt.title('Action value across time for Colored Noise Motion')

    plt.show()
