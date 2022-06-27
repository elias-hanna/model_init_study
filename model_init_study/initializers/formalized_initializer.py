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

        ## /!\ alpha is a functor ##
        self.alpha = self.get_alpha() ## alphas are immutable over iterations

        ## Initialize the random action policies
        self.actions = []
        for _ in range(self._n_init_episodes + self._n_test_episodes):
            ## /!\ z is a tab ##
            self.z = self.get_z() ## zs are mutable over iterations
            loc_actions = []
            loc_alpha_z = [self.alpha(i,t)*self.z(i,t) for i in range(self._env_max_h)]
            for t in range(self._env_max_h):
                action = sum(loc_alpha_z[:t])
                loc_actions.append(action)
            loc_actions = np.clip(loc_actions, params['action_min'], params['action_max'])
            self.actions.append(loc_actions)

    def _get_action(self, idx, obs, t):
        return self.actions[idx][t]

    @abstractmethod
    def get_z(self, i, t):
        raise NotImplementedError

    @abstractmethod
    def get_alpha(self, i, t):
        raise NotImplementedError


## Code test and plot for random walks
if __name__ == '__main__':
    
