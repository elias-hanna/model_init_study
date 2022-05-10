from model_init_study.initializers.initializer import Initializer

import numpy as np

class RandomActionsInitializer(Initializer):
    def __init__(self, params):
        ## Call super init
        super().__init__(params)
        ## Initialize the random action policies
        self.actions = []
        self._action_lasting_steps = params['action_lasting_steps']
        for _ in range(self._n_init_episodes + self._n_test_episodes):
            loc_actions = []
            for t in range(self._env_max_h):
                if t%self._action_lasting_steps == 0:
                    action = np.random.uniform(low=self._action_min,
                                               high=self._action_max,
                                               size=self._act_dim)
                loc_actions.append(action)
            self.actions.append(loc_actions)
            
    def _get_action(self, idx, obs, t):
        return self.actions[idx][t]
