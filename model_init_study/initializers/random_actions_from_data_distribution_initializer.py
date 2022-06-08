from model_init_study.initializers.initializer import Initializer

import numpy as np

class RandomActionsFromDataDistributionInitializer(Initializer):
    def __init__(self, params):
        ## Call super init
        super().__init__(params)

        if 'init_distribution_path' not in params:
            raise ValueError('RandomActionsFromDataDistributionInitializer: init_distribution_path not in params')
        init_data_distribution = np.load(params['init_distribution_path'])
        ## Initialize the random action policies
        self.actions = []
        self._action_lasting_steps = params['action_lasting_steps']
        for _ in range(self._n_init_episodes + self._n_test_episodes):
            loc_actions = []
            for t in range(self._env_max_h):
                if t%self._action_lasting_steps == 0:
                    action_ind = np.random.choice(init_data_distribution.shape[0])
                    action = init_data_distribution[action_ind]

                loc_actions.append(action)
            self.actions.append(loc_actions)
            
    def _get_action(self, idx, obs, t):
        return self.actions[idx][t]
