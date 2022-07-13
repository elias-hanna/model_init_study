## Local imports
from model_init_study.initializers.initializer import Initializer

## Utils imports
import numpy as np

class RARPHybridInitializer(Initializer):
    def __init__(self, params):
        ## Call super init
        super().__init__(params)
        
        ## Random actions part
        self.actions = []
        self._action_lasting_steps = params['action_lasting_steps']

        ## Random policies part
        self._controller = params['controller_type'](params)
        ## Get policy parameters init min and max
        self._policy_param_init_min = params['policy_param_init_min'] 
        self._policy_param_init_max = params['policy_param_init_max']
        ## Get size of policy parameter vector
        policy_representation_dim = len(self._controller.get_parameters())
        self.controllers = []
        
        for _ in range((self._n_init_episodes + self._n_test_episodes)//2):
            loc_actions = []
            for t in range(self._env_max_h):
                if t%self._action_lasting_steps == 0:
                    action = np.random.uniform(low=self._action_min,
                                               high=self._action_max,
                                               size=self._act_dim)
                loc_actions.append(action)
            self.actions.append(loc_actions)
        
        for _ in range((self._n_init_episodes + self._n_test_episodes)//2 + (self._n_init_episodes + self._n_test_episodes)%2):
            ## Create a random policy parametrization 
            x = np.random.uniform(low=self._policy_param_init_min,
                                  high=self._policy_param_init_max,
                                  size=policy_representation_dim)
            ## Copy the controller type
            controller = self._controller.copy()
            ## Set controller parameters
            controller.set_parameters(x)
            ## Add new controller to controller list
            # self.controllers.append(controller)
            self.actions.append(controller)
            
    def _get_action(self, idx, obs, t):
        if idx < (self._n_init_episodes + self._n_test_episodes)//2:
            return self.actions[idx][t]
        else:
            # return self.controllers[idx](obs)
            return self.actions[idx](obs)
