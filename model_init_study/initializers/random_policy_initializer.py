from model_init_study.initializers.initializer import Initializer

import numpy as np

class RandomPolicyInitializer(Initializer):
    def __init__(self, params):
        ## Call super init
        super().__init__(params)
        ## Get the controller and initialize it from params
        self._controller = params['controller_type'](params)
        ## Get policy parameters init min and max
        self._policy_param_init_min = params['policy_param_init_min'] 
        self._policy_param_init_max = params['policy_param_init_max']
        ## Get size of policy parameter vector
        policy_representation_dim = len(self._controller.get_parameters())
        self.controllers = []
        ## Initialize nb_eval random policies
        for _ in range(self._n_init_episodes + self._n_test_episodes):
            ## Create a random policy parametrization 
            x = np.random.uniform(low=self._policy_param_init_min,
                                  high=self._policy_param_init_max,
                                  size=policy_representation_dim)
            ## Copy the controller type
            controller = self._controller.copy()
            ## Set controller parameters
            controller.set_parameters(x)
            ## Add new controller to controller list
            self.controllers.append(controller)

    def _get_action(self, idx, obs, t):
        return self.controllers[idx](obs)
