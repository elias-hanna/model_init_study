## local imports
from model_init_study.initializers.formalized_initializer import FormalizedInitializer

## Utils imports
import numpy as np

class BrownianMotion(FormalizedInitializer):
    def __init__(self, params):
        ## Call super init
        super().__init__(params)
        
    def _alpha(self, i, t):
        return 1

    def get_z(self):
        return np.random.uniform(low=-self.step_size,
                                 high=self.step_size,
                                 size=self._env_max_h)
