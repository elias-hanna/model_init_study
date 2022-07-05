## Local imports
from model_init_study.initializers.formalized_initializer import FormalizedInitializer

## Utils imports
import numpy as np
from scipy.stats import levy
import random

class LevyFlight(FormalizedInitializer):
    def __init__(self, params):
        ## Call super init
        self.c = 1/96
        super().__init__(params)
        
    def _alpha(self, i, t):
        return 1

    def get_z(self):
        levy_samples = levy.rvs(scale=self.c, size=(self._env_max_h, self._act_dim))
        levy_samples = [each*random.choice([-1,1]) for each in levy_samples]
        return levy_samples
