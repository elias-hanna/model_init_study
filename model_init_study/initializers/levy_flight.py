from model_init_study.initializers.initializer import Initializer

import numpy as np

from scipy.stats import levy

class BrownianMotion(FormalizedInitializer):
    def __init__(self, params):
        ## Call super init
        super().__init__(params)
        
    def _alpha(self, i, t):
        return 1

    def get_alpha(self, i, t):
        return self._alpha

    def get_z(self, i, t):
        return levy.rvs(size=self._max_env_h)
