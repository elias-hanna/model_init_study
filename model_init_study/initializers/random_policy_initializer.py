from model_init_study.initializer import Initializer

class RandomPolicyInitializer(Initializer):
    def __init__(self, params):
        ## Initialize the random policies
        pass

    def _get_action(self, idx, obs, t):
        return controller[idx](obs)
