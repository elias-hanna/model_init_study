from model_init_study.initializer import Initializer

class RandomActionsInitializer(Initializer):
    def __init__(self, params):
        ## Initialize the random policies
        pass

    def _get_action(self, idx, obs, t):
        return actions[idx][t]
