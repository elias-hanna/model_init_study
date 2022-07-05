## Abstract layer imports
from abc import abstractmethod

## Multiprocessing imports
from multiprocessing import cpu_count
from multiprocessing import Pool

## Local imports
from model_init_study.initializers.initializer import Initializer

## Utils imports
import numpy as np

class FormalizedInitializer(Initializer):
    def __init__(self, params):
        ## Call super init
        super().__init__(params)

        self.step_size = params['step_size']

        ## /!\ alpha is a matrix ##
        self.alpha = self.get_alpha() ## alphas are immutable over iterations

        # ## Sequential gen
        # ## Initialize the random action policies
        # self.actions = []
        # for _ in range(self._n_init_episodes + self._n_test_episodes):
        #     ## /!\ z is a tab ##
        #     self.z = self.get_z() ## zs are mutable over iterations
        #     self.z = np.clip(self.z, self._action_min, self._action_max)
        #     loc_actions = []
        #     for t in range(self._env_max_h):
        #         loc_alpha_z = [self._action_init + self.alpha[i,t]*self.z[i]
        #                        for i in range(self._env_max_h)]
        #         action = sum(loc_alpha_z[:t+1])
        #         loc_actions.append(action)
        #     loc_actions = np.clip(loc_actions, self._action_min, self._action_max)
        #     self.actions.append(loc_actions)

        ## Parallel gen
        pool = Pool(processes=self.nb_thread)
        self.actions = pool.starmap(self._gen_action_sequence,
                                    zip(range(self._n_init_episodes + self._n_test_episodes)))
        
    def _gen_action_sequence(self, idx):
        ## /!\ z is a tab ##
        self.z = self.get_z() ## zs are mutable over iterations
        self.z = np.clip(self.z, self._action_min, self._action_max)
        loc_actions = []
        for t in range(self._env_max_h):
            loc_alpha_z = [self._action_init + self.alpha[i,t]*self.z[i]
                           for i in range(self._env_max_h)]
            action = sum(loc_alpha_z[:t+1])
            loc_actions.append(action)
        loc_actions = np.clip(loc_actions, self._action_min, self._action_max)
        return loc_actions

    def _get_action(self, idx, obs, t):
        return self.actions[idx][t]

    def get_actions(self):
        return self.actions

    @abstractmethod
    def get_z(self):
        raise NotImplementedError

    @abstractmethod
    def _alpha(self, i, t):
        raise NotImplementedError
    
    def get_alpha(self):
        ## Alpha is a square matrix of all possibles value combination for a i and a t
        alpha = np.zeros((self._env_max_h, self._env_max_h))
        for i in range(self._env_max_h):
            for t in range(self._env_max_h):
                alpha[i, t] = self._alpha(i,t)
        return alpha


## Code test and plot for random walks
if __name__ == '__main__':
    from brownian_motion import BrownianMotion
    from levy_flight import LevyFlight
    from colored_noise_motion import ColoredNoiseMotion
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.tsa.stattools import acf

    import matplotlib.pyplot as plt
    import mb_ge
    import gym

    env = gym.make('BallInCup3d-v0')

    ## Dummy vals for testing
    obs_dim = 1
    act_dim = 1

    ss_min = -0.4
    ss_max = 0.4

    max_step = 300

    nreps = 1000
    nlags = round(min(10*np.log10(max_step), max_step - 1))
    
    params = \
    {
        'obs_dim': obs_dim,
        'action_dim': act_dim,

        'n_init_episodes': nreps,
        # 'n_test_episodes': int(.2*args.init_episodes), # 20% of n_init_episodes
        'n_test_episodes': 0,
        
        # 'controller_type': NeuralNetworkController,
        # 'controller_params': controller_params,

        # 'dynamics_model_params': dynamics_model_params,

        'action_min': -1,
        'action_max': 1,
        'action_init': 0, 
        'action_lasting_steps': 5,

        'state_min': ss_min,
        'state_max': ss_max,

        'step_size': 0.1,
        'noise_beta': 2,
        
        'policy_param_init_min': -5,
        'policy_param_init_max': 5,
        
        # 'dump_path': 'default_dump/',
        # 'path_to_test_trajectories': 'examples/'+args.environment+'_example_trajectories.npz',
        # 'path_to_test_trajectories': path_to_examples,

        'env': env,
        'env_max_h': max_step,
    }

    ###### Simple test of each random walk ######
    
    ## Brownian Motion
    bm = BrownianMotion(params)

    plt.figure()

    plt.plot(range(max_step), bm.actions[0])

    plt.title('Action value across time for Brownian Motion')

    ## Levy Flight
    lf = LevyFlight(params)

    plt.figure()

    plt.plot(range(max_step), lf.actions[0])

    plt.title('Action value across time for Levy Flight')

    # ## Colored Noise motion
    # cnm = ColoredNoiseMotion(params)

    # plt.figure()

    # plt.plot(range(max_step), cnm.actions[0])

    # plt.title('Action value across time for Colored Noise Motion')

    # ## Plot correlogram for each random walk
    
    # plot_acf(bm.actions[0], title='Correlogram for Brownian Motion')
    # plot_acf(cnm.actions[0], title='Correlogram for Colored Noise Motion')
    # plot_acf(lf.actions[0], title='Correlogram for Levy Flight')

    # plt.show()

    ###### Plot mean autocorrelation over a larger number of sampled action sequences ######
    
    # bm_acf_res = np.empty((nreps, nlags))
    # lf_acf_res = np.empty((nreps, nlags))
    # cnm_acf_res = np.empty((nreps, nlags))
    
    # bm = BrownianMotion(params)
    # lf = LevyFlight(params)
    # # cnm = ColoredNoiseMotion(params)

    # for i in range(nreps):

    #     bm_acf_res[i] = acf(bm.actions[i], nlags=nlags)
    #     lf_acf_res[i] = acf(lf.actions[i], nlags=nlags)
    #     # cnm_acf_res[i] = acf(cnm.actions[i], nlags=nlags)

    # plt.figure()

    # plt.plot(range(1, nlags+1), np.nanmean(bm_acf_res, axis=0))

    # plt.title('Correlogram for Brownian Motion')

    # plt.figure()

    # plt.plot(range(1, nlags+1), np.nanmean(lf_acf_res, axis=0))

    # plt.title('Correlogram for Levy Flight')

    # # plt.figure()

    # # plt.plot(range(1, nlags+1), np.nanmean(cnm_acf_res, axis=0))

    # # plt.title('Correlogram for Colored Noise Motion')

    # plt.show()

    ###### Plot mean autocorrelation over a larger number of sampled noise sequences ######

    params['n_init_episodes'] = 1
    params['action_dim'] = nreps
    
    bm_acf_res = np.empty((nreps, nlags+1))
    lf_acf_res = np.empty((nreps, nlags+1))
    cnm_acf_res = np.empty((nreps, nlags+1))
    
    bm = BrownianMotion(params)
    lf = LevyFlight(params)
    cnm = ColoredNoiseMotion(params)

    bm_noise = bm.get_z()
    lf_noise = lf.get_z()
    cnm_noise = cnm.get_z()

    import pdb; pdb.set_trace()
    for i in range(nreps):

        bm_acf_res[i] = acf(bm_noise[i], nlags=nlags)
        lf_acf_res[i] = acf(lf_noise[i], nlags=nlags)
        cnm_acf_res[i] = acf(cnm_noise[i], nlags=nlags)
        
    plt.figure()

    plt.plot(range(1, nlags+1), np.nanmean(bm_acf_res, axis=0))

    plt.title('Correlogram for White Noise')

    plt.figure()

    plt.plot(range(1, nlags+1), np.nanmean(lf_acf_res, axis=0))

    plt.title('Correlogram for Levy Distribution Noise')

    plt.figure()

    plt.plot(range(1, nlags+1), np.nanmean(cnm_acf_res, axis=0))

    plt.title('Correlogram for Colored Noise')

    plt.show()
