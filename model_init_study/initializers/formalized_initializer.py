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
            actions = [self._action_init]
            for i in range(t):
                actions.append(actions[-1]+loc_alpha_z[i][0])
                if actions[-1] > self._action_max:
                    actions[-1] = self._action_max
                elif actions[-1] < self._action_min:
                    actions[-1] = self._action_min

            action = actions[-1]
            # action = sum(loc_alpha_z[:t+1])
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
    from random_actions_initializer import RandomActionsInitializer
    from colored_noise_motion import ColoredNoiseMotion
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.tsa.stattools import acf

    import matplotlib.pyplot as plt
    from matplotlib import cm
    import matplotlib
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
    # nlags = max_step - 1
    step_size = .1
    sigma_cnrw = .005
    
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

        'step_size': step_size, # for URW and CNRW
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

    cmap = plt.cm.get_cmap('hsv', 4+1)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=4+1)

    colors = cm.ScalarMappable(norm=norm, cmap=cmap)

    ## Linestyles for plots

    linestyles = ['-', '--', ':', '-.']

    fig_all, ax_all = plt.subplots()

    fig, axs = plt.subplots(2, 2)

    # ## Uniform Random Walk
    # urw = BrownianMotion(params)

    # axs[0,0].plot(range(max_step), urw.actions[0])

    # axs[0,0].set_title('Action value across time for Uniform Random Walk')

    ## Random Actions
    ra = RandomActionsInitializer(params)

    axs[0,0].plot(range(max_step), ra.actions[0], color='black')
    ax_all.plot(range(max_step), ra.actions[0], linestyle=linestyles[0],
                color=colors.to_rgba(0), label='Random Actions', linewidth=2)

    axs[0,0].set_title('Action value across time for Random Actions')

    axs[0,0].set_xlim(0, max_step)
    axs[0,0].set_xlabel('Timestep', fontsize=40)
    axs[0,0].set_ylabel('Action Value', fontsize=40)

    ## Colored Noise motion beta = 0 (Brownian Motion)
    params['noise_beta'] = 0
    cnm_0 = ColoredNoiseMotion(params)

    axs[0,1].plot(range(max_step), cnm_0.actions[0], color='black')
    ax_all.plot(range(max_step), cnm_0.actions[0], linestyle=linestyles[1],
                color=colors.to_rgba(1), label='CNRW_0', linewidth=2)

    axs[0,1].set_title('Action value across time for Colored Noise Motion ' \
                       'with beta = 0 (Brownian Motion)')

    axs[0,1].set_xlim(0, max_step)
    axs[0,1].set_xlabel('Timestep', fontsize=40)
    axs[0,1].set_ylabel('Action Value', fontsize=40)

    ## Colored Noise motion beta = 1
    params['noise_beta'] = 1
    cnm_1 = ColoredNoiseMotion(params)

    axs[1,0].plot(range(max_step), cnm_1.actions[0], color='black')
    ax_all.plot(range(max_step), cnm_1.actions[0], linestyle=linestyles[2],
                color=colors.to_rgba(2), label='CNRW_1', linewidth=2)

    axs[1,0].set_title('Action value across time for Colored Noise Motion ' \
                       'with beta = 1')

    axs[1,0].set_xlim(0, max_step)
    axs[1,0].set_xlabel('Timestep', fontsize=40)
    axs[1,0].set_ylabel('Action Value', fontsize=40)

    ## Colored Noise motion beta = 2
    params['noise_beta'] = 2
    cnm_2 = ColoredNoiseMotion(params)

    axs[1,1].plot(range(max_step), cnm_2.actions[0], color='black')
    ax_all.plot(range(max_step), cnm_2.actions[0], linestyle=linestyles[3],
                color=colors.to_rgba(3), label='CNRW_2', linewidth=2)

    axs[1,1].set_title('Action value across time for Colored Noise Motion ' \
                       'with beta = 2')

    axs[1,1].set_xlim(0, max_step)
    axs[1,1].set_xlabel('Timestep', fontsize=40)
    axs[1,1].set_ylabel('Action Value', fontsize=40)

    plt.suptitle('Examples of RW in action space for different noise sequence generators')

    ax_all.tick_params(axis='x', labelsize=25)
    ax_all.tick_params(axis='y', labelsize=25)
    ax_all.set_xlim(0, max_step)
    ax_all.set_xlabel('Timestep', fontsize=40)
    ax_all.set_ylabel('Action Value', fontsize=40)
    ax_all.legend(prop = { "size": 25 })

    ###### Plot mean autocorrelation over a larger number of sampled action sequences ######
    fig_all, ax_all = plt.subplots()
    fig, axs = plt.subplots(2, 2)

    # urw_acf_res = np.empty((nreps, nlags+1))
    ra_acf_res = np.empty((nreps, nlags+1))
    cnm_0_acf_res = np.empty((nreps, nlags+1))
    cnm_1_acf_res = np.empty((nreps, nlags+1))
    cnm_2_acf_res = np.empty((nreps, nlags+1))

    # params['step_size'] = step_size
    # urw = BrownianMotion(params)
    ra = RandomActionsInitializer(params)
    params['step_size'] = sigma_cnrw
    params['noise_beta'] = 0
    cnm_0 = ColoredNoiseMotion(params)
    params['noise_beta'] = 1
    cnm_1 = ColoredNoiseMotion(params)
    params['noise_beta'] = 2
    cnm_2 = ColoredNoiseMotion(params)
    
    for i in range(nreps):
        # urw_acf_res[i] = acf(urw.actions[i], nlags=nlags)
        ra_acf_res[i] = acf(ra.actions[i], nlags=nlags)
        cnm_0_acf_res[i] = acf(cnm_0.actions[i], nlags=nlags)
        cnm_1_acf_res[i] = acf(cnm_1.actions[i], nlags=nlags)
        cnm_2_acf_res[i] = acf(cnm_2.actions[i], nlags=nlags)

    # axs[0,0].plot(range(nlags+1), np.nanmean(urw_acf_res, axis=0))

    # axs[0,0].set_title('Correlogram for Uniform Random Walk')

    axs[0,0].plot(range(nlags+1), np.nanmean(ra_acf_res, axis=0), color='black')
    ax_all.plot(range(nlags+1), np.nanmean(ra_acf_res, axis=0), linestyle=linestyles[0],
                color=colors.to_rgba(0), label='Random Actions', linewidth=2)

    axs[0,0].set_title('Correlogram for Random Actions')

    axs[0,0].set_xlim(0, nlags)
    axs[0,0].set_xlabel('Number of lags')
    axs[0,0].set_ylabel('Correlation value')

    axs[0,1].plot(range(nlags+1), np.nanmean(cnm_0_acf_res, axis=0), color='black')
    ax_all.plot(range(nlags+1), np.nanmean(cnm_0_acf_res, axis=0), linestyle=linestyles[1],
                color=colors.to_rgba(1), label='CNRW_0', linewidth=2)

    axs[0,1].set_title('Correlogram for Colored Noise Motion with beta = 0 (Brownian Motion)')

    axs[0,1].set_xlim(0, nlags)
    axs[0,1].set_xlabel('Number of lags')
    axs[0,1].set_ylabel('Correlation value')

    axs[1,0].plot(range(nlags+1), np.nanmean(cnm_1_acf_res, axis=0), color='black')
    ax_all.plot(range(nlags+1), np.nanmean(cnm_1_acf_res, axis=0), linestyle=linestyles[2],
                color=colors.to_rgba(2), label='CNRW_1', linewidth=2)

    axs[1,0].set_title('Correlogram for Colored Noise Motion with beta = 1')

    axs[1,0].set_xlim(0, nlags)
    axs[1,0].set_xlabel('Number of lags')
    axs[1,0].set_ylabel('Correlation value')

    axs[1,1].plot(range(nlags+1), np.nanmean(cnm_2_acf_res, axis=0), color='black')
    ax_all.plot(range(nlags+1), np.nanmean(cnm_2_acf_res, axis=0), linestyle=linestyles[3],
                color=colors.to_rgba(3), label='CNRW_2', linewidth=2)

    axs[1,1].set_title('Correlogram for Colored Noise Motion with beta = 2')

    axs[1,1].set_xlim(0, nlags)
    axs[1,1].set_xlabel('Number of lags')
    axs[1,1].set_ylabel('Correlation value')
    
    ax_all.tick_params(axis='x', labelsize=25)
    ax_all.tick_params(axis='y', labelsize=25)
    ax_all.set_xlim(0, nlags)
    ax_all.set_xlabel('Number of lags', fontsize=40)
    ax_all.set_ylabel('Correlation value', fontsize=40)
    ax_all.legend(prop = { "size": 25 })
    
    plt.suptitle('Correlograms of RW in action space for different noise sequence generators')

    ###### Plot mean autocorrelation over a larger number of sampled noise sequences ######

    params['n_init_episodes'] = 1
    params['action_dim'] = nreps

    # urw_acf_res = np.empty((nreps, nlags+1))
    ra_acf_res = np.empty((nreps, nlags+1))
    cnm_0_acf_res = np.empty((nreps, nlags+1))
    cnm_1_acf_res = np.empty((nreps, nlags+1))
    cnm_2_acf_res = np.empty((nreps, nlags+1))

    # params['step_size'] = step_size
    # urw = BrownianMotion(params)
    params['step_size'] = sigma_cnrw
    params['noise_beta'] = 0
    cnm_0 = ColoredNoiseMotion(params)
    params['noise_beta'] = 1
    cnm_1 = ColoredNoiseMotion(params)
    params['noise_beta'] = 2
    cnm_2 = ColoredNoiseMotion(params)
    
    # urw_noise = urw.get_z()
    cnm_noise_0 = cnm_0.get_z()
    cnm_noise_1 = cnm_1.get_z()
    cnm_noise_2 = cnm_2.get_z()

    for i in range(nreps):
        # print(nreps, i, len(urw_noise[:]), len(urw_noise))
        # urw_acf_res[i] = acf(urw_noise[:,i], nlags=nlags)
        cnm_0_acf_res[i] = acf(cnm_noise_0[:,i], nlags=nlags)
        cnm_1_acf_res[i] = acf(cnm_noise_1[:,i], nlags=nlags)
        cnm_2_acf_res[i] = acf(cnm_noise_2[:,i], nlags=nlags)

    # fig, axs = plt.subplots(2, 2)

    # # axs[0,0].plot(range(nlags+1), np.nanmean(urw_acf_res, axis=0))

    # # axs[0,0].set_title('Correlogram for Uniform Random Walk')

    # axs[0,1].plot(range(nlags+1), np.nanmean(cnm_0_acf_res, axis=0))

    # axs[0,1].set_title('Correlogram for Colored Noise with beta = 0 (Brownian Motion)')

    # axs[1,0].plot(range(nlags+1), np.nanmean(cnm_1_acf_res, axis=0))

    # axs[1,0].set_title('Correlogram for Colored Noise with beta = 1')

    # axs[1,1].plot(range(nlags+1), np.nanmean(cnm_2_acf_res, axis=0))

    # axs[1,1].set_title('Correlogram for Colored Noise with beta = 2')

    # plt.suptitle('Correlograms for different noise sequence generators')
    fig_all, ax_all = plt.subplots()

    fig, axs = plt.subplots(3, 1)

    # axs[0,0].plot(range(nlags+1), np.nanmean(urw_acf_res, axis=0))

    # axs[0,0].set_title('Correlogram for Uniform Random Walk')

    axs[0].plot(range(nlags+1), np.nanmean(cnm_0_acf_res, axis=0), color='black')
    ax_all.plot(range(nlags+1), np.nanmean(cnm_0_acf_res, axis=0), linestyle=linestyles[1],
                color=colors.to_rgba(1), label='CNRW_0')

    axs[0].set_title('Correlogram for Colored Noise with beta = 0 (Brownian Motion)')

    axs[0].set_xlim(0, nlags)
    axs[0].set_xlabel('Number of lags')
    axs[0].set_ylabel('Correlation value')

    axs[1].plot(range(nlags+1), np.nanmean(cnm_1_acf_res, axis=0), color='black')
    ax_all.plot(range(nlags+1), np.nanmean(cnm_1_acf_res, axis=0), linestyle=linestyles[2],
                color=colors.to_rgba(2), label='CNRW_1')

    axs[1].set_title('Correlogram for Colored Noise with beta = 1')

    axs[1].set_xlim(0, nlags)
    axs[1].set_xlabel('Number of lags')
    axs[1].set_ylabel('Correlation value')

    axs[2].plot(range(nlags+1), np.nanmean(cnm_2_acf_res, axis=0), color='black')
    ax_all.plot(range(nlags+1), np.nanmean(cnm_2_acf_res, axis=0), linestyle=linestyles[3],
                color=colors.to_rgba(3), label='CNRW_2')

    axs[2].set_title('Correlogram for Colored Noise with beta = 2')

    axs[2].set_xlim(0, nlags)
    axs[2].set_xlabel('Number of lags')
    axs[2].set_ylabel('Correlation value')

    ax_all.tick_params(axis='x', labelsize=25)
    ax_all.tick_params(axis='y', labelsize=25)
    ax_all.set_xlim(0, nlags)
    ax_all.set_xlabel('Number of lags', fontsize=40)
    ax_all.set_ylabel('Correlation value', fontsize=40)
    ax_all.legend(prop = { "size": 25 })

    plt.suptitle('Correlograms for different noise sequence generators')
    
    # plt.figure()

    # plt.plot(range(nlags+1), np.nanmean(urw_acf_res, axis=0))

    # plt.title('Correlogram for Uniform Random Walk')

    # plt.figure()

    # plt.plot(range(nlags+1), np.nanmean(cnm_0_acf_res, axis=0))

    # plt.title('Correlogram for Colored Noise with beta = 0 (Brownian Motion)')

    # plt.figure()

    # plt.plot(range(nlags+1), np.nanmean(cnm_1_acf_res, axis=0))

    # plt.title('Correlogram for Colored Noise with beta = 1')

    # plt.figure()

    # plt.plot(range(nlags+1), np.nanmean(cnm_2_acf_res, axis=0))

    # plt.title('Correlogram for Colored Noise with beta = 2')

    plt.show()
