import numpy as np
import copy
import os
import matplotlib.pyplot as plt
import multiprocessing

from model_init_study.visualization.visualization import VisualizationMethod

class DynamicsVisualization(VisualizationMethod):
    def __init__(self, params):
        super().__init__(params=params)
        self._process_params(params)
        
    def _process_params(self, params):
        super()._process_params(params)
        self._env = params['env'] ## get the gym env
        self.hor = params['sample_hor']
        self.sample_budget = params['sample_budget']
        self.num_cores = params['num_cores']
        
    def set_hor(self, hor):
        self.hor = hor

    def sample(self, n_samples):
        env = copy.copy(self._env) ## copy gym env
        transitions = []
        deltas = []
        obs_shape = env.observation_space.shape
        
        for i in range(n_samples):
            ## Reset environment
            env.reset()
            ## Add a new array to record new transitions
            transitions.append([])
            deltas.append([])
            ## Sample an action in action space
            a = env.action_space.sample()
            ## Sample a starting state in state space
            # s = env.observation_space.sample() # ?
            # s = np.random.uniform(low=self._state_min, high=self._state_max, size=obs_shape)
            # ## Set env at sampled state
            # qpos = s[:len(s)//2]
            # qvel = s[len(s)//2:]
            # ## set_state in env for PETS envs
            # ## set_state in env.env for other envs
            qpos, qvel, s = env.sample_q_vectors()
            
            env.set_state(qpos, qvel)
            ## Perform a self.hor steps in the environment
            for t in range(self.hor):
                ns, r, done, info = env.step(a)
                ## Add observed transition
                transitions[-1].append((copy.copy(s),
                                        copy.copy(a),
                                        copy.copy(ns)))
                ## Sample a new action in action space
                a = env.action_space.sample()
                deltas[-1].append(ns-s)
                s = ns
                
        return transitions, deltas
    
    def dump_plots(self, curr_budget, itr=0, show=False):
        ## Create multiprocessing pool
        pool = multiprocessing.Pool(self.num_cores)
        ## Create arrays to separate budget between jobs
        samples_per_job = self.sample_budget//self.num_cores
        remainder = self.sample_budget%self.num_cores
        n_samples_array = [samples_per_job]*self.num_cores
        n_samples_array[-1] += remainder
        ## Compute the transitions and associated deltas
        results = pool.map(self.sample, n_samples_array)

        ## For debug don't multiprocess:
        # results = self.sample(self.sample_budget)

        transitions = []
        deltas = []
        for result in results:
            for i in range(len(result[0])):
                for t in range(self.hor):
                    loc_transitions = result[0][i][t]
                    transitions.append(loc_transitions)
                    loc_deltas = result[1][i][t]
                    deltas.append(loc_deltas)

        import pdb; pdb.set_trace()
        ## Plot the deltas as boxplots
        fig, ax = plt.subplots()

        deltas_per_state = np.array(deltas).transpose()
        ax.boxplot(deltas_per_state)

        ## Faire sampling de 10000 etats
        ## faire sampling de même 1000 actions par etat
        ## comparer la distribution des transition entre chaque etat
        ## soit regarder par action ou avec toute les actions
        ## Comparer similarité distribution de transition issu
        ## de N etats + 1 action (la meme) à distribution uniforme
        ## Si loin de distriubtion uniforme -> pas uniforme
        ## Si proche -> uniforme
        ## Si uniforme -> facile d'apprendre un modèle, RA suffisent
        ## Si pas uniforme -> plus dur d'apprnedr eun modele,
        ## autres methodes plus corrélé peuvent etre mieux si elles
        ## permettent de voir "plus" de la dynamique du systeme
        
        ax.set_xlabel('')
        ax.set_ylabel('State accross all environment')

        plt.title('')
        fig.set_size_inches(10, 10)
        plt.legend()
        plt.show()
        
        plt.savefig('', dpi=300, bbox_inches='tight')
        ## Format and save the data
        ## Compute median, 1st and 3rd quartile
        np.savez(...)
