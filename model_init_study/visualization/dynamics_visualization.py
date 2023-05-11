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
        for i in range(n_samples):
            ## Reset environment
            env.reset()
            ## Add a new array to record new transitions
            transitions.append([])
            ## Sample an action in action space
            a = env.action_space.sample()
            ## Sample a starting state in state space
            s = env.state_space.sample() # ?
            s = np.random.uniform(self._state_min, self._state_max)
            ## Set env at sampled state
            env.set_state(s)
            ## Perform a self.hor steps in the environment
            for t in range(self.hor):
                ns, r, done, info = env.step(a)
                ## Add observed transition
                transitions[-1].append((copy.copy(s),
                                        copy.copy(a),
                                        copy.copy(ns)))
                ## Sample a new action in action space
                a = env.action_space.sample()
                s = ns
                deltas.append(ns-n)
        return transitions
    
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
        transitions = []
        deltas = []
        for i in range(len(results)):
            loc_transitions = results[0]
            transitions += loc_transitions
            loc_deltas = results[1]
            deltas += loc_deltas

        ## Plot the deltas as boxplots
        fig, ax = plt.subplots()

        ax.boxplot(deltas)

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
        plt.savefig('', dpi=300, bbox_inches='tight')
    
        ## Format and save the data
        ## Compute median, 1st and 3rd quartile
        np.savez(...)
