import numpy as np
import copy
import os
import matplotlib.pyplot as plt
import multiprocessing
from itertools import repeat

from model_init_study.visualization.visualization import VisualizationMethod

class DynamicsVisualization(VisualizationMethod):
    def __init__(self, params):
        super().__init__(params=params)
        self._process_params(params)
        
    def _process_params(self, params):
        super()._process_params(params)
        self._env = params['env'] ## get the gym env
        self._env_name = params['env_name'] ## get the gym env
        self.hor = params['sample_hor']
        self.state_sample_budget = params['state_sample_budget']
        self.action_sample_budget = params['action_sample_budget']
        self.num_cores = params['num_cores']
        
    def set_hor(self, hor):
        self.hor = hor

    def sample_per_action(self, n_actions, n_samples):
        '''
        This method samples n_actions actions from the action space and for
        each of the actions samples n_samples states from the state space and
        perform the action in each of the sampled states.
        Returns the observed transitions and corresponding state deltas
        in a tab organized per action sampled
        '''
        env = copy.copy(self._env) ## copy gym env
        transitions = []
        deltas = []
        obs_shape = env.observation_space.shape

        for i in range(n_actions):
            ## Add a new array to record new transitions
            transitions.append([])
            deltas.append([])
            ## sample a single action
            a = env.action_space.sample()
            a = np.clip(a, -1, 1)
            for j in range(n_samples):
                ## Reset environment
                env.reset()
                qpos, qvel, s = env.sample_q_vectors()
                env.set_state(qpos, qvel)
                ns, r, done, info = env.step(a)
                ## Add observed transition
                transitions[-1].append((copy.copy(s),
                                        copy.copy(a),
                                        copy.copy(ns)))
                deltas[-1].append(ns-s)
                s = ns
                
        return transitions, deltas
        
    def sample(self, n_samples):
        '''
        This method samples n_samples state and actions from state-action space
        and perform the sampled action(s) in each of the sampled states up to
        self.hor horizon.
        Returns the observed transitions and corresponding state deltas
        in a tab organized per action-state initial samples.
        '''
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
            qpos, qvel, s = env.sample_q_vectors()
            
            env.set_state(qpos, qvel)
            ## Perform self.hor steps in the environment
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

    def transition_postproc(self, transitions):
        proc_deltas = []
        proc_transitions = []
        ## For all fastsim based envs
        if 'maze' in self._env_name:
            ## Change transitions to only take into account
            ## vector between end point and starting point
            ## independantly of robot orientation
            ## fastsim obs are: [pos_x, pos_y, vel_x, vel_y, sin(ori), cos(ori)]

            ## Do pos_xy / vel_xy then - pos_xy / vel_xy start
            for transition in transitions:
                start = transition[0]
                act = transition[1]
                end = transition[2]
                proc_delta = end[:4] - start[:4]
                ## Compute or (atan2 of sin(ori) and cos(ori))
                ori = np.arctan2(start[4], start[5])
                ## Rx rotates around x axis to account for fastsim frame ori
                Rx = np.array([[1, 0, 0],
                               [0, np.cos(np.pi), -np.sin(np.pi)],
                               [0, np.sin(np.pi), np.cos(np.pi)]])
                ## Rz rotates around z axis to align with robot ori
                Rz = np.array([[np.cos(ori), -np.sin(ori), 0],
                               [np.sin(ori), np.cos(ori), 0],
                               [0, 0, 1]])
                ## Apply rotation matrix to [delta_pos_xy, delta_vel_xy] vector
                to_rot = [0]*3
                to_rot[:2] = proc_delta[:2]
                proc_delta[:2] = Rx.T.dot(Rz.T.dot(to_rot))[:2]
                to_rot[:2] = proc_delta[2:]
                proc_delta[2:] = Rx.T.dot(Rz.T.dot(to_rot))[:2]
                proc_deltas.append(proc_delta)
                print(proc_delta)
            
            # import pdb; pdb.set_trace()
            proc_transitions = transitions
        else:
            proc_transitions = transitions
        return proc_transitions, proc_deltas
    
    def dump_plots(self, curr_budget, itr=0, show=False):
        ## Create multiprocessing pool
        pool = multiprocessing.Pool(self.num_cores)

        # ## Create arrays to separate budget between jobs
        # samples_per_job = self.state_sample_budget//self.num_cores
        # remainder = self.state_sample_budget%self.num_cores
        # n_samples_array = [samples_per_job]*self.num_cores
        # n_samples_array[-1] += remainder
        # ## Compute the transitions and associated deltas
        # results = pool.map(self.sample, n_samples_array)

        # ## For debug don't multiprocess:
        # # results = self.sample(self.state_sample_budget)

        # transitions = []
        # deltas = []
        # for result in results:
        #     for i in range(len(result[0])):
        #         for t in range(self.hor):
        #             loc_transitions = result[0][i][t]
        #             transitions.append(loc_transitions)
        #             loc_deltas = result[1][i][t]
        #             deltas.append(loc_deltas)

        # import pdb; pdb.set_trace()
        # ## Plot the deltas as boxplots
        # fig, ax = plt.subplots()

        # deltas_per_state = np.array(deltas).transpose()

        # ax.boxplot(deltas_per_state)
        # plt.show()


        ## Sample N actions, execute each action in M different states

        ## Create arrays to separate budget between jobs
        samples_per_job = self.action_sample_budget//self.num_cores
        remainder = self.action_sample_budget%self.num_cores
        n_actions_samples_array = [samples_per_job]*self.num_cores
        n_actions_samples_array[-1] += remainder
        
        args = zip(n_actions_samples_array,
                    repeat(self.state_sample_budget))
        
        results = pool.starmap(self.sample_per_action, args)

        # ## For debug
        # results = self.sample_per_action(self.action_sample_budget, self.state_sample_budget)


        ## Regroup results from pool, still keep them per action
        # transitions = []
        # deltas = []
        transitions = results[0][0]
        deltas = results[0][1]
        
        for result in results[1:]:
            transitions += result[0]            
            deltas += result[1]

        ## Process transitions to reflect data symmetry / invariance
        proc_transitions = []
        proc_deltas = []
        for action_transitions in transitions:
            proc_action_transitions, proc_action_deltas = \
                self.transition_postproc(action_transitions)
            proc_transitions.append(proc_action_transitions)
            proc_deltas.append(proc_action_deltas)

        transitions = np.array(transitions)
        deltas = np.array(deltas)
        if 'maze' in self._env_name:
            transitions = np.array(proc_transitions)
            deltas = np.array(proc_deltas)

        ## Sort per action vector norm (lowest to highest, reverse=False)
        # sorted_deltas = [x for _,x in sorted(zip(transitions,deltas),
        #                                      key=lambda x:np.linalg.norm(x[0][0,1]))]
        # sorted_transitions = sorted(transitions,
        #                             key=lambda x:np.linalg.norm(x[0,1]))
        
        # ## Plot the deltas as boxplots
        # deltas = np.array(sorted_deltas)
        # transitions = np.array(sorted_transitions)

        ## iterate over state dim
        import pdb; pdb.set_trace()
        for i in range(deltas.shape[2]):
            # fig, ax = plt.subplots()
            # ## I don't get why I need to transpose, normally it takes columns
            # ax.boxplot(deltas[:,:,i].T)
            # plt.show()
            fig, ax = plt.subplots()
            to_plot = [deltas[j,:,i] for j in range(self.action_sample_budget)]
            ## I don't get why I need to transpose, normally it takes columns
            # ret_dict = ax.boxplot(to_plot, 0, '') ## options don't show outliers
            ret_dict = ax.boxplot(to_plot) 
            plt.show()

        
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
