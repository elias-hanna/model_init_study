import numpy as np
import copy
import os
import matplotlib.pyplot as plt

from mb_ge.visualization.visualization import VisualizationMethod
from mb_ge.visualization.plot_utils import progress_bar
    
class DiscretizedStateSpaceVisualization(VisualizationMethod):
    def __init__(self, params=None):
        super().__init__(params=params)
        self._process_params(params)
        self._rope_length = .3
        self._pos_step = 0.2
        self._vel_step = 1.
        self._vel_min = -2
        self._vel_max = 2
        ## Discretize all state space as defined in params and get centroids
        self._centroids = self._get_centroids()

    def _process_params(self, params):
        super()._process_params(params)
        if 'fixed_grid_min' in params:
            self._grid_min = params['fixed_grid_min']
        else:
            raise Exception('DiscretizedStateSpaceVisualization _process_params error: fixed_grid_min not in params')
        if 'fixed_grid_max' in params:
            self._grid_max = params['fixed_grid_max']
        else:
            raise Exception('DiscretizedStateSpaceVisualization _process_params error: fixed_grid_max not in params')
        if 'fixed_grid_div' in params:
            self._grid_div = params['fixed_grid_div']
        else:
            raise Exception('DiscretizedStateSpaceVisualization _process_params error: fixed_grid_div not in params')
        if 'model' in params:
            self.model = params['model']
        else:
            raise Exception('DiscretizedStateSpaceVisualization _process_params error: model not in params')
        if 'env_max_h' in params:
            self.env_max_h = params['env_max_h']
        else:
            raise Exception('DiscretizedStateSpaceVisualization _process_params error: env_max_h not in params')
        if 'controller_type' in params:
            ## Associate an instance of controller_type with given params
            self.controller = params['controller_type'](params=params)
        else:
            raise Exception('DiscretizedStateSpaceVisualization _process_params error: controller_type not in params')
        if 'dynamics_model_params' in params:
            dynamics_model_params = params['dynamics_model_params']
            if 'obs_dim' in dynamics_model_params:
                self._obs_dim = dynamics_model_params['obs_dim']
            else:
                raise Exception('DiscretizedStateSpaceVisualization _process_params error: obs_dim not in params')
            if 'action_dim' in dynamics_model_params:
                self._action_dim = dynamics_model_params['action_dim']
            else:
                raise Exception('DiscretizedStateSpaceVisualization _process_params error: action_dim not in params')
        else:
            raise Exception('DiscretizedStateSpaceVisualization _process_params error: dynamics_model_params not in params')
        if 'nb_of_samples_per_state' in params:
            self._samples_per_state = params['nb_of_samples_per_state']
            self._actions_sampled = np.random.uniform(low=-1, high=1,
                                                      size=(self._samples_per_state,
                                                            self._action_dim))
        else:
            raise Exception('DiscretizedStateSpaceVisualization _process_params error: fixed_grid_min not in params')

    def _reachable(self, centroid):
        return (np.linalg.norm(centroid[:3]) < 0.3)

    def _get_centroids(self):
        centroids = []
        pos_range = np.arange(self._grid_min, self._grid_max, self._pos_step)
        vel_range = np.arange(self._vel_min, self._vel_max, self._vel_step)
        print("Computing centroids for discretized state-spac visualization...")
        progress_cpt = 0
        for x in pos_range:
            for y in pos_range:
                for z in pos_range:
                    for vx in vel_range:
                        for vy in vel_range:
                            for vz in vel_range:
                                centroid = [x, y, z, vx, vy, vz]
                                if self._reachable(centroid):
                                    centroids.append(centroid)
                                progress_bar(progress_cpt, len(pos_range))
            progress_cpt += 1
        print()
        print("Finished computing centroids")
        return centroids

    def _get_state_disagr(self, centroids):
        A = np.tile(self._actions_sampled, (len(centroids), 1))

        all_s = []
        # Get all states to estimate uncertainty for
        for centroid in centroids:
            all_s.append(centroid)
        S = np.repeat(all_s, self._samples_per_state, axis=0)
        # Batch prediction
        batch_pred_delta_ns, batch_disagreement = self.model.forward_multiple(A, S,
                                                                              mean=True,
                                                                              disagr=True)
        centroids_disagrs = []
        for i in range(len(centroids)):
            disagrs = batch_disagreement[i*self._samples_per_state:
                                         i*self._samples_per_state+
                                         self._samples_per_state]
            
            centroids_disagrs.append(np.mean([np.mean(disagr.detach().numpy())
                                          for disagr in disagrs]))
        return centroids_disagrs
    
    def dump_plots(self, curr_budget, itr=0, show=False):
        ## For each centroid compute mean model ensemble disagreement over sampled actions
        ## Use self.sampled_actions
        centroids_disagrs = self._get_state_disagr(self._centroids)
        ## Normalize disagr
        min_disagr = min(centroids_disagrs)
        max_disagr = max(centroids_disagrs)

        norm_centroids_disagrs = (centroids_disagrs - np.min(centroids_disagrs))/ \
                                 (np.max(centroids_disagrs) - np.min(centroids_disagrs))
        
        ## Create fig and ax
        # fig = plt.figure(figsize=(8, 8), dpi=160)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ## Prepare plot 
        # self.prepare_plot(plt, fig, ax, mode='2d')

        ## Plot histogram
        bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]

        n, bins, patches = plt.hist(norm_centroids_disagrs, bins=bins, rwidth=.5)

        plt.title(f"Disagreement repartition in whole state space\nmin disagreement={min_disagr} and max disagreement={max_disagr}")
        plt.xlabel("Normalized value of disagreement for data samples")
        plt.ylabel("Number of data samples for each bin")

        ## Modify labels to display value range
        rects = ax.patches
        labels = [str(bins[i])+"-"+str(bins[i+1]) for i in range(len(bins)-1)]
        
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height+0.01, label,
                    ha='center', va='bottom')
        
        ## Save fig
        plt.savefig(f"{self.dump_path}/results_{itr}/test_trajectories_pred_error_{curr_budget}",
                    bbox_inches='tight')

        if show:
            plt.show()
        plt.close()




