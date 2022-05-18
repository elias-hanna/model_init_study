import numpy as np
import copy
import os
import matplotlib.pyplot as plt

from model_init_study.visualization.visualization import VisualizationMethod
from model_init_study.visualization.plot_utils import progress_bar
    
class StateSpaceRepartitionVisualization(VisualizationMethod):
    def __init__(self, params):
        super().__init__(params=params)
        self._process_params(params)
        self._trajs = None

    def _process_params(self, params):
        super()._process_params(params)
        if 'state_min' in params:
            self._state_min = params['state_min']
        else:
            raise Exception('StateSpaceRepartitionVisualization _process_params error: state_min not in params')
        if 'state_max' in params:
            self._state_max = params['state_max']
        else:
            raise Exception('StateSpaceRepartitionVisualization _process_params error: state_max not in params')
        if 'obs_dim' in params:
            self._obs_dim = params['obs_dim']
        else:
            raise Exception('StateSpaceRepartitionVisualization _process_params error: obs_dim not in params')

    def set_trajectories(self, trajs):
        self._trajs = trajs
        
    def dump_plots(self, env_name, init_name, num_episodes, traj_type, itr=0, show=False,
                   spe_fig_path=None, label=''):
        if self._trajs is None:
            raise Exception('StateSpaceRepartitionVisualization dump_plots error: trajs not set')
        ## Make dump dirs
        run_name = f'{env_name}_{init_name}_{num_episodes}'
        fig_path = os.path.join(self.dump_path, f'{run_name}/disagr')
        os.makedirs(fig_path, exist_ok=True)

        ## Flatten trajs
        trajectories = np.empty((self._trajs.shape[0]*self._trajs.shape[1],
                                 self._trajs.shape[2]))
        for i in range(self._trajs.shape[0]): ## Over number of trajs
            for j in range(self._trajs.shape[1]): ## Over steps in traj
                trajectories[i*self._trajs.shape[1]+j,:] = self._trajs[i,j,:]
        ## Normalize states
        # Get mins and maxs of obs for each dim
        min_obs = np.min(trajectories, axis=0)
        max_obs = np.max(trajectories, axis=0)
        import pdb; pdb.set_trace()

        ## Careful need to do it dim by dim
        norm_trajs = (trajectories - min_obs)/ \
                     (max_obs - min_obs)
        
        ## Create fig and ax
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ## Plot histogram
        bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]

        n, bins, patches = plt.hist(norm_centroids_disagrs, bins=bins, rwidth=.5)

        plt.title(f"Repartition of {traj_type} observations in whole observation space\nmin obs={min_obs} and max obs={max_obs}")
        plt.xlabel("Min-max normalized value of observation for data samples")
        plt.ylabel("Number of data samples for each bin")

        ## Modify labels to display value range
        rects = ax.patches
        labels = [str(bins[i])+"-"+str(bins[i+1]) for i in range(len(bins)-1)]
        
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height+0.01, label,
                    ha='center', va='bottom')
        if not spe_fig_path:
            ## Save fig
            plt.savefig(f"{fig_path}/results_{itr}/state_space_repartition_",
                        bbox_inches='tight')
        else:
            plt.savefig(f"{spe_fig_path}/", bbox_inches='tight')
        if show:
            plt.show()
        plt.close()




