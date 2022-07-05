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
        self._concurrent_trajs = None
        
    def _process_params(self, params):
        super()._process_params(params)
        if 'obs_dim' in params:
            self._obs_dim = params['obs_dim']
        else:
            raise Exception('StateSpaceRepartitionVisualization _process_params error: obs_dim not in params')

    def set_trajectories(self, trajs):
        self._trajs = trajs

    def set_concurrent_trajectories(self, trajs):
        self._concurrent_trajs = trajs
        
    def dump_plots(self, env_name, init_name, num_episodes, traj_type, dim_type='action',
                   itr=0, show=False, spe_fig_path=None, label='', use_concurrent_trajs=False,
                   legends=['', '']):
        if self._trajs is None:
            raise Exception('StateSpaceRepartitionVisualization dump_plots error: _trajs not set')
        if use_concurrent_trajs:
            if self._concurrent_trajs is None:
                raise Exception('StateSpaceRepartitionVisualization dump_plots error: _concurrent_trajs not set')
        ## Make dump dirs
        run_name = f'{env_name}_{num_episodes}'
        fig_path = os.path.join(self.dump_path, f'{run_name}')
        os.makedirs(fig_path, exist_ok=True)

        if len(self._trajs.shape) > 2:
            ## Flatten trajs
            trajectories = np.empty((self._trajs.shape[0]*self._trajs.shape[1],
                                     self._trajs.shape[2]))
            for i in range(self._trajs.shape[0]): ## Over number of trajs
                for j in range(self._trajs.shape[1]): ## Over steps in traj
                    trajectories[i*self._trajs.shape[1]+j,:] = self._trajs[i,j,:]
            if use_concurrent_trajs:
                concurrent_trajectories = np.empty((self._concurrent_trajs.shape[0]*
                                                    self._concurrent_trajs.shape[1],
                                                    self._concurrent_trajs.shape[2]))
                for i in range(self._concurrent_trajs.shape[0]): ## Over number of trajs
                    for j in range(self._concurrent_trajs.shape[1]): ## Over steps in traj
                        concurrent_trajectories[i*self._concurrent_trajs.shape[1]+j,:] = self._concurrent_trajs[i,j,:]
        else:
            trajectories = self._trajs
            if use_concurrent_trajs:
                concurrent_trajectories = self._concurrent_trajs
        ## Normalize states
        if use_concurrent_trajs:
            stacked_trajs = np.vstack((trajectories, concurrent_trajectories))
            # Get mins and maxs of obs for each dim
            min_obs = np.min(stacked_trajs, axis=0)
            max_obs = np.max(stacked_trajs, axis=0)
            
            ## Careful need to do it dim by dim
            norm_trajs = (trajectories - min_obs)/ \
                         (max_obs - min_obs)
            ## Careful need to do it dim by dim
            norm_concurrent_trajs = (concurrent_trajectories - min_obs)/ \
                                    (max_obs - min_obs)
        
        else:
            # Get mins and maxs of obs for each dim
            min_obs = np.min(trajectories, axis=0)
            max_obs = np.max(trajectories, axis=0)
            
            ## Careful need to do it dim by dim
            norm_trajs = (trajectories - min_obs)/ \
                         (max_obs - min_obs)
        
        ## Create fig and ax
        bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]

        for dim in range(self._trajs.shape[1]):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ## Plot histogram
            n, h_bins, patches = ax.hist(norm_trajs[:, dim], bins=bins, rwidth=.5)
            if use_concurrent_trajs:
                c_n, c_h_bins, c_patches = ax.hist(norm_concurrent_trajs[:, dim],
                                                   bins=bins, rwidth=.5)

                ## Get highest patch
                max_height = max(n) if max(n) > max(c_n) else max(c_n)
                ax.set_ylim([-max_height, max_height])
                ## Modify rects to make them reverse
                # rects = ax[dim].patches
                # rects = ax.c_patches

                # for rect, label in zip(rects, labels):
                for rect in c_patches:
                    height = rect.get_height()
                    rect.set_height(-height)
                
            plt.title(f"Training data distribution for {dim_type} dimension {dim}")
            plt.legend(legends, prop={'size': 15})
            plt.xlabel("Min-max normalized value of state for data samples")
            plt.ylabel("Number of data samples per bin")
            
            if show:
                plt.show()
            
            if spe_fig_path is None:
                ## Save fig
                plt.savefig(f"{fig_path}/state_space_repartition_dim_{dim}_{label}",
                            bbox_inches='tight')
            else:
                plt.savefig(f"{spe_fig_path}_dim_{dim}_{label}", bbox_inches='tight')

            plt.close()



