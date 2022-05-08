import numpy as np
import copy
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from mb_ge.visualization.visualization import VisualizationMethod

class ArchiveVisualization(VisualizationMethod):
    def __init__(self, params=None):
        super().__init__(params=params)
        self._process_params(params)

    def _process_params(self, params):
        super()._process_params(params)
        if 'archive' in params:
            self._archive = params['archive']._archive
            self._grid_min = params['archive']._grid_min
            self._grid_max = params['archive']._grid_max
            self._grid_div = params['archive']._grid_div
        else:
            raise Exception('ArchiveVisualization _process_params error: archive not in params')
            
    def dump_plots(self, curr_budget, itr=0, show=False,
                   plot=False, plot_disagr=False, plot_novelty=False):
        x = []
        y = []
        z = []
        disagrs = []
        novelties = []
        ## Add the BD data from archive:
        for key in self._archive.keys():
            elements = self._archive[key].get_elements()
            for el in elements:
                x.append(el.descriptor[0])
                y.append(el.descriptor[1])
                z.append(el.descriptor[2])
                disagrs.append(el.end_state_disagr)
                novelties.append(el.novelty)

        limits = [self._grid_min, self._grid_max,   # x
                  self._grid_min, self._grid_max,   # y
                  self._grid_min, self._grid_max, ] # z
        if plot:
            ## Create fig and ax
            fig = plt.figure(figsize=(8, 8), dpi=160)  
            ax = fig.add_subplot(111, projection='3d')  
x
            self.prepare_plot(plt, fig, ax)
            
            ## Scatter plot 
            ax.scatter(x, y, z)
            
            ## Set plot title
            plt.title(f'State Archive at {curr_budget} evaluations', fontsize=8)
            
            ## Save fig
            plt.savefig(f"{self.dump_path}/results_{itr}/state_archive_at_{curr_budget}_eval", bbox_inches='tight')
        
        if plot_disagr:
            # fig = plt.figure(figsize=(8, 8), dpi=160)  
            fig = plt.figure()  
            ax = fig.add_subplot(111, projection='3d')  

            self.prepare_plot(plt, fig, ax)

            min_disagr = np.min(disagrs)
            max_disagr = np.max(disagrs)
            ## Handle case where disagr can't be computed
            can_plot = True
            if (max_disagr - min_disagr) == 0:
                can_plot = False
                print("WARNING: Can't plot archive with disagreement")
            if can_plot:
                norm_disagr =  (disagrs - np.min(disagrs))/(np.max(disagrs)-np.min(disagrs))
                
                cm = plt.cm.get_cmap()
                sc = ax.scatter(x, y, z, c=norm_disagr, cmap=cm)
                clb = plt.colorbar(sc)
                ## Round for cleaner plot
                min_disagr = round(min_disagr,4)
                max_disagr = round(max_disagr,4)
                clb.set_label('Normalized model ensemble disagreement for each reached state')
                clb.ax.set_title(f'min disagr={min_disagr} and max disagr={max_disagr}')
                
                ## Set plot title
                plt.title(f'State Archive at {curr_budget} evaluations', fontsize=8)
                
                ## Save fig
                plt.savefig(f"{self.dump_path}/results_{itr}/disagr_state_archive_at_{curr_budget}_eval", bbox_inches='tight')

        if plot_novelty:
            fig = plt.figure(figsize=(8, 8), dpi=160)  
            ax = fig.add_subplot(111, projection='3d')  

            self.prepare_plot(plt, fig, ax)

            min_nov = np.min(novelties)
            max_nov = np.max(novelties)
            ## Handle case where novelty can't be computed
            can_plot = True
            if (max_nov - min_nov) == 0:
                can_plot = False
                print("WARNING: Can't plot archive with novelty")
            if can_plot:
                norm_nov =  (novelties - np.min(novelties))/(np.max(novelties)-np.min(novelties))
                
                cm = plt.cm.get_cmap()
                sc = ax.scatter(x, y, z, c=norm_nov, cmap=cm)
                clb = plt.colorbar(sc)
                ## Round for cleaner plot
                min_nov = round(min_nov,4)
                max_nov = round(max_nov,4)
                
                clb.set_label('Normalized novelty for each reached state')
                clb.ax.set_title(f'min nov={min_nov} and max nov={max_nov}')
                
                ## Set plot title
                plt.title(f'State Archive at {curr_budget} evaluations', fontsize=8)
                
                ## Save fig
                plt.savefig(f"{self.dump_path}/results_{itr}/novelty_state_archive_at_{curr_budget}_eval", bbox_inches='tight')

        if show:
            plt.show()
        plt.close()
