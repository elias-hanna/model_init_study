from abc import abstractmethod
import os
import numpy as np
import copy

class VisualizationMethod():
    def __init__(self, params=None):
        self._process_params(params=params)
        
    def _process_params(self, params):
        if 'dump_path' in params:
            self.dump_path = params['dump_path']
        else:
            curr_dir = os.getcwd()
            self.dump_path = curr_dir
        if 'state_min' in params:
            self._state_min = params['state_min']
        else:
            self._state_min = -1
        if 'state_max' in params:
            self._state_max = params['state_max']
        else:
            self._state_max = -1
            
    def prepare_plot(self, plt, fig, ax, mode='3d',
                     ax_labels=['X', 'Y', 'Z'],
                     limits=[0., 1., 0., 1., 0., 1.]):
        ## Set plot labels
        ax.set_xlabel(ax_labels[0])
        ax.set_ylabel(ax_labels[1])
        
        ## Set plot limits
        x_min = limits[0]; x_max = limits[1]; y_min = limits[2]; y_max = limits[3]
        
        # x_min = y_min = z_min = self._state_min
        # x_max = y_max = z_max = self._state_max
        
        ax.set_xlim(x_min,x_max)
        ax.set_ylim(y_min,y_max)

        if mode == '3d':
            z_min = limits[4]; z_max = limits[5]
            ax.set_zlabel(ax_labels[2])
            ax.set_zlim(z_min,z_max)

            ## Set axes ticks (for the grid)
            ticks = [self._state_min + i*(self._state_max - self._state_min)/self._grid_div
                     for i in range(self._grid_div)]
        
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_zticks(ticks)

            ## Add grid to plot 
            plt.grid(True,which="both", linestyle='--')

            ## Set ticks label to show label for only 10 ticks
            mod_val = len(ticks)//10 if len(ticks) > 50 else len(ticks)//5
            ticks_labels = [round(ticks[i],2) if (i)%mod_val==0 else ' '
                            for i in range(len(ticks))]
            ticks_labels[-1] = self._state_max
            ax.set_xticklabels(ticks_labels)
            ax.set_yticklabels(ticks_labels)
            ax.set_zticklabels(ticks_labels)
            ## Invert zaxis for the plot to be like reality
            plt.gca().invert_zaxis()
        
    @abstractmethod
    def dump_plots(self, curr_budget, itr=0, show=False):
        """
        Dump plots to file system
        """
        raise NotImplementedError
