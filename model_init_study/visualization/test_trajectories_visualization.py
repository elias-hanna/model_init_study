import numpy as np
import copy
import os
import matplotlib.pyplot as plt

from model_init_study.visualization.visualization import VisualizationMethod

class TestTrajectoriesVisualization(VisualizationMethod):
    def __init__(self, params):
        super().__init__(params=params)
        self._process_params(params)
        self._pred_error_thresh = .03 # 3 cm
        
    def _process_params(self, params):
        super()._process_params(params)
        if 'path_to_test_trajectories' in params:
            self.test_trajectories = np.load(params['path_to_test_trajectories'])['examples']
            self.test_params = np.load(params['path_to_test_trajectories'])['params']
            ## /!\ Warning, trajs must be of shape (nb_of_trajs, nb_of_steps, state_dim)
        else:
            raise Exception('TestTrajectoriesVisualization _process_params error: path_to_test_trajectories not in params')
        if 'env_max_h' in params:
            self.env_max_h = params['env_max_h']
        else:
            raise Exception('TestTrajectoriesVisualization _process_params error: env_max_h not in params')
        if 'controller_type' in params:
            ## Associate an instance of controller_type with given params
            self.controller = params['controller_type'](params=params)
        else:
            raise Exception('ExplorationMethod _process_params error: controller_type not in params')
        if 'model' in params:
            self.model = params['model']
        else:
            raise Exception('TestTrajectoriesVisualization _process_params error: model not in params')
        if 'dynamics_model_params' in params:
            dynamics_model_params = params['dynamics_model_params']
            if 'obs_dim' in dynamics_model_params:
                self._obs_dim = dynamics_model_params['obs_dim']
            else:
                raise Exception('TestTrajectoriesVisualization _process_params error: obs_dim not in params')
            if 'action_dim' in dynamics_model_params:
                self._action_dim = dynamics_model_params['action_dim']
            else:
                raise Exception('TestTrajectoriesVisualization _process_params error: action_dim not in params')
        else:
            raise Exception('TestTrajectoriesVisualization _process_params error: dynamics_model_params not in params')

    def set_test_trajectories(self, test_trajectories):
        self.test_trajectories = test_trajectories
        
    def _execute_test_trajectories_on_model(self):
        controller_list = []

        traj_list = []
        disagreements_list = []
        prediction_errors_list = []

        pred_trajs = np.empty((len(self.test_trajectories), self.env_max_h, self._obs_dim))
        disagrs = np.empty((len(self.test_trajectories), self.env_max_h))
        pred_errors = np.empty((len(self.test_trajectories), self.env_max_h))
        
        A = np.empty((len(self.test_trajectories), self._action_dim))
        S = np.empty((len(self.test_trajectories), self._obs_dim))
        
        for i in range(len(self.test_trajectories)):
            ## Create a copy of the controller
            controller_list.append(self.controller.copy())
            ## Set controller parameters
            controller_list[-1].set_parameters(self.test_params[i])
            ## Init starting state
            S[i,:] = self.test_trajectories[i,0,:]
        
        for i in range(self.env_max_h):
            for j in range(len(self.test_trajectories)):
                A[j,:] = controller_list[j](S[j,:])
            batch_pred_delta_ns, batch_disagreement = self.model.forward_multiple(A, S,
                                                                                  mean=True,
                                                                                  disagr=True)
            for j in range(len(self.test_trajectories)):
                ## Compute mean prediction from model samples
                next_step_pred = batch_pred_delta_ns[j]
                mean_pred = [np.mean(next_step_pred[:,k]) for k in range(len(next_step_pred[0]))]
                S[j,:] += mean_pred.copy()
                pred_trajs[j,i,:] = mean_pred.copy()
                disagrs[j,i] = np.mean(batch_disagreement[j].detach().numpy())
                pred_errors[j,i] = np.linalg.norm(S[j,:]-self.test_trajectories[j,i,:])
                if pred_errors[j, i] == np.inf:# or pred_errors[j, i] > 100:
                    pred_errors[j, i] = np.nan
                    
        return pred_trajs, disagrs, pred_errors
        
    def dump_plots(self, env_name, init_name, num_episodes, traj_type, dump_separate=False, show=False, model_trajs=None):
        ## Get results of test trajectories on model on last model update
        if model_trajs == None:
            pred_trajs, disagrs, pred_errors = self._execute_test_trajectories_on_model()
        else:
            pred_trajs, disagrs, pred_errors = model_trajs
            
        ## Make dump dirs
        run_name = f'{env_name}_{init_name}_{num_episodes}'
        fig_path = os.path.join(self.dump_path, f'{run_name}/disagr')
        os.makedirs(fig_path, exist_ok=True)        
        ## Compute mean and stddev of trajs disagreement
        mean_disagr = np.nanmean(disagrs, axis=0)
        std_disagr = np.nanstd(disagrs, axis=0)
        ## Compute mean and stddev of trajs prediction error
        mean_pred_error = np.nanmean(pred_errors, axis=0)
        std_pred_error = np.nanstd(pred_errors, axis=0)
            
        ## Create fig and ax
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ## Prepare plot
        labels = ['Number of steps on environment', 'Mean ensemble disagreement']
        limits = [0, len(mean_disagr),
                  min(mean_disagr-std_disagr), max(mean_disagr+std_disagr)]
        self.prepare_plot(plt, fig, ax, mode='2d', limits=limits, ax_labels=labels)

        ## Figure for model ensemble disagreement
        plt.plot(range(len(mean_disagr)), mean_disagr, 'k-')
        plt.fill_between(range(len(mean_disagr)),
                         mean_disagr-std_disagr,
                         mean_disagr+std_disagr,
                         facecolor='green', alpha=0.5)
        ## Set plot title
        plt.title(f"Mean model ensemble disagreeement along test trajectories\n {init_name} on {num_episodes} episodes")
        ## Save fig
        plt.savefig(f"{fig_path}/mean_test_trajectories_disagr_{traj_type}",
                    bbox_inches='tight')

        if dump_separate:
            for i in range(len(disagrs)):
                ## Create fig and ax
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ## Prepare plot
                labels = ['Number of steps on environment', 'Ensemble disagreement']
                limits = [0, len(disagrs[i]),
                          min(disagrs[i]), max(disagrs[i])]
                self.prepare_plot(plt, fig, ax, mode='2d', limits=limits, ax_labels=labels)
                
                ## Figure for model ensemble disagreement
                plt.plot(range(len(disagrs[i])), disagrs[i], 'k-')
                # plt.fill_between(range(len(mean_disagr)),
                                 # mean_disagr-std_disagr,
                                 # mean_disagr+std_disagr,
                                 # facecolor='green', alpha=0.5)
                ## Set plot title
                plt.title(f"Mean model ensemble disagreeement along single test trajectory\n {init_name} on {num_episodes} episodes")
                ## Save fig
                plt.savefig(f"{fig_path}/{i}_test_trajectories_disagr_{traj_type}",
                            bbox_inches='tight')

        ## Make dump dirs
        fig_path = os.path.join(self.dump_path, f'{run_name}/pred_error')
        os.makedirs(fig_path, exist_ok=True)        
            
        ## Create fig and ax
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ## Prepare plot
        labels = ['Number of steps on environment', 'Mean prediction error']
        limits = [0, len(mean_pred_error),
                  # min(mean_pred_error-std_pred_error), max(mean_pred_error+std_pred_error)]
                  0, 10]
        self.prepare_plot(plt, fig, ax, mode='2d', limits=limits, ax_labels=labels)

        ## Figure for prediction error
        plt.plot(range(len(mean_pred_error)), mean_pred_error, 'k-')
        
        plt.fill_between(range(len(mean_pred_error)),
                         mean_pred_error-std_pred_error,
                         mean_pred_error+std_pred_error,
                         facecolor='green', alpha=0.5)
        ## Set plot title
        plt.title(f"Mean prediction error along test trajectories")
        ## Save fig
        plt.savefig(f"{fig_path}/mean_test_trajectories_pred_error_{traj_type}",
                    bbox_inches='tight')

        if dump_separate:
            for i in range(len(pred_errors)):
                ## Create fig and ax
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ## Prepare plot
                labels = ['Number of steps on environment', 'Prediction Error']
                limits = [0, len(pred_errors[i]),
                          min(pred_errors[i]), max(pred_errors[i])]
                self.prepare_plot(plt, fig, ax, mode='2d', limits=limits, ax_labels=labels)
                
                ## Figure for model ensemble disagreement
                plt.plot(range(len(pred_errors[i])), pred_errors[i], 'k-')
                # plt.fill_between(range(len(mean_disagr)),
                                 # mean_disagr-std_disagr,
                                 # mean_disagr+std_disagr,
                                 # facecolor='green', alpha=0.5)
                ## Set plot title
                plt.title(f"Mean prediction error along single test trajectory")
                ## Save fig
                plt.savefig(f"{fig_path}/{i}_test_trajectories_pred_error_{traj_type}",
                            bbox_inches='tight')

        if show:
            plt.show()
        plt.close()

        return pred_trajs, disagrs, pred_errors
