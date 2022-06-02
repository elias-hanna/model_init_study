import numpy as np
import copy
import os
import matplotlib.pyplot as plt

from model_init_study.visualization.visualization import VisualizationMethod

class NStepErrorVisualization(VisualizationMethod):
    def __init__(self, params):
        super().__init__(params=params)
        self._process_params(params)
        self._pred_error_thresh = .03 # 3 cm
        self._n = 1
        
    def _process_params(self, params):
        super()._process_params(params)
        self.traj_separator = params['separator']()
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

    def set_n(self, n):
        self._n = n
        
    def _execute_test_trajectories_on_model(self):
        controller_list = []

        traj_list = []
        disagreements_list = []
        prediction_errors_list = []

        disagrs = np.empty((len(self.test_trajectories), self.env_max_h))
        disagrs[:] = np.nan
        pred_errors = np.empty((len(self.test_trajectories), self.env_max_h, self._obs_dim))
        pred_errors[:] = np.nan
        
        A = np.empty((len(self.test_trajectories), self._action_dim))
        S = np.empty((len(self.test_trajectories), self._obs_dim))
        
        for i in range(len(self.test_trajectories)):
            ## Create a copy of the controller
            controller_list.append(self.controller.copy())
            ## Set controller parameters
            controller_list[-1].set_parameters(self.test_params[i])
            # Init starting state
            # S[i,:] = self.test_trajectories[i,0,:]

        loc_A = np.empty((len(self.test_trajectories), self._action_dim))
        loc_S = np.empty((len(self.test_trajectories), self._obs_dim))

        ended = [False] * len(self.test_trajectories)
        
        for i in range(self.env_max_h-self._n):
            for j in range(len(self.test_trajectories)):
                S[j,:] = self.test_trajectories[j,i,:]
                A[j,:] = controller_list[j](S[j,:])
                if np.isnan(self.test_trajectories[j,i,:]).any():
                    ended[j] = True

            if all(ended):
                break
            
            ## This will store recursively obtained pred err and disagr for given n-steps
            cum_disagr = [0.]*len(self.test_trajectories)
            
            loc_S = S.copy()
            loc_A = A.copy()
            ## For each traj point we do n step predictions to see disagr and error
            for k in range(self._n):
                for j in range(len(self.test_trajectories)):
                    # loc_S[j,:] = self.test_trajectories[j,i,:]
                    loc_A[j,:] = controller_list[j](loc_S[j,:])

                batch_pred_delta_ns, batch_disagreement = self.model.forward_multiple(loc_A,
                                                                                      loc_S,
                                                                                      mean=True,
                                                                                      disagr=True)
                for j in range(len(self.test_trajectories)):
                    ## Compute mean prediction from model samples
                    next_step_pred = batch_pred_delta_ns[j]
                    mean_pred = [np.mean(next_step_pred[:,k])
                                 for k in range(len(next_step_pred[0]))]
                    loc_S[j,:] += mean_pred.copy()
                    loc_d = np.mean(batch_disagreement[j].detach().numpy())
                    cum_disagr[j] += np.mean(batch_disagreement[j].detach().numpy())
                    
            ## Now save the (cumulated) disagr and pred error values
            for j in range(len(self.test_trajectories)):
                if ended[j]:
                    continue
                
                # disagr for one point == cumulated disagr over prediction horizon self._n
                disagrs[j,i] = cum_disagr[j]
                # pred error for one point == pred error between recursive prediction and GT
                for dim in range(self._obs_dim):
                    # pred_errors[j,i,dim] = np.linalg.norm(loc_S[j,dim]-self.test_trajectories[j,i+self._n,dim])
                    ## Keep it signed
                    pred_errors[j,i,dim] = loc_S[j,dim]-self.test_trajectories[j,i+self._n,dim]
            
        pred_trajs = self.test_trajectories
        return pred_trajs, disagrs, pred_errors

    def compute_pred_error(self, traj1, traj2):
        pred_errors = np.empty((len(traj1), self.env_max_h))
        has_nan = [False for _ in range(len(pred_errors))]

        for i in range(len(traj1)//2):
            for t in range(self.env_max_h):
                ind = i*len(self.test_trajectories)
                pred_errors[ind,t] = np.linalg.norm(traj1[ind,t,:]-traj2[0,t,:])
                pred_errors[ind+1,t] = np.linalg.norm(traj1[ind+1,t,:]-traj2[1,t,:])
                if has_nan[ind] or np.isinf(pred_errors[ind, t]) or np.isnan(pred_errors[ind, t]):
                    has_nan[ind] = True
                    pred_errors[ind, t] = np.nan
                if has_nan[ind+1] or np.isinf(pred_errors[ind+1, t]) or np.isnan(pred_errors[ind+1, t]):
                    has_nan[ind+1] = True
                    pred_errors[ind, t] = np.nan
                # if pred_errors[ind, t] > 20:
                    # pred_errors[ind, t] = 20
                # if pred_errors[ind+1, t] > 20:
                    # pred_errors[ind, t] = 20
        return pred_errors
    
    def dump_plots(self, env_name, init_name, num_episodes, traj_type, dump_separate=False,
                   show=False, model_trajs=None, plot_stddev=True, no_sep=False):
        ## Get results of test trajectories on model on last model update
        if model_trajs == None:
            pred_trajs, disagrs, pred_errors = self._execute_test_trajectories_on_model()
        else:
            pred_trajs, disagrs, pred_errors = model_trajs

        if no_sep: ## Typically use this when just foraging data
            return self.dump_plot(env_name, init_name, num_episodes, traj_type,
                                  dump_separate=dump_separate, show=show,
                                  model_trajs=(pred_trajs, disagrs, pred_errors),
                                  plot_stddev=plot_stddev)
        
        separated_trajs, labels = self.traj_separator.separate_trajs(pred_trajs)
        test_separated_trajs, labels = self.traj_separator.separate_trajs(self.test_trajectories)
        pred_errors = []
        for i in range(len(labels)):
            pred_errors.append(self.compute_pred_error(separated_trajs[i],
                                                       test_separated_trajs[i]))
        
        for i in range(len(labels)):
            self.dump_plot(env_name, init_name, num_episodes, traj_type,
                           dump_separate=dump_separate, show=show,
                           model_trajs=(separated_trajs[i], disagrs, pred_errors[i]),
                           plot_stddev=plot_stddev, label=labels[i])

    def dump_plot(self, env_name, init_name, num_episodes, traj_type, dump_separate=False,
                   show=False, model_trajs=None, plot_stddev=True, label=''):
        # Get results of test trajectories on model on last model update
        # if model_trajs == None:
            # pred_trajs, disagrs, pred_errors = self._execute_test_trajectories_on_model()
        # else:
            # pred_trajs, disagrs, pred_errors = model_trajs
        pred_trajs, disagrs, pred_errors = model_trajs
        ## Make dump dirs
        run_name = f'{env_name}_{init_name}_{num_episodes}'
        fig_path_disagr = os.path.join(self.dump_path, f'{run_name}/disagr')
        os.makedirs(fig_path_disagr, exist_ok=True)

        fig_path_pred_error = os.path.join(self.dump_path, f'{run_name}/pred_error')
        os.makedirs(fig_path_pred_error, exist_ok=True)

        print(f'Current working dir: {os.getcwd()}')
        print(f'{self._n} step error vis dumping figs on {fig_path_pred_error} and {fig_path_disagr}')

        ## For each pred_traj (here == test trajs)
        # for pred_traj in pred_trajs:
        for i in range(len(pred_trajs)):
            pred_traj = pred_trajs[i]
            pred_error = pred_errors[i]
            disagr = disagrs[i]
            for dim in range(pred_traj.shape[1]):
                ### Model prediction error ###

                ## Create fig and ax
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ## Prepare plot
                labels = ['Number of steps on environment',
                          f'Trajectory on dimension {dim} on label {label}']
                limits = [0, len(pred_traj[:,dim]),
                          min(min(pred_traj[:, dim]), min(pred_traj[:, dim]+pred_error[:, dim])),
                          max(max(pred_traj[:, dim]), max(pred_traj[:,dim]+pred_error[:, dim]))]

                self.prepare_plot(plt, fig, ax, mode='2d', limits=limits, ax_labels=labels)
                
                ## Figure for model ensemble disagreement
                plt.plot(range(len(pred_traj[:,dim])), pred_traj[:,dim], 'k-')

                ## Add the lines for each pred error
                for t in range(len(pred_traj)):
                    x = [t, t]
                    y = [pred_traj[t, dim], pred_traj[t, dim] + pred_error[t, dim]]
                    plt.plot(x, y, 'g')
                
                ## Add the pred error for each step
                ## Set plot title
                plt.title(f"{self._n} step model ensemble prediction error along {traj_type} trajectories on dimension {dim}\n{init_name} on {num_episodes} episodes\n{label}")
                ## Save fig
                fig_name = f"{i}_{self._n}_step_trajectories_{label}_pred_error_{traj_type}_dim_{dim}"
                plt.savefig(f"{fig_path_pred_error}/{fig_name}", bbox_inches='tight')

                plt.close()
                ### Model Ensemble Disagreement ###

                ## Create fig and ax
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ## Prepare plot
                labels = ['Number of steps on environment',
                          f'Trajectory on dimension {dim} on label {label}']
                limits = [0, len(pred_traj[:,dim]),
                          min(min(pred_traj[:, dim]), min(pred_traj[:, dim]+disagr[:])),
                          max(max(pred_traj[:, dim]), max(pred_traj[:,dim]+disagr[:]))]
                self.prepare_plot(plt, fig, ax, mode='2d', limits=limits, ax_labels=labels)
                
                ## Figure for model ensemble disagreement
                plt.plot(range(len(pred_traj[:,dim])), pred_traj[:,dim], 'k-')

                ## Add the lines for each pred error
                for t in range(len(pred_traj)):
                    x = [t, t]
                    y = [pred_traj[t, dim], pred_traj[t, dim]+disagr[t]]
                    plt.plot(x, y, 'g')
                    
                ## Add the pred error for each step
                ## Set plot title
                plt.title(f"{self._n} step model ensemble disagreement along {traj_type} trajectories on dimension {dim}\n{init_name} on {num_episodes} episodes\n{label}")
                ## Save fig
                fig_name = f"{i}_{self._n}_step_trajectories_{label}_disagr_{traj_type}_dim_{dim}"
                plt.savefig(f"{fig_path_disagr}/{fig_name}", bbox_inches='tight')

                plt.close()
            
        if show:
            plt.show()
        plt.close()

        return pred_trajs, disagrs, pred_errors
