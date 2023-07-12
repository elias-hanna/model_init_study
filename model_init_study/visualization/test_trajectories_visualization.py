import numpy as np
import copy
import os
import matplotlib.pyplot as plt
import tqdm

from model_init_study.visualization.visualization import VisualizationMethod

class TestTrajectoriesVisualization(VisualizationMethod):
    def __init__(self, params):
        super().__init__(params=params)
        self._process_params(params)
        self._pred_error_thresh = .03 # 3 cm
        self._n = 1
        self.ctrl_input = 'obs'
        self.ctrl_type = 'nn'
        
    def _process_params(self, params):
        super()._process_params(params)
        if params['separator'] is not None:
            self.traj_separator = params['separator']()
        if 'path_to_test_trajectories' in params:
            self.test_trajectories = np.load(params['path_to_test_trajectories'])['examples']
            self.test_params = np.load(params['path_to_test_trajectories'])['params']
            ## /!\ Warning, trajs must be of shape (nb_of_trajs, nb_of_steps, state_dim)
        else:
            # raise Exception('TestTrajectoriesVisualization _process_params error: path_to_test_trajectories not in params')
            print('WARNING: TestTrajectoriesVisualization _process_params error: path_to_test_trajectories not in params')
        if 'env_max_h' in params:
            self.env_max_h = params['env_max_h']
        else:
            raise Exception('TestTrajectoriesVisualization _process_params error: env_max_h not in params')
        if 'controller_type' in params:
            ## Associate an instance of controller_type with given params
            self.controller = params['controller_type'](params=params)
        else:
            # raise Exception('ExplorationMethod _process_params error: controller_type not in params')
            print('WARNING: TestTrajectoriesVisualization _process_params error: controller_type not in params')
            print('WARNING: You have to set controller manually through NStepErrorVisualization.set_controller(controller)')
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

    def set_controller(self, controller, actions_lists=None, ctrl_type='actions_list', ctrl_input='time'):
        self.ctrl_type = ctrl_type
        self.ctrl_input = ctrl_input
        self.controller = controller
        self.test_params = actions_lists
    
    def _execute_test_trajectories_on_model(self):
        controller_list = []

        traj_list = []
        disagreements_list = []
        prediction_errors_list = []

        pred_trajs = np.empty((len(self.test_trajectories), self.env_max_h, self._obs_dim))
        pred_trajs[:] = np.nan
        disagrs = np.empty((len(self.test_trajectories), self.env_max_h))
        disagrs[:] = np.nan
        pred_errors = np.empty((len(self.test_trajectories), self.env_max_h))
        pred_errors[:] = np.nan
        
        A = np.empty((len(self.test_trajectories), self._action_dim))
        S = np.empty((len(self.test_trajectories), self._obs_dim))
        
        for i in range(len(self.test_trajectories)):
            if self.ctrl_type == 'actions_list':
                controller_list.append(self.test_params[i]) ## append act list
            elif self.ctrl_type ==  'nn':
                ## Create a copy of the controller
                controller_list.append(self.controller.copy())
                ## Set controller parameters
                controller_list[-1].set_parameters(self.test_params[i])
                ## Init starting state
            S[i,:] = self.test_trajectories[i,0,:]

        ended = [False] * len(self.test_trajectories)

        # for i in range(self.env_max_h-1):
        for i in tqdm.tqdm(range(self.env_max_h-1), total=self.env_max_h-1):
            for j in range(len(self.test_trajectories)):
                # S[j,:] = self.test_trajectories[j,i,:]
                if self.ctrl_type == 'actions_list':
                    A[j,:] = controller_list[j][i]
                elif self.ctrl_type ==  'nn':
                    A[j,:] = controller_list[j](S[j,:])
                if np.isnan(self.test_trajectories[j,i,:]).any():
                    ended[j] = True
                    
            if all(ended):
                break
            
            batch_pred_delta_ns, batch_disagreement = self.model.forward_multiple(A, S,
                                                                                  mean=True,
                                                                                  disagr=True)
            for j in range(len(self.test_trajectories)):
                if ended[j]:
                    continue
                ## Compute mean prediction from model samples
                next_step_pred = batch_pred_delta_ns[j]
                mean_pred = [np.mean(next_step_pred[:,k]) for k in range(len(next_step_pred[0]))]
                pred_trajs[j,i,:] = S[j,:]

                if self.env_name in ['cartpole', 'pusher', 'reacher']:
                    S[j,:] = mean_pred.copy()
                else:
                    S[j,:] += mean_pred.copy()
                # pred_trajs[j,i,:] = mean_pred.copy()
                # if self.ctrl_type == 'nn':
                    ## DO THIS WHEN TENSORS rets (legacy MIS)
                    # disagrs[j,i] = np.mean(batch_disagreement[j].detach().numpy())
                # else:
                disagrs[j,i] = np.mean(batch_disagreement[j])
                pred_errors[j,i] = np.linalg.norm(S[j,:]-self.test_trajectories[j,i,:])

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
        run_name = '{}_{}_{}'.format(env_name, init_name, num_episodes)
        fig_path = os.path.join(self.dump_path, '{}/disagr'.format(run_name))
        os.makedirs(fig_path, exist_ok=True)

        print('Current working dir: {}'.format(os.getcwd()))
        print('Test trajectories vis dumping figs on {}'.format(fig_path))
        
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
        if plot_stddev:
            plt.fill_between(range(len(mean_disagr)),
                             mean_disagr-std_disagr,
                             mean_disagr+std_disagr,
                             facecolor='green', alpha=0.5)
        ## Set plot title
        plt.title("Mean model ensemble disagreeement along {} trajectories \n{} on {} episodes\n{}".format(traj_type, init_name, num_episodes, label))
        ## Save fig
        plt.savefig("{}/mean_test_trajectories_{}_disagr_{}".format(fig_path, label, traj_type),
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
                ## Set plot title
                plt.title("Mean model ensemble disagreeement along single {} trajectory\n{} on {} episodes\n{}".format(traj_type, init_name, num_episodes, label))
                ## Save fig
                plt.savefig("{}/{}_test_trajectories_{}_disagr_{}".format(fig_path, i, label, traj_type),
                            bbox_inches='tight')

        ## Make dump dirs
        fig_path = os.path.join(self.dump_path, '{}/pred_error'.format(run_name))
        os.makedirs(fig_path, exist_ok=True)        
            
        ## Create fig and ax
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ## Prepare plot
        labels = ['Number of steps on environment', 'Mean prediction error']
        limits = [0, len(mean_pred_error),
                  min(mean_pred_error-std_pred_error), max(mean_pred_error+std_pred_error)]
        self.prepare_plot(plt, fig, ax, mode='2d', limits=limits, ax_labels=labels)

        ## Figure for prediction error
        plt.plot(range(len(mean_pred_error)), mean_pred_error, 'k-')
        
        if plot_stddev:
            plt.fill_between(range(len(mean_pred_error)),
                             mean_pred_error-std_pred_error,
                             mean_pred_error+std_pred_error,
                             facecolor='green', alpha=0.5)
        ## Set plot title
        plt.title("Mean prediction error along {} trajectories\n{} on {} episodes\n{}".format(traj_type, init_name, num_episodes, label))
        ## Save fig
        plt.savefig("{}/mean_test_trajectories_{}_pred_error_{}".format(fig_path, label, traj_type),
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
                ## Set plot title
                plt.title("Mean prediction error along single {} trajectory\n{} on {} episodes\n{}".format(traj_type, init_name, num_episodes, label))
                ## Save fig
                plt.savefig("{}/{}_test_trajectories_{}_pred_error_{}".format(fig_path, i, label, traj_type),
                            bbox_inches='tight')

        if show:
            plt.show()
        plt.close()

        return pred_trajs, disagrs, pred_errors
