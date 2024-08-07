## Call this piece of code from the top folder containing all env data ##
if __name__ == '__main__':
    # Local imports
    from model_init_study.visualization.test_trajectories_visualization \
        import TestTrajectoriesVisualization

    from model_init_study.models.dynamics_model \
        import DynamicsModel

    from model_init_study.controller.nn_controller \
        import NeuralNetworkController

    # Env imports
    import gym
    import diversity_algorithms.environments.env_imports ## Contains deterministic ant + fetch
    import mb_ge ## Contains ball in cup
    import redundant_arm ## contains redundant arm
    
    # Utils imports
    import numpy as np
    import os
    import argparse
    import model_init_study
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import matplotlib

    # cap_val = 50
    stop_hor = -1
    
    module_path = os.path.dirname(model_init_study.__file__)

    parser = argparse.ArgumentParser(description='Process run parameters.')

    parser.add_argument('--init-methods', nargs="*",
                        type=str, default=['brownian-motion', 'levy-flight', 'colored-noise-beta-1', 'colored-noise-beta-2', 'random-actions'])

    parser.add_argument('--init-episodes', nargs="*",
                        type=int, default=[5, 10, 20])

    parser.add_argument('--disagr-plot-upper-lim', type=float, default=5.)
    parser.add_argument('--pred-err-plot-upper-lim', type=float, default=5.)
    
    parser.add_argument('--dump-path', type=str, default='default_dump/')

    parser.add_argument('--environment', '-e', type=str, default='ball_in_cup')

    args = parser.parse_args()

    dynamics_model = DynamicsModel

    env_name = args.environment

    env_register_id = 'BallInCup3d-v0'
    gym_args = {}
    is_pets_env = False
    if args.environment == 'ball_in_cup':
        env_register_id = 'BallInCup3d-v0'
        ss_min = -0.4
        ss_max = 0.4
    elif args.environment == 'redundant_arm':
        env_register_id = 'RedundantArmPos-v0'
        ss_min = -1
        ss_max = 1
    elif args.environment == 'redundant_arm_no_walls':
        env_register_id = 'RedundantArmPosNoWalls-v0'
        ss_min = -1
        ss_max = 1
    elif args.environment == 'redundant_arm_no_walls_no_collision':
        env_register_id = 'RedundantArmPosNoWallsNoCollision-v0'
        ss_min = -1
        ss_max = 1
    elif args.environment == 'redundant_arm_no_walls_limited_angles':
        env_register_id = 'RedundantArmPosNoWallsLimitedAngles-v0'
        ss_min = -1
        ss_max = 1
        gym_args['dof'] = 100
    elif args.environment == 'fastsim_maze':
        env_register_id = 'FastsimSimpleNavigationPos-v0'
        ss_min = -10
        ss_max = 10
    elif args.environment == 'fastsim_maze_traps':
        env_register_id = 'FastsimSimpleNavigationPos-v0'
        ss_min = -10
        ss_max = 10
        gym_args['physical_traps'] = True
    elif args.environment == 'fetch_pick_and_place':
        env_register_id = 'FetchPickAndPlaceDeterministic-v1'
        ss_min = -1
        ss_max = 1
    elif args.environment == 'ant':
        env_register_id = 'AntBulletEnvDeterministicPos-v0'
        ss_min = -10
        ss_max = 10
    elif args.environment == 'cartpole':
        is_pets_env = True
        max_step = 200
        obs_dim = 4
        act_dim = 1
        ss_min = -10
        ss_max = 10
    elif args.environment == 'pusher':
        is_pets_env = True
        max_step = 150
        obs_dim = 20
        act_dim = 7
        ss_min = -10
        ss_max = 10
    elif args.environment == 'reacher':
        is_pets_env = True
        max_step = 150
        obs_dim = 17
        act_dim = 7
        ss_min = -10
        ss_max = 10
    else:
        raise ValueError(f"{args.environment} is not a defined environment")

    if not is_pets_env:
        env = gym.make(env_register_id, **gym_args)

        try:
            max_step = env._max_episode_steps
        except:
            try:
                max_step = env.max_steps
            except:
                raise AttributeError("Env does not allow access to _max_episode_steps or to max_steps")

        path_to_examples = os.path.join(module_path,
                                        'examples/',
                                        args.environment+'_example_trajectories.npz')
        obs = env.reset()
        if isinstance(obs, dict):
            obs_dim = env.observation_space['observation'].shape[0]
        else:
            obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

    rep_folders = next(os.walk(f'.'))[1]

    rep_folders = [x for x in rep_folders if (x.isdigit())]

    rep_data = np.load(f'{rep_folders[0]}/{args.environment}_{args.init_methods[0]}_{args.init_episodes[0]}_data.npz')

    tmp_data = rep_data['test_pred_trajs']
    ## Get parameters (should do a params.npz... cba)
    # trajs_per_rep = len(tmp_data)
    trajs_per_rep = 10
    n_total_trajs = trajs_per_rep*len(rep_folders)
    task_h = len(tmp_data[0])

    n_init_method = len(args.init_methods) # 2
    init_methods = args.init_methods #['random-policies', 'random-actions']
    n_init_episodes = len(args.init_episodes) # 4
    init_episodes = args.init_episodes #[5, 10, 15, 20]
    
    test_pred_trajs = np.empty((n_init_method, n_init_episodes, n_total_trajs, task_h, obs_dim))
    test_disagrs = np.empty((n_init_method, n_init_episodes, n_total_trajs, task_h))
    test_pred_errors = np.empty((n_init_method, n_init_episodes, n_total_trajs, task_h))

    example_pred_trajs = np.empty((n_init_method, n_init_episodes, n_total_trajs, task_h, obs_dim))
    example_disagrs = np.empty((n_init_method, n_init_episodes, n_total_trajs, task_h))
    example_pred_errors = np.empty((n_init_method, n_init_episodes, n_total_trajs, task_h))

    example_1_step_trajs = np.empty((n_total_trajs, task_h, obs_dim))
    example_1_step_disagrs = np.empty((n_total_trajs, task_h))
    example_1_step_pred_errors = np.empty((n_total_trajs, task_h, obs_dim))

    ## Stores all pred errors
    all_example_1_step_pred_errors = np.empty((n_init_method, n_total_trajs,
                                               task_h, obs_dim))
    norm_all_example_1_step_pred_errors = np.empty((n_init_method, n_total_trajs,
                                                    task_h, obs_dim))
    
    # example_5_step_trajs = np.empty((n_total_trajs, task_h, obs_dim))
    # example_5_step_disagrs = np.empty((n_total_trajs, task_h))
    # example_5_step_pred_errors = np.empty((n_total_trajs, task_h, obs_dim))

    # example_10_step_trajs = np.empty((n_total_trajs, task_h, obs_dim))
    # example_10_step_disagrs = np.empty((n_total_trajs, task_h))
    # example_10_step_pred_errors = np.empty((n_total_trajs, task_h, obs_dim))

    example_20_step_trajs = np.empty((n_total_trajs, task_h, obs_dim))
    example_20_step_disagrs = np.empty((n_total_trajs, task_h))
    example_20_step_pred_errors = np.empty((n_total_trajs, task_h, obs_dim))

    norm_example_20_step_pred_errors = np.empty((n_total_trajs, task_h, obs_dim))

    ## Do a new plot
    ## Will contain all means on a single plot
    # Create 4 figs
    test_fig_disagr = plt.figure()
    test_ax_disagr = test_fig_disagr.add_subplot(111)

    test_fig_pred_error = plt.figure()
    test_ax_pred_error = test_fig_pred_error.add_subplot(111)

    test_fig_pred_disagr = plt.figure()
    test_ax_pred_disagr = test_fig_pred_disagr.add_subplot(111)

    example_fig_disagr = plt.figure()
    example_ax_disagr = example_fig_disagr.add_subplot(111)

    example_fig_pred_error = plt.figure()
    example_ax_pred_error = example_fig_pred_error.add_subplot(111)

    example_fig_pred_disagr = plt.figure()
    example_ax_pred_disagr = example_fig_pred_disagr.add_subplot(111)
    # Init limits for each fig
    test_limits_disagr = [0, max_step,
                          0, 0]
    test_limits_pred_error = [0, max_step,
                              0, 0]
    test_limits_pred_disagr = [0, max_step,
                               0, 0]
    example_limits_disagr = [0, max_step,
                             0, 0]
    example_limits_pred_error = [0, max_step,
                                 0, 0]
    example_limits_pred_disagr = [0, max_step,
                                  0, 0]

    # Init labels for each fig
    test_labels_disagr = ['Number of steps on environment', 'Mean ensemble disagreement']
    test_labels_pred_error = ['Number of steps on environment', 'Mean prediction error']
    test_labels_pred_disagr = ['Mean ensemble disagreement', 'Mean prediction error']
    example_labels_disagr = ['Number of steps on environment', 'Mean ensemble disagreement']
    example_labels_pred_error = ['Number of steps on environment', 'Mean prediction error']
    example_labels_pred_disagr = ['Mean ensemble disagreement', 'Mean prediction error']

    ## Colors for plots
    # evenly_spaced_interval = np.linspace(0, 1, n_init_method*n_init_episodes+1)
    # colors = [cm.rainbow(x) for x in evenly_spaced_interval]

    # colors = plt.cm.get_cmap('hsv', n_init_method*n_init_episodes+1)
    # cmap = plt.cm.get_cmap('hsv', n_init_method*n_init_episodes+1)
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=n_init_method*n_init_episodes+1)
    cmap = plt.cm.get_cmap('hsv', n_init_method+1)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=n_init_method+1)

    colors = cm.ScalarMappable(norm=norm, cmap=cmap)

    ## Linestyles for plots

    linestyles = ['-', '--', ':', '-.',
                  (0, (5, 10)), (0, (5, 1)), (0, (3, 10, 1, 10)), (0, (3, 1, 1, 1))]
    
    ## Create a numpy array to store the mean and stddev of predicition errors for each method
    ## and ways of predicting
    # pred_steps = [1, 5, 10, 20, max_step]
    pred_steps = [1, 20, max_step]

    ## The mean pred errors are only computed on example trajectories! (we don't have the n step
    ## average data for test trajectories, could add a database of test trajectories to do
    ## testing tho...)
    mean_pred_errors = np.zeros((n_init_method, n_init_episodes, len(pred_steps)))
    std_pred_errors = np.zeros((n_init_method, n_init_episodes, len(pred_steps)))
    ## The mean pred errors are only computed on example trajectories! (we don't have the n step
    ## average data for test trajectories, could add a database of test trajectories to do
    ## testing tho...)
    norm_mean_pred_errors = np.zeros((n_init_method, n_init_episodes, len(pred_steps)))
    norm_std_pred_errors = np.zeros((n_init_method, n_init_episodes, len(pred_steps)))
    
    ## Plot table with mean prediction error for n step predictions    
    column_headers = [init_method for init_method in init_methods]
    row_headers = [init_episode for init_episode in init_episodes]
    cell_text_1_step = [["" for _ in range(len(column_headers))] for _ in range(len(row_headers))]
    cell_text_20_step = [["" for _ in range(len(column_headers))] for _ in range(len(row_headers))]
    cell_text_full = [["" for _ in range(len(column_headers))] for _ in range(len(row_headers))]
    rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))


    # for i in range(n_init_method):
        # init_method =  init_methods[i]
    for j in range(n_init_episodes):
        init_episode = init_episodes[j]
        # for j in range(n_init_episodes):
            # init_episode = init_episodes[j]
        for i in range(n_init_method):
            init_method =  init_methods[i]
            rep_cpt = 0
            for rep_path in rep_folders:
                rep_data = np.load(f'{rep_path}/{args.environment}_{init_method}_{init_episode}_data.npz')

                # test_pred_trajs[i,j,rep_cpt*trajs_per_rep:
                #                 rep_cpt*trajs_per_rep
                #                 + trajs_per_rep] =rep_data['test_pred_trajs']
                # test_disagrs[i,j,rep_cpt*trajs_per_rep:
                #              rep_cpt*trajs_per_rep
                #              + trajs_per_rep] = rep_data['test_disagrs']
                # test_pred_errors[i,j,rep_cpt*trajs_per_rep:
                #                  rep_cpt*trajs_per_rep
                #                  + trajs_per_rep] = rep_data['test_pred_errors']
                
                example_pred_trajs[i,j,rep_cpt*trajs_per_rep:
                                   rep_cpt*trajs_per_rep
                                   + trajs_per_rep] = rep_data['examples_pred_trajs']
                example_disagrs[i,j,rep_cpt*trajs_per_rep:
                                rep_cpt*trajs_per_rep
                                + trajs_per_rep] = rep_data['examples_disagrs']
                example_pred_errors[i,j,rep_cpt*trajs_per_rep:
                                    rep_cpt*trajs_per_rep
                                    + trajs_per_rep] = rep_data['examples_pred_errors']
                
                ### For N step vis ###
                example_1_step_trajs[rep_cpt*trajs_per_rep:
                                     rep_cpt*trajs_per_rep
                                     + trajs_per_rep] = rep_data['examples_1_step_trajs'][:,:-1,:]
                example_1_step_disagrs[rep_cpt*trajs_per_rep:
                                       rep_cpt*trajs_per_rep
                                       + trajs_per_rep] = rep_data['examples_1_step_disagrs']
                example_1_step_pred_errors[rep_cpt*trajs_per_rep:
                                           rep_cpt*trajs_per_rep
                                           + trajs_per_rep] = rep_data['examples_1_step_pred_errors']

                all_example_1_step_pred_errors[i, rep_cpt*trajs_per_rep:
                                               rep_cpt*trajs_per_rep
                                               + trajs_per_rep] = \
                                        rep_data['examples_1_step_pred_errors']
                # example_5_step_trajs[rep_cpt*trajs_per_rep:
                #                      rep_cpt*trajs_per_rep
                #                      + trajs_per_rep] = rep_data['examples_5_step_trajs']
                # example_5_step_disagrs[rep_cpt*trajs_per_rep:
                #                        rep_cpt*trajs_per_rep
                #                        + trajs_per_rep] = rep_data['examples_5_step_disagrs']
                # example_5_step_pred_errors[rep_cpt*trajs_per_rep:
                #                            rep_cpt*trajs_per_rep
                #                            + trajs_per_rep] = rep_data['examples_5_step_pred_errors']

                # example_10_step_trajs[rep_cpt*trajs_per_rep:
                #                      rep_cpt*trajs_per_rep
                #                      + trajs_per_rep] = rep_data['examples_10_step_trajs']
                # example_10_step_disagrs[rep_cpt*trajs_per_rep:
                #                        rep_cpt*trajs_per_rep
                #                        + trajs_per_rep] = rep_data['examples_10_step_disagrs']
                # example_10_step_pred_errors[rep_cpt*trajs_per_rep:
                #                            rep_cpt*trajs_per_rep
                #                            + trajs_per_rep] = rep_data['examples_10_step_pred_errors']

                if not is_pets_env:
                    example_20_step_trajs[rep_cpt*trajs_per_rep:
                                          rep_cpt*trajs_per_rep
                                          + trajs_per_rep] = rep_data['examples_20_step_trajs']
                    example_20_step_disagrs[rep_cpt*trajs_per_rep:
                                            rep_cpt*trajs_per_rep
                                            + trajs_per_rep] = rep_data['examples_20_step_disagrs']
                    example_20_step_pred_errors[rep_cpt*trajs_per_rep:
                                                rep_cpt*trajs_per_rep
                                                + trajs_per_rep] = rep_data['examples_20_step_pred_errors']
                else:
                    example_20_step_trajs[rep_cpt*trajs_per_rep:
                                          rep_cpt*trajs_per_rep
                                          + trajs_per_rep] = rep_data['examples_plan_h_step_trajs'][:,:-1,:]
                    example_20_step_disagrs[rep_cpt*trajs_per_rep:
                                            rep_cpt*trajs_per_rep
                                            + trajs_per_rep] = rep_data['examples_plan_h_step_disagrs']
                    example_20_step_pred_errors[rep_cpt*trajs_per_rep:
                                                rep_cpt*trajs_per_rep
                                                + trajs_per_rep] = rep_data['examples_plan_h_step_pred_errors']
                    
                rep_cpt += 1
            
            ## Mean pred error on full length recursive prediction on example trajs
            pred_error_vals = []

            for pred_errors in example_pred_errors[i,j]:
                max_non_nan_idx = (~np.isnan(pred_errors)).cumsum().argmax()
                pred_error_vals.append(pred_errors[max_non_nan_idx])

            # mean_pred_errors[i, j, 4] = np.nanmean(pred_error_vals)
            # std_pred_errors[i, j, 4] = np.nanstd(pred_error_vals)


            abs_full_pred_errors = np.absolute(pred_error_vals)
            # abs_20_step_pred_errors = abs_20_step_pred_errors[abs_20_step_pred_errors < 20]
            # abs_full_pred_errors[abs_full_pred_errors > cap_val] = cap_val

            # mean_pred_errors[i, j, 2] = np.nanmean(np.absolute(pred_error_vals))
            # std_pred_errors[i, j, 2] = np.nanstd(np.absolute(pred_error_vals))
            mean_pred_errors[i, j, 2] = np.nanmean(abs_full_pred_errors[:stop_hor])
            std_pred_errors[i, j, 2] = np.nanstd(abs_full_pred_errors[:stop_hor])

            # cell_text[i][j] = f"\u0394 s = {mean_ds} \u00B1 {std_ds}"
            cell_text_full[j][i] = f"{round(mean_pred_errors[i,j,2],1)} \u00B1 {round(std_pred_errors[i,j,2],1)}"
            #### N step Disagreement and Prediction Error ####
            #### We only do it on example trajs as test trajs changes... between reps ####

            run_name = f'{env_name}_{init_method}_{init_episode}'
            fig_path_disagr = os.path.join(args.dump_path, f'{run_name}/disagr')
            os.makedirs(fig_path_disagr, exist_ok=True)
            
            fig_path_pred_error = os.path.join(args.dump_path, f'{run_name}/pred_error')
            os.makedirs(fig_path_pred_error, exist_ok=True)
        
            ## n = 1

            mean_pred_errors[i, j, 0] = np.nanmean(np.absolute(example_1_step_pred_errors[:stop_hor]))
            std_pred_errors[i, j, 0] = np.nanstd(np.absolute(example_1_step_pred_errors[:stop_hor]))
            cell_text_1_step[j][i] = f"{round(mean_pred_errors[i,j,0],3)} \u00B1 {round(std_pred_errors[i,j,0],3)}"

            # for r in range(trajs_per_rep):
            #     pred_trajs = example_1_step_trajs[r::2]
            #     pred_errors = example_1_step_pred_errors[r::2]
            #     disagrs = example_1_step_disagrs[r::2]

            #     pred_traj = np.nanmean(pred_trajs, axis=0)
            #     pred_error = np.nanmean(pred_errors, axis=0)
            #     disagr = np.nanmean(disagrs, axis=0)

            #     n = 1
            #     for dim in range(pred_traj.shape[1]):
            #         ### Model prediction error ###
                    
            #         ## Create fig and ax
            #         fig = plt.figure()
            #         ax = fig.add_subplot(111)
            #         ## Prepare plot
            #         labels = ['Number of steps on environment',
            #                   f'Trajectory on dimension {dim}']
            #         limits = [0, len(pred_traj[:,dim]),
            #                   min(min(pred_traj[:, dim]),
            #                       min(pred_traj[:, dim]+pred_error[:, dim])),
            #                   max(max(pred_traj[:, dim]),
            #                       max(pred_traj[:, dim]+pred_error[:, dim]))]

            #         ## Set plot labels
            #         ax.set_xlabel(labels[0])
            #         ax.set_ylabel(labels[1])
                    
            #         ## Set plot limits
            #         x_min = limits[0]; x_max = limits[1]; y_min = limits[2]; y_max = limits[3]
                    
            #         ax.set_xlim(x_min,x_max)
            #         ax.set_ylim(y_min,y_max)

            #         ## Figure for model ensemble disagreement
            #         plt.plot(range(len(pred_traj[:,dim])), pred_traj[:,dim], 'k-')
                    
            #         ## Add the lines for each pred error
            #         for t in range(len(pred_traj)):
            #             x = [t, t]
            #             y = [pred_traj[t, dim], pred_traj[t, dim] + pred_error[t, dim]]
            #             plt.plot(x, y, 'g')
                
            #         ## Add the pred error for each step
            #         ## Set plot title
            #         plt.title(f"{n} step model ensemble prediction error along example trajectories on dimension {dim}\n{init_method} on {init_episode} episodes\n{args.environment}")
            #         ## Save fig
            #         fig_name = f"{i}_{n}_step_trajectories_{args.environment}_pred_error_example_dim_{dim}"
            #         plt.savefig(f"{fig_path_pred_error}/{fig_name}",
            #                     bbox_inches='tight')
                    
            #         plt.close()
            #         ### Model Ensemble Disagreement ###
                    
            #         ## Create fig and ax
            #         fig = plt.figure()
            #         ax = fig.add_subplot(111)
            #         ## Prepare plot
            #         labels = ['Number of steps on environment',
            #                   f'Trajectory on dimension {dim}']
            #         limits = [0, len(pred_traj[:,dim]),
            #                   min(min(pred_traj[:, dim]), min(pred_traj[:, dim]+disagr[:])),
            #                   max(max(pred_traj[:, dim]), max(pred_traj[:, dim]+disagr[:]))]

            #         ## Set plot labels
            #         ax.set_xlabel(labels[0])
            #         ax.set_ylabel(labels[1])
                    
            #         ## Set plot limits
            #         x_min = limits[0]; x_max = limits[1]; y_min = limits[2]; y_max = limits[3]
                    
            #         ax.set_xlim(x_min,x_max)
            #         ax.set_ylim(y_min,y_max)

            #         ## Figure for model ensemble disagreement
            #         plt.plot(range(len(pred_traj[:,dim])), pred_traj[:,dim], 'k-')
                    
            #         ## Add the lines for each pred error
            #         for t in range(len(pred_traj)):
            #             x = [t, t]
            #             y = [pred_traj[t, dim], pred_traj[t, dim]+disagr[t]]
            #             plt.plot(x, y, 'g')
                        
            #         ## Add the pred error for each step
            #         ## Set plot title
            #         plt.title(f"{n} step model ensemble disagreement along example trajectories on dimension {dim}\n{init_method} on {init_episode} episodes\n{args.environment}")
            #         ## Save fig
            #         fig_name = f"{i}_{n}_step_trajectories_{args.environment}_disagr_example_dim_{dim}"
            #         plt.savefig(f"{fig_path_disagr}/{fig_name}",
            #                     bbox_inches='tight')
                    
            #         plt.close()
        
            ## n = 5
            
            # mean_pred_errors[i, j, 1] = np.nanmean(example_5_step_pred_errors)
            
            # for r in range(trajs_per_rep):
            #     pred_trajs = example_5_step_trajs[r::2]
            #     pred_errors = example_5_step_pred_errors[r::2]
            #     disagrs = example_5_step_disagrs[r::2]

            #     pred_traj = np.nanmean(pred_trajs, axis=0)
            #     pred_error = np.nanmean(pred_errors, axis=0)
            #     disagr = np.nanmean(disagrs, axis=0)

            #     n = 5
            #     for dim in range(pred_traj.shape[1]):
            #         ### Model prediction error ###
                    
            #         ## Create fig and ax
            #         fig = plt.figure()
            #         ax = fig.add_subplot(111)
            #         ## Prepare plot
            #         labels = ['Number of steps on environment',
            #                   f'Trajectory on dimension {dim}']
            #         limits = [0, len(pred_traj[:,dim]),
            #                   min(min(pred_traj[:, dim]),
            #                       min(pred_traj[:, dim]+pred_error[:, dim])),
            #                   max(max(pred_traj[:, dim]),
            #                       max(pred_traj[:, dim]+pred_error[:, dim]))]

            #         ## Set plot labels
            #         ax.set_xlabel(labels[0])
            #         ax.set_ylabel(labels[1])
                    
            #         ## Set plot limits
            #         x_min = limits[0]; x_max = limits[1]; y_min = limits[2]; y_max = limits[3]
                    
            #         ax.set_xlim(x_min,x_max)
            #         ax.set_ylim(y_min,y_max)

            #         ## Figure for model ensemble disagreement
            #         plt.plot(range(len(pred_traj[:,dim])), pred_traj[:,dim], 'k-')
                    
            #         ## Add the lines for each pred error
            #         for t in range(len(pred_traj)):
            #             x = [t, t]
            #             y = [pred_traj[t, dim], pred_traj[t, dim] + pred_error[t, dim]]
            #             plt.plot(x, y, 'g')
                
            #         ## Add the pred error for each step
            #         ## Set plot title
            #         plt.title(f"{n} step model ensemble prediction error along example trajectories on dimension {dim}\n{init_method} on {init_episode} episodes\n{args.environment}")
            #         ## Save fig
            #         fig_name = f"{i}_{n}_step_trajectories_{args.environment}_pred_error_example_dim_{dim}"
            #         plt.savefig(f"{fig_path_pred_error}/{fig_name}",
            #                     bbox_inches='tight')
                    
            #         plt.close()
            #         ### Model Ensemble Disagreement ###
                    
            #         ## Create fig and ax
            #         fig = plt.figure()
            #         ax = fig.add_subplot(111)
            #         ## Prepare plot
            #         labels = ['Number of steps on environment',
            #                   f'Trajectory on dimension {dim}']
            #         limits = [0, len(pred_traj[:,dim]),
            #                   min(min(pred_traj[:, dim]), min(pred_traj[:, dim]+disagr[:])),
            #                   max(max(pred_traj[:, dim]), max(pred_traj[:, dim]+disagr[:]))]

            #         ## Set plot labels
            #         ax.set_xlabel(labels[0])
            #         ax.set_ylabel(labels[1])
                    
            #         ## Set plot limits
            #         x_min = limits[0]; x_max = limits[1]; y_min = limits[2]; y_max = limits[3]
                    
            #         ax.set_xlim(x_min,x_max)
            #         ax.set_ylim(y_min,y_max)

            #         ## Figure for model ensemble disagreement
            #         plt.plot(range(len(pred_traj[:,dim])), pred_traj[:,dim], 'k-')
                    
            #         ## Add the lines for each pred error
            #         for t in range(len(pred_traj)):
            #             x = [t, t]
            #             y = [pred_traj[t, dim], pred_traj[t, dim]+disagr[t]]
            #             plt.plot(x, y, 'g')
                        
            #         ## Add the pred error for each step
            #         ## Set plot title
            #         plt.title(f"{n} step model ensemble disagreement along example trajectories on dimension {dim}\n{init_method} on {init_episode} episodes\n{args.environment}")
            #         ## Save fig
            #         fig_name = f"{i}_{n}_step_trajectories_{args.environment}_disagr_example_dim_{dim}"
            #         plt.savefig(f"{fig_path_disagr}/{fig_name}",
            #                     bbox_inches='tight')
                    
            #         plt.close()

            # ## n = 10
            
            # mean_pred_errors[i, j, 2] = np.nanmean(example_10_step_pred_errors)

            # for r in range(trajs_per_rep):
            #     pred_trajs = example_10_step_trajs[r::2]
            #     pred_errors = example_10_step_pred_errors[r::2]
            #     disagrs = example_10_step_disagrs[r::2]

            #     pred_traj = np.nanmean(pred_trajs, axis=0)
            #     pred_error = np.nanmean(pred_errors, axis=0)
            #     disagr = np.nanmean(disagrs, axis=0)

            #     n = 10
            #     for dim in range(pred_traj.shape[1]):
            #         ### Model prediction error ###
                    
            #         ## Create fig and ax
            #         fig = plt.figure()
            #         ax = fig.add_subplot(111)
            #         ## Prepare plot
            #         labels = ['Number of steps on environment',
            #                   f'Trajectory on dimension {dim}']
            #         limits = [0, len(pred_traj[:,dim]),
            #                   min(min(pred_traj[:, dim]),
            #                       min(pred_traj[:, dim]+pred_error[:, dim])),
            #                   max(max(pred_traj[:, dim]),
            #                       max(pred_traj[:, dim]+pred_error[:, dim]))]

            #         ## Set plot labels
            #         ax.set_xlabel(labels[0])
            #         ax.set_ylabel(labels[1])
                    
            #         ## Set plot limits
            #         x_min = limits[0]; x_max = limits[1]; y_min = limits[2]; y_max = limits[3]
                    
            #         ax.set_xlim(x_min,x_max)
            #         ax.set_ylim(y_min,y_max)

            #         ## Figure for model ensemble disagreement
            #         plt.plot(range(len(pred_traj[:,dim])), pred_traj[:,dim], 'k-')
                    
            #         ## Add the lines for each pred error
            #         for t in range(len(pred_traj)):
            #             x = [t, t]
            #             y = [pred_traj[t, dim], pred_traj[t, dim] + pred_error[t, dim]]
            #             plt.plot(x, y, 'g')
                
            #         ## Add the pred error for each step
            #         ## Set plot title
            #         plt.title(f"{n} step model ensemble prediction error along example trajectories on dimension {dim}\n{init_method} on {init_episode} episodes\n{args.environment}")
            #         ## Save fig
            #         fig_name = f"{i}_{n}_step_trajectories_{args.environment}_pred_error_example_dim_{dim}"
            #         plt.savefig(f"{fig_path_pred_error}/{fig_name}",
            #                     bbox_inches='tight')
                    
            #         plt.close()
            #         ### Model Ensemble Disagreement ###
                    
            #         ## Create fig and ax
            #         fig = plt.figure()
            #         ax = fig.add_subplot(111)
            #         ## Prepare plot
            #         labels = ['Number of steps on environment',
            #                   f'Trajectory on dimension {dim}']
            #         limits = [0, len(pred_traj[:,dim]),
            #                   min(min(pred_traj[:, dim]), min(pred_traj[:, dim]+disagr[:])),
            #                   max(max(pred_traj[:, dim]), max(pred_traj[:, dim]+disagr[:]))]

            #         ## Set plot labels
            #         ax.set_xlabel(labels[0])
            #         ax.set_ylabel(labels[1])
                    
            #         ## Set plot limits
            #         x_min = limits[0]; x_max = limits[1]; y_min = limits[2]; y_max = limits[3]
                    
            #         ax.set_xlim(x_min,x_max)
            #         ax.set_ylim(y_min,y_max)

            #         ## Figure for model ensemble disagreement
            #         plt.plot(range(len(pred_traj[:,dim])), pred_traj[:,dim], 'k-')
                    
            #         ## Add the lines for each pred error
            #         for t in range(len(pred_traj)):
            #             x = [t, t]
            #             y = [pred_traj[t, dim], pred_traj[t, dim]+disagr[t]]
            #             plt.plot(x, y, 'g')
                        
            #         ## Add the pred error for each step
            #         ## Set plot title
            #         plt.title(f"{n} step model ensemble disagreement along example trajectories on dimension {dim}\n{init_method} on {init_episode} episodes\n{args.environment}")
            #         ## Save fig
            #         fig_name = f"{i}_{n}_step_trajectories_{args.environment}_disagr_example_dim_{dim}"
            #         plt.savefig(f"{fig_path_disagr}/{fig_name}",
            #                     bbox_inches='tight')
                    
            #         plt.close()

            ## n = 20

            # mean_pred_errors[i, j, 3] = np.nanmean(example_20_step_pred_errors)
            # std_pred_errors[i, j, 3] = np.nanstd(example_20_step_pred_errors)

            abs_20_step_pred_errors = np.absolute(example_20_step_pred_errors)
            # abs_20_step_pred_errors = abs_20_step_pred_errors[abs_20_step_pred_errors < 20]
            # abs_20_step_pred_errors[abs_20_step_pred_errors > cap_val] = cap_val

            # mean_pred_errors[i, j, 1] = np.nanmean(np.absolute(example_20_step_pred_errors))
            # std_pred_errors[i, j, 1] = np.nanstd(np.absolute(example_20_step_pred_errors))

            mean_pred_errors[i, j, 1] = np.nanmean(abs_20_step_pred_errors[:stop_hor])
            std_pred_errors[i, j, 1] = np.nanstd(abs_20_step_pred_errors[:stop_hor])
            cell_text_20_step[j][i] = f"{round(mean_pred_errors[i,j,1],3)} \u00B1 {round(std_pred_errors[i,j,1],3)}"
                

            # for r in range(trajs_per_rep):
            #     pred_trajs = example_20_step_trajs[r::2]
            #     pred_errors = example_20_step_pred_errors[r::2]
            #     disagrs = example_20_step_disagrs[r::2]

            #     pred_traj = np.nanmean(pred_trajs, axis=0)
            #     pred_error = np.nanmean(pred_errors, axis=0)
            #     disagr = np.nanmean(disagrs, axis=0)

            #     n = 20
            #     for dim in range(pred_traj.shape[1]):
            #         ### Model prediction error ###
                    
            #         ## Create fig and ax
            #         fig = plt.figure()
            #         ax = fig.add_subplot(111)
            #         ## Prepare plot
            #         labels = ['Number of steps on environment',
            #                   f'Trajectory on dimension {dim}']
            #         limits = [0, len(pred_traj[:,dim]),
            #                   min(min(pred_traj[:, dim]),
            #                       min(pred_traj[:, dim]+pred_error[:, dim])),
            #                   max(max(pred_traj[:, dim]),
            #                       max(pred_traj[:, dim]+pred_error[:, dim]))]

            #         ## Set plot labels
            #         ax.set_xlabel(labels[0])
            #         ax.set_ylabel(labels[1])
                    
            #         ## Set plot limits
            #         x_min = limits[0]; x_max = limits[1]; y_min = limits[2]; y_max = limits[3]
                    
            #         ax.set_xlim(x_min,x_max)
            #         ax.set_ylim(y_min,y_max)

            #         ## Figure for model ensemble disagreement
            #         plt.plot(range(len(pred_traj[:,dim])), pred_traj[:,dim], 'k-')
                    
            #         ## Add the lines for each pred error
            #         for t in range(len(pred_traj)):
            #             x = [t, t]
            #             y = [pred_traj[t, dim], pred_traj[t, dim] + pred_error[t, dim]]
            #             plt.plot(x, y, 'g')
                
            #         ## Add the pred error for each step
            #         ## Set plot title
            #         plt.title(f"{n} step model ensemble prediction error along example trajectories on dimension {dim}\n{init_method} on {init_episode} episodes\n{args.environment}")
            #         ## Save fig
            #         fig_name = f"{i}_{n}_step_trajectories_{args.environment}_pred_error_example_dim_{dim}"
            #         plt.savefig(f"{fig_path_pred_error}/{fig_name}",
            #                     bbox_inches='tight')
                    
            #         plt.close()
            #         ### Model Ensemble Disagreement ###
                    
            #         ## Create fig and ax
            #         fig = plt.figure()
            #         ax = fig.add_subplot(111)
            #         ## Prepare plot
            #         labels = ['Number of steps on environment',
            #                   f'Trajectory on dimension {dim}']
            #         limits = [0, len(pred_traj[:,dim]),
            #                   min(min(pred_traj[:, dim]), min(pred_traj[:, dim]+disagr[:])),
            #                   max(max(pred_traj[:, dim]), max(pred_traj[:, dim]+disagr[:]))]

            #         ## Set plot labels
            #         ax.set_xlabel(labels[0])
            #         ax.set_ylabel(labels[1])
                    
            #         ## Set plot limits
            #         x_min = limits[0]; x_max = limits[1]; y_min = limits[2]; y_max = limits[3]
                    
            #         ax.set_xlim(x_min,x_max)
            #         ax.set_ylim(y_min,y_max)

            #         ## Figure for model ensemble disagreement
            #         plt.plot(range(len(pred_traj[:,dim])), pred_traj[:,dim], 'k-')
                    
            #         ## Add the lines for each pred error
            #         for t in range(len(pred_traj)):
            #             x = [t, t]
            #             y = [pred_traj[t, dim], pred_traj[t, dim]+disagr[t]]
            #             plt.plot(x, y, 'g')
                        
            #         ## Add the pred error for each step
            #         ## Set plot title
            #         plt.title(f"{n} step model ensemble disagreement along example trajectories on dimension {dim}\n{init_method} on {init_episode} episodes\n{args.environment}")
            #         ## Save fig
            #         fig_name = f"{i}_{n}_step_trajectories_{args.environment}_disagr_example_dim_{dim}"
            #         plt.savefig(f"{fig_path_disagr}/{fig_name}",
            #                     bbox_inches='tight')
                    
            #         plt.close()
        
            #### Mean Disagreement, Mean Prediction Error, Prediction Error vs Disagreement ####
            # Add to plot

            # ## On test trajs
            # ## Compute mean and stddev of trajs disagreement
            # test_mean_disagr = np.nanmean(test_disagrs[i,j], axis=0)
            # test_std_disagr = np.nanstd(test_disagrs[i,j], axis=0)
            # ## Compute mean and stddev of trajs prediction error
            # test_mean_pred_error = np.nanmean(test_pred_errors[i,j], axis=0)
            # test_std_pred_error = np.nanstd(test_pred_errors[i,j], axis=0)

            ## Do small trick for label because data has a wrong name lol
            label_method = f'{init_method}'
            if label_method == 'brownian-motion':
                label_method = 'uniform-random-walk'
            if label_method == 'colored-noise-beta-0':
                label_method = 'brownian-motion'
            label = f'{label_method}_{init_episode}'

            # ## Update plot params
            # if min(test_mean_disagr) < test_limits_disagr[2]:
            #     test_limits_disagr[2] = min(test_mean_disagr)
            # if max(test_mean_disagr) > test_limits_disagr[3]:
            #     test_limits_disagr[3] = max(test_mean_disagr)
            # ## Figure for model ensemble disagreement
            # test_ax_disagr.plot(range(max_step), test_mean_disagr,
            #                     # color=colors.to_rgba(i*n_init_episodes + j),
            #                     # linestyle=linestyles[(i*n_init_episodes + j)%len(linestyles)],
            #                     color=colors.to_rgba(i),
            #                     linestyle=linestyles[i],
            #                     # label=f'{init_method}_{init_episode}')
            #                     label=label)
            # ## Update plot params
            # if min(test_mean_pred_error) < test_limits_pred_error[2]:
            #     test_limits_pred_error[2] = min(test_mean_pred_error)
            # if max(test_mean_pred_error) > test_limits_pred_error[3]:
            #     test_limits_pred_error[3] = max(test_mean_pred_error)
            # ## Figure for pred_error
            # test_ax_pred_error.plot(range(max_step), test_mean_pred_error,
            #                         # color=colors.to_rgba(i*n_init_episodes + j),
            #                         # linestyle=linestyles[(i*n_init_episodes + j)%len(linestyles)],
            #                         color=colors.to_rgba(i),
            #                         linestyle=linestyles[i],
            #                         # label=f'{init_method}_{init_episode}')
            #                         label=label)

            # sorted_idxs = test_mean_disagr.argsort()
            # test_ax_pred_disagr.plot(test_mean_disagr[sorted_idxs],
            #                          test_mean_pred_error[sorted_idxs],
            #                          # color=colors.to_rgba(i*n_init_episodes + j),
            #                          # linestyle=linestyles[(i*n_init_episodes + j)%len(linestyles)],
            #                          color=colors.to_rgba(i),
            #                          linestyle=linestyles[i],
            #                          # label=f'{init_method}_{init_episode}')
            #                          label=label)
            ## On example trajs
            ## Compute mean and stddev of trajs disagreement
            example_mean_disagr = np.nanmean(example_disagrs[i,j], axis=0)
            example_std_disagr = np.nanstd(example_disagrs[i,j], axis=0)
            ## Compute mean and stddev of trajs prediction error
            example_mean_pred_error = np.nanmean(example_pred_errors[i,j], axis=0)
            example_std_pred_error = np.nanstd(example_pred_errors[i,j], axis=0)

            ## Update plot params
            if min(example_mean_disagr) < example_limits_disagr[2]:
                example_limits_disagr[2] = min(example_mean_disagr)
            if max(example_mean_disagr) < example_limits_disagr[3]:
                example_limits_disagr[3] = max(example_mean_disagr)
            ## Figure for model ensemble disagreement
            example_ax_disagr.plot(range(max_step), example_mean_disagr,
                                   # color=colors.to_rgba(i*n_init_episodes + j),
                                   # linestyle=linestyles[(i*n_init_episodes + j)%len(linestyles)],
                                   color=colors.to_rgba(i),
                                   linestyle=linestyles[i],
                                   # label=f'{init_method}_{init_episode}')
                                   label=label)
            ## Update plot params
            if min(example_mean_pred_error) < example_limits_pred_error[2]:
                example_limits_pred_error[2] = min(example_mean_pred_error)
            if max(example_mean_pred_error) < example_limits_pred_error[3]:
                example_limits_pred_error[3] = max(example_mean_pred_error)
            ## Figure for pred_error
            example_ax_pred_error.plot(range(max_step), example_mean_pred_error,
                                       # color=colors.to_rgba(i*n_init_episodes + j),
                                       # linestyle=linestyles[(i*n_init_episodes + j)%len(linestyles)],
                                       color=colors.to_rgba(i),
                                       linestyle=linestyles[i],
                                       # label=f'{init_method}_{init_episode}')
                                       label=label)

            sorted_idxs = example_mean_disagr.argsort()
            example_ax_pred_disagr.plot(example_mean_disagr[sorted_idxs],
                                        example_mean_pred_error[sorted_idxs],
                                        # color=colors.to_rgba(i*n_init_episodes + j),
                                        # linestyle=linestyles[(i*n_init_episodes + j)%len(linestyles)],
                                        color=colors.to_rgba(i),
                                        linestyle=linestyles[i],
                                        # label=f'{init_method}_{init_episode}')
                                        label=label)

            print(f"\nPlotted for init_method {init_method} and init_episode {init_episode}\n")
            # if plot_stddev:
                # ax_disagr.fill_between(range(len(mean_disagr)),
                                       # mean_disagr-std_disagr,
                                       # mean_disagr+std_disagr,
                                       # facecolor='green', alpha=0.5)
            ## init method = i; init episode = j

        #########################################################################
        ############################  WIP WIP WIP WIP ###########################
        #########################################################################
        ## All examples shape:
        # (n_init_method, n_init_episodes, n_total_trajs, task_h, obs_dim)
        mins = np.nanmin(all_example_1_step_pred_errors, axis=(0,1,2))
        maxs = np.nanmax(all_example_1_step_pred_errors, axis=(0,1,2))
        norm_all_example_1_step_pred_errors = \
            (all_example_1_step_pred_errors - mins)/(maxs - mins)

        # for i in range(n_init_method):
            # init_method =  init_methods[i]

        norm_example_1_step_pred_errors = np.nanmean(norm_all_example_1_step_pred_errors,
                                                     axis=(1,2))
        norm_example_1_step_pred_errors = np.nanmean(norm_all_example_1_step_pred_errors,
                                                     axis=(1,2))
        for i in range(n_init_method):
            norm_mean_pred_errors[i, j, 0] = \
                np.nanmean(norm_all_example_1_step_pred_errors[i])
            norm_std_pred_errors[i, j, 0] = \
                np.nanstd(norm_all_example_1_step_pred_errors[i])
        ## NB: je retiens les mins/maxs, et ensuite je classe les erreurs de modele
        
        # test_limits_disagr = [0, max_step,
        #                       0, args.disagr_plot_upper_lim]
        # test_limits_pred_error = [0, max_step,
        #                           0, args.pred_err_plot_upper_lim]
        example_limits_disagr = [0, max_step,
                                 0, args.disagr_plot_upper_lim]
        example_limits_pred_error = [0, max_step,
                                     0, args.pred_err_plot_upper_lim]
        
        ## Plot params
        # Set plot labels
        # test_ax_disagr.set_xlabel(test_labels_disagr[0])
        # test_ax_disagr.set_ylabel(test_labels_disagr[1])
        # test_ax_pred_error.set_xlabel(test_labels_pred_error[0])
        # test_ax_pred_error.set_ylabel(test_labels_pred_error[1])
        # test_ax_pred_disagr.set_xlabel(test_limits_pred_disagr[0])
        # test_ax_pred_disagr.set_xlabel(test_limits_pred_disagr[1])
        
        example_ax_disagr.set_xlabel(example_labels_disagr[0])
        example_ax_disagr.set_ylabel(example_labels_disagr[1])
        example_ax_pred_error.set_xlabel(example_labels_pred_error[0])
        example_ax_pred_error.set_ylabel(example_labels_pred_error[1])
        example_ax_pred_disagr.set_xlabel(example_limits_pred_disagr[0])
        example_ax_pred_disagr.set_xlabel(example_limits_pred_disagr[1])
        
        ## Set log scale if fastsim
        if args.environment == 'fastsim_maze' or args.environment == 'fastsim_maze_traps':
            # test_ax_pred_error.set_yscale('log')

            # test_ax_pred_disagr.set_yscale('log')

            example_ax_pred_error.set_yscale('log')

            example_ax_pred_disagr.set_yscale('log')
        
        ## Set plot limits
        # x_min = test_limits_disagr[0]; x_max = test_limits_disagr[1];
        # y_min = test_limits_disagr[2]; y_max = test_limits_disagr[3]
        
        # test_ax_disagr.set_xlim(x_min,x_max)
        # test_ax_disagr.set_ylim(y_min,y_max)
        
        # x_min = test_limits_pred_error[0]; x_max = test_limits_pred_error[1];
        # y_min = test_limits_pred_error[2]; y_max = test_limits_pred_error[3]
        
        # test_ax_pred_error.set_xlim(x_min,x_max)
        # test_ax_pred_error.set_ylim(y_min,y_max)
        
        # x_min = test_limits_disagr[2]; x_max = test_limits_disagr[3];
        # y_min = test_limits_pred_error[2]; y_max = test_limits_pred_error[3]
        
        # test_ax_pred_disagr.set_xlim(x_min,x_max)
        # test_ax_pred_disagr.set_ylim(y_min,y_max)
        
        x_min = example_limits_disagr[0]; x_max = example_limits_disagr[1];
        y_min = example_limits_disagr[2]; y_max = example_limits_disagr[3]
        
        example_ax_disagr.set_xlim(x_min,x_max)
        example_ax_disagr.set_ylim(y_min,y_max)

        x_min = example_limits_pred_error[0]; x_max = example_limits_pred_error[1];
        y_min = example_limits_pred_error[2]; y_max = example_limits_pred_error[3]
        
        example_ax_pred_error.set_xlim(x_min,x_max)
        example_ax_pred_error.set_ylim(y_min,y_max)
        
        x_min = example_limits_disagr[2]; x_max = example_limits_disagr[3];
        y_min = example_limits_pred_error[2]; y_max = example_limits_pred_error[3]
        
        example_ax_pred_disagr.set_xlim(x_min,x_max)
        example_ax_pred_disagr.set_ylim(y_min,y_max)
        
        ## Set legend
        
        # test_ax_disagr.legend()
        # test_ax_pred_error.legend()
        # test_ax_pred_disagr.legend()
        
        example_ax_disagr.legend()
        example_ax_pred_error.legend()
        example_ax_pred_disagr.legend()
        
        ## Set plot title
        # test_ax_disagr.set_title(f"Mean model ensemble disagreeement along test trajectories with {init_episode} budget")
        # test_ax_pred_error.set_title(f"Mean prediction error along test trajectories with {init_episode} budget")
        # test_ax_pred_disagr.set_title(f"Mean prediction error vs model ensemble disagreement on test trajectories with {init_episode} budget")
        
        example_ax_disagr.set_title(f"Mean model ensemble disagreeement along example trajectories with {init_episode} budget")
        example_ax_pred_error.set_title(f"Mean prediction error along example trajectories with {init_episode} budget")
        example_ax_pred_disagr.set_title(f"Mean prediction error vs model ensemble disagreement on example trajectories with {init_episode} budget")
        ## Save fig
        # test_fig_disagr.savefig(f"{args.dump_path}/{args.environment}_test_trajectories_disagr_{init_episode}",
        #                         bbox_inches='tight')
        # test_fig_pred_error.savefig(f"{args.dump_path}/{args.environment}_test_trajectories_pred_error_{init_episode}",
        #                             bbox_inches='tight')
        # test_fig_pred_disagr.savefig(f"{args.dump_path}/{args.environment}_test_trajectories_pred_disagr_{init_episode}",
        #                              bbox_inches='tight')
        
        example_fig_disagr.savefig(f"{args.dump_path}/{args.environment}_example_trajectories_disagr_{init_episode}",
                                   bbox_inches='tight')
        example_fig_pred_error.savefig(f"{args.dump_path}/{args.environment}_example_trajectories_pred_error_{init_episode}",
                                       bbox_inches='tight')
        example_fig_pred_disagr.savefig(f"{args.dump_path}/{args.environment}_example_trajectories_pred_disagr_{init_episode}",
                                        bbox_inches='tight')
        
        ## Clear figs
        
        # test_ax_disagr.cla()
        # test_ax_pred_error.cla()
        # test_ax_pred_disagr.cla()
        
        example_ax_disagr.cla()
        example_ax_pred_error.cla()
        example_ax_pred_disagr.cla()
            

    if env_name == 'cartpole':
        ## change order on 25 step because of dump order error in
        ## MIS PETS TS1 
        tmp_0 = mean_pred_errors[1][0][1]
        tmp_1 = mean_pred_errors[2][0][1]
        tmp_2 = mean_pred_errors[3][0][1]
        mean_pred_errors[1][0][1] = tmp_1
        mean_pred_errors[2][0][1] = tmp_2
        mean_pred_errors[3][0][1] = tmp_0
        ## Same for cell text
        tmp_0 = cell_text_20_step[0][1]
        tmp_1 = cell_text_20_step[0][2]
        tmp_2 = cell_text_20_step[0][3]
        cell_text_20_step[0][1] = tmp_1
        cell_text_20_step[0][2] = tmp_2
        cell_text_20_step[0][3] = tmp_0
        
    ## Save aggregated data
    np.savez(f"{env_name}_pred_error_data.npz",
             mean_pred_errors=mean_pred_errors,
             std_pred_errors=std_pred_errors)

    print(f'Saved file {env_name}_pred_error_data.npz')

    ###############################################################
    ###############################################################
    ##################### Pred error tables #######################
    ########################### below #############################
    ###############################################################

    ## Dump the error table as a latex table

    from table_latex_dump import dump_pred_error_latex_table

    dump_pred_error_latex_table(mean_pred_errors, std_pred_errors,
                                args.dump_path, env_name,
                                init_methods, init_episodes, pred_steps)

    dump_pred_error_latex_table(norm_mean_pred_errors, norm_std_pred_errors,
                                args.dump_path, env_name, init_methods,
                                init_episodes, pred_steps, norm=True)
    
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    the_table = plt.table(cellText=cell_text_1_step,
                          rowLabels=row_headers,
                          rowColours=rcolors,
                          rowLoc='right',
                          colColours=ccolors,
                          colLabels=column_headers,
                          loc='center')
    fig.tight_layout()
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(6)
    plt.title(f'Mean prediction error and standard deviation for 1 step predictions on {args.environment} environment', y=.7)
    
    plt.savefig(f"{args.environment}_quant_pred_error_1_step", dpi=300, bbox_inches='tight')
    
    print(f'Saved figure {args.environment}_quant_pred_error_1_step.png')
    
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    the_table = plt.table(cellText=cell_text_20_step,
                          rowLabels=row_headers,
                          rowColours=rcolors,
                          rowLoc='right',
                          colColours=ccolors,
                          colLabels=column_headers,
                          loc='center')
    fig.tight_layout()
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(6)
    plt.title(f'Mean prediction error and standard deviation for 20 step predictions on {args.environment} environment', y=.7)
    
    plt.savefig(f"{args.environment}_quant_pred_error_20_step", dpi=300, bbox_inches='tight')

    print(f'Saved figure {args.environment}_quant_pred_error_20_step.png')

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    the_table = plt.table(cellText=cell_text_full,
                          rowLabels=row_headers,
                          rowColours=rcolors,
                          rowLoc='right',
                          colColours=ccolors,
                          colLabels=column_headers,
                          loc='center')
    fig.tight_layout()
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(6)
    plt.title(f'Mean prediction error and standard deviation for complete horizon predictions on {args.environment} environment', y=.7)
    
    plt.savefig(f"{args.environment}_quant_pred_error_full", dpi=300, bbox_inches='tight')

    print(f'Saved figure {args.environment}_quant_pred_error_full.png')

    ###############################################################
    ###############################################################
    #################### Plot all on same fig #####################
    ########################### below #############################
    ###############################################################
            
    # test_limits_disagr = [0, max_step,
    #                       0, args.disagr_plot_upper_lim]
    # test_limits_pred_error = [0, max_step,
    #                           0, args.pred_err_plot_upper_lim]
    example_limits_disagr = [0, max_step,
                             0, args.disagr_plot_upper_lim]
    example_limits_pred_error = [0, max_step,
                                 0, args.pred_err_plot_upper_lim]

    ## Plot params
    # Set plot labels
    # test_ax_disagr.set_xlabel(test_labels_disagr[0])
    # test_ax_disagr.set_ylabel(test_labels_disagr[1])
    # test_ax_pred_error.set_xlabel(test_labels_pred_error[0])
    # test_ax_pred_error.set_ylabel(test_labels_pred_error[1])
    # test_ax_pred_disagr.set_xlabel(test_limits_pred_disagr[0])
    # test_ax_pred_disagr.set_xlabel(test_limits_pred_disagr[1])

    example_ax_disagr.set_xlabel(example_labels_disagr[0])
    example_ax_disagr.set_ylabel(example_labels_disagr[1])
    example_ax_pred_error.set_xlabel(example_labels_pred_error[0])
    example_ax_pred_error.set_ylabel(example_labels_pred_error[1])
    example_ax_pred_disagr.set_xlabel(example_limits_pred_disagr[0])
    example_ax_pred_disagr.set_xlabel(example_limits_pred_disagr[1])


    ## Set plot limits
    # x_min = test_limits_disagr[0]; x_max = test_limits_disagr[1];
    # y_min = test_limits_disagr[2]; y_max = test_limits_disagr[3]
    
    # test_ax_disagr.set_xlim(x_min,x_max)
    # test_ax_disagr.set_ylim(y_min,y_max)

    # x_min = test_limits_pred_error[0]; x_max = test_limits_pred_error[1];
    # y_min = test_limits_pred_error[2]; y_max = test_limits_pred_error[3]
    
    # test_ax_pred_error.set_xlim(x_min,x_max)
    # test_ax_pred_error.set_ylim(y_min,y_max)

    # x_min = test_limits_disagr[2]; x_max = test_limits_disagr[3];
    # y_min = test_limits_pred_error[2]; y_max = test_limits_pred_error[3]
    
    # test_ax_pred_disagr.set_xlim(x_min,x_max)
    # test_ax_pred_disagr.set_ylim(y_min,y_max)

    x_min = example_limits_disagr[0]; x_max = example_limits_disagr[1];
    y_min = example_limits_disagr[2]; y_max = example_limits_disagr[3]
    
    example_ax_disagr.set_xlim(x_min,x_max)
    example_ax_disagr.set_ylim(y_min,y_max)

    x_min = example_limits_pred_error[0]; x_max = example_limits_pred_error[1];
    y_min = example_limits_pred_error[2]; y_max = example_limits_pred_error[3]
    
    example_ax_pred_error.set_xlim(x_min,x_max)
    example_ax_pred_error.set_ylim(y_min,y_max)

    x_min = example_limits_disagr[2]; x_max = example_limits_disagr[3];
    y_min = example_limits_pred_error[2]; y_max = example_limits_pred_error[3]
    
    example_ax_pred_disagr.set_xlim(x_min,x_max)
    example_ax_pred_disagr.set_ylim(y_min,y_max)

    ## Set legend

    # test_ax_disagr.legend(prop={'size': 1})
    # test_ax_pred_error.legend(prop={'size': 1})
    # test_ax_pred_disagr.legend(prop={'size': 1})

    example_ax_disagr.legend(prop={'size': 1})
    example_ax_pred_error.legend(prop={'size': 1})
    example_ax_pred_disagr.legend(prop={'size': 1})

    ## Set plot title
    # test_ax_disagr.set_title(f"Mean model ensemble disagreeement along test trajectories")
    # test_ax_pred_error.set_title(f"Mean prediction error along test trajectories")
    # test_ax_pred_disagr.set_title(f"Mean prediction error vs model ensemble disagreement on test trajectories")

    example_ax_disagr.set_title(f"Mean model ensemble disagreeement along example trajectories")
    example_ax_pred_error.set_title(f"Mean prediction error along example trajectories")
    example_ax_pred_disagr.set_title(f"Mean prediction error vs model ensemble disagreement on example trajectories")
    ## Save fig
    # test_fig_disagr.savefig(f"{args.dump_path}/{args.environment}_test_trajectories_disagr",
    #                         bbox_inches='tight')
    # test_fig_pred_error.savefig(f"{args.dump_path}/{args.environment}_test_trajectories_pred_error",
    #                         bbox_inches='tight')
    # test_fig_pred_disagr.savefig(f"{args.dump_path}/{args.environment}_test_trajectories_pred_disagr",
    #                         bbox_inches='tight')

    example_fig_disagr.savefig(f"{args.dump_path}/{args.environment}_example_trajectories_disagr",
                               bbox_inches='tight')
    example_fig_pred_error.savefig(f"{args.dump_path}/{args.environment}_example_trajectories_pred_error",
                               bbox_inches='tight')
    example_fig_pred_disagr.savefig(f"{args.dump_path}/{args.environment}_example_trajectories_pred_disagr",
                               bbox_inches='tight')

    print()
    
