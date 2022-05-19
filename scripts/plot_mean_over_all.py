## Call this piece of code from the top folder containing all env data ##
if __name__ == '__main__':
    # Local imports
    from model_init_study.visualization.test_trajectories_visualization \
        import TestTrajectoriesVisualization

    from model_init_study.models.dynamics_model \
        import DynamicsModel

    from model_init_study.controller.nn_controller \
        import NeuralNetworkController

    from model_init_study.visualization.fetch_pick_and_place_separator \
        import FetchPickAndPlaceSeparator
    from model_init_study.visualization.ant_separator \
        import AntSeparator
    from model_init_study.visualization.ball_in_cup_separator \
        import BallInCupSeparator
    from model_init_study.visualization.redundant_arm_separator \
        import RedundantArmSeparator
    
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
    
    module_path = os.path.dirname(model_init_study.__file__)

    parser = argparse.ArgumentParser(description='Process run parameters.')

    parser.add_argument('--init-methods', nargs="*",
                        type=str, default=['random-policies', 'random-actions'])

    parser.add_argument('--init-episodes', nargs="*",
                        type=int, default=[5, 10, 15, 20])

    parser.add_argument('--disagr-plot-upper-lim', type=float, default=5.)
    parser.add_argument('--pred-err-plot-upper-lim', type=float, default=5.)
    
    parser.add_argument('--dump-path', type=str, default='default_dump/')

    parser.add_argument('--environment', '-e', type=str, default='ball_in_cup')

    args = parser.parse_args()

    dynamics_model = DynamicsModel
    
    ## Framework methods
    env_register_id = 'BallInCup3d-v0'
    if args.environment == 'ball_in_cup':
        env_register_id = 'BallInCup3d-v0'
        separator = BallInCupSeparator
        ss_min = -0.4
        ss_max = 0.4
    if args.environment == 'redundant_arm':
        env_register_id = 'RedundantArm-v0'
        separator = RedundantArmSeparator
        ss_min = -1
        ss_max = 1
    if args.environment == 'fetch_pick_and_place':
        env_register_id = 'FetchPickAndPlaceDeterministic-v1'
        separator = FetchPickAndPlaceSeparator
        ss_min = -1
        ss_max = 1
    if args.environment == 'ant':
        env_register_id = 'AntBulletEnvDeterministic-v0'
        separator = AntSeparator
        ss_min = -10
        ss_max = 10
        
    env = gym.make(env_register_id)

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
    trajs_per_rep = len(tmp_data)
    n_total_trajs = trajs_per_rep*len(rep_folders)
    task_h = len(tmp_data[0])

    n_init_method = 2
    init_methods = args.init_methods #['random-policies', 'random-actions']
    n_init_episodes = 4
    init_episodes = args.init_episodes #[5, 10, 15, 20]
    
    test_pred_trajs = np.empty((n_init_method, n_init_episodes, n_total_trajs, task_h, obs_dim))
    test_disagrs = np.empty((n_init_method, n_init_episodes, n_total_trajs, task_h))
    test_pred_errors = np.empty((n_init_method, n_init_episodes, n_total_trajs, task_h))

    example_pred_trajs = np.empty((n_init_method, n_init_episodes, n_total_trajs, task_h, obs_dim))
    example_disagrs = np.empty((n_init_method, n_init_episodes, n_total_trajs, task_h))
    example_pred_errors = np.empty((n_init_method, n_init_episodes, n_total_trajs, task_h))
    

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
    test_limits_disagr = [0, env._max_episode_steps,
                          0, 0]
    test_limits_pred_error = [0, env._max_episode_steps,
                              0, 0]
    test_limits_pred_disagr = [0, env._max_episode_steps,
                               0, 0]
    example_limits_disagr = [0, env._max_episode_steps,
                             0, 0]
    example_limits_pred_error = [0, env._max_episode_steps,
                                 0, 0]
    example_limits_pred_disagr = [0, env._max_episode_steps,
                                  0, 0]

    # Init labels for each fig
    test_labels_disagr = ['Number of steps on environment', 'Mean ensemble disagreement']
    test_labels_pred_error = ['Number of steps on environment', 'Mean prediction error']
    test_labels_pred_disagr = ['Mean ensemble disagreement', 'Mean prediction error']
    example_labels_disagr = ['Number of steps on environment', 'Mean ensemble disagreement']
    example_labels_pred_error = ['Number of steps on environment', 'Mean prediction error']
    example_labels_pred_disagr = ['Mean ensemble disagreement', 'Mean prediction error']
    
    evenly_spaced_interval = np.linspace(0, 1, n_init_method*n_init_episodes)
    colors = [cm.rainbow(x) for x in evenly_spaced_interval]

    colors = plt.cm.get_cmap('hsv', n_init_method*n_init_episodes+1)
    
    for i in range(n_init_method):
        init_method =  init_methods[i]
        for j in range(n_init_episodes):
            init_episode = init_episodes[j]
            rep_cpt = 0
            for rep_path in rep_folders:
                rep_data = np.load(f'{rep_path}/{args.environment}_{init_method}_{init_episode}_data.npz')

                test_pred_trajs[i,j,rep_cpt*trajs_per_rep:
                                rep_cpt*trajs_per_rep
                                + trajs_per_rep] =rep_data['test_pred_trajs']
                test_disagrs[i,j,rep_cpt*trajs_per_rep:
                             rep_cpt*trajs_per_rep
                             + trajs_per_rep] = rep_data['test_disagrs']
                test_pred_errors[i,j,rep_cpt*trajs_per_rep:
                                 rep_cpt*trajs_per_rep
                                 + trajs_per_rep] = rep_data['test_pred_errors']
                
                example_pred_trajs[i,j,rep_cpt*trajs_per_rep:
                                   rep_cpt*trajs_per_rep
                                   + trajs_per_rep] = rep_data['examples_pred_trajs']
                example_disagrs[i,j,rep_cpt*trajs_per_rep:
                                rep_cpt*trajs_per_rep
                                + trajs_per_rep] = rep_data['examples_disagrs']
                example_pred_errors[i,j,rep_cpt*trajs_per_rep:
                                    rep_cpt*trajs_per_rep
                                    + trajs_per_rep] = rep_data['examples_pred_errors']

                rep_cpt += 1

            # Add to plot

            ## On test trajs
            ## Compute mean and stddev of trajs disagreement
            test_mean_disagr = np.nanmean(test_disagrs[i,j], axis=0)
            test_std_disagr = np.nanstd(test_disagrs[i,j], axis=0)
            ## Compute mean and stddev of trajs prediction error
            test_mean_pred_error = np.nanmean(test_pred_errors[i,j], axis=0)
            test_std_pred_error = np.nanstd(test_pred_errors[i,j], axis=0)

            ## Update plot params
            if min(test_mean_disagr) < test_limits_disagr[2]:
                test_limits_disagr[2] = min(test_mean_disagr)
            if max(test_mean_disagr) > test_limits_disagr[3]:
                test_limits_disagr[3] = max(test_mean_disagr)
            ## Figure for model ensemble disagreement
            test_ax_disagr.plot(range(env._max_episode_steps), test_mean_disagr, '-',
                                color=colors(i*n_init_episodes + j),
                                label=f'{init_method}_{init_episode}')
            ## Update plot params
            if min(test_mean_pred_error) < test_limits_pred_error[2]:
                test_limits_pred_error[2] = min(test_mean_pred_error)
            if max(test_mean_pred_error) > test_limits_pred_error[3]:
                test_limits_pred_error[3] = max(test_mean_pred_error)
            ## Figure for pred_error
            test_ax_pred_error.plot(range(env._max_episode_steps), test_mean_pred_error, '-',
                                color=colors(i*n_init_episodes + j),
                                    label=f'{init_method}_{init_episode}')

            sorted_idxs = test_mean_disagr.argsort()
            test_ax_pred_disagr.plot(test_mean_disagr[sorted_idxs],
                                     test_mean_pred_error[sorted_idxs], '-',
                                     color=colors(i*n_init_episodes + j),
                                     label=f'{init_method}_{init_episode}')
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
            example_ax_disagr.plot(range(env._max_episode_steps), example_mean_disagr, '-',
                                color=colors(i*n_init_episodes + j),
                                   label=f'{init_method}_{init_episode}')
            ## Update plot params
            if min(example_mean_pred_error) < example_limits_pred_error[2]:
                example_limits_pred_error[2] = min(example_mean_pred_error)
            if max(example_mean_pred_error) < example_limits_pred_error[3]:
                example_limits_pred_error[3] = max(example_mean_pred_error)
            ## Figure for pred_error
            example_ax_pred_error.plot(range(env._max_episode_steps), example_mean_pred_error,'-',
                                color=colors(i*n_init_episodes + j),
                                       label=f'{init_method}_{init_episode}')

            sorted_idxs = example_mean_disagr.argsort()
            example_ax_pred_disagr.plot(example_mean_disagr[sorted_idxs],
                                        example_mean_pred_error[sorted_idxs], '-',
                                        color=colors(i*n_init_episodes + j),
                                        label=f'{init_method}_{init_episode}')
            # if plot_stddev:
                # ax_disagr.fill_between(range(len(mean_disagr)),
                                       # mean_disagr-std_disagr,
                                       # mean_disagr+std_disagr,
                                       # facecolor='green', alpha=0.5)
    test_limits_disagr = [0, env._max_episode_steps,
                          0, args.disagr_plot_upper_lim]
    test_limits_pred_error = [0, env._max_episode_steps,
                              0, args.pred_err_plot_upper_lim]
    example_limits_disagr = [0, env._max_episode_steps,
                             0, args.disagr_plot_upper_lim]
    example_limits_pred_error = [0, env._max_episode_steps,
                                 0, args.pred_err_plot_upper_lim]

    ## Plot params
    # Set plot labels
    test_ax_disagr.set_xlabel(test_labels_disagr[0])
    test_ax_disagr.set_ylabel(test_labels_disagr[1])
    test_ax_pred_error.set_xlabel(test_labels_pred_error[0])
    test_ax_pred_error.set_ylabel(test_labels_pred_error[1])
    test_ax_pred_disagr.set_xlabel(test_limits_pred_disagr[0])
    test_ax_pred_disagr.set_xlabel(test_limits_pred_disagr[1])

    example_ax_disagr.set_xlabel(example_labels_disagr[0])
    example_ax_disagr.set_ylabel(example_labels_disagr[1])
    example_ax_pred_error.set_xlabel(example_labels_pred_error[0])
    example_ax_pred_error.set_ylabel(example_labels_pred_error[1])
    example_ax_pred_disagr.set_xlabel(example_limits_pred_disagr[0])
    example_ax_pred_disagr.set_xlabel(example_limits_pred_disagr[1])


    ## Set plot limits
    x_min = test_limits_disagr[0]; x_max = test_limits_disagr[1];
    y_min = test_limits_disagr[2]; y_max = test_limits_disagr[3]
    
    test_ax_disagr.set_xlim(x_min,x_max)
    test_ax_disagr.set_ylim(y_min,y_max)

    x_min = test_limits_pred_error[0]; x_max = test_limits_pred_error[1];
    y_min = test_limits_pred_error[2]; y_max = test_limits_pred_error[3]
    
    test_ax_pred_error.set_xlim(x_min,x_max)
    test_ax_pred_error.set_ylim(y_min,y_max)

    x_min = test_limits_disagr[2]; x_max = test_limits_disagr[3];
    y_min = test_limits_pred_error[2]; y_max = test_limits_pred_error[3]
    
    test_ax_pred_disagr.set_xlim(x_min,x_max)
    test_ax_pred_disagr.set_ylim(y_min,y_max)

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

    test_ax_disagr.legend()
    test_ax_pred_error.legend()
    test_ax_pred_disagr.legend()

    example_ax_disagr.legend()
    example_ax_pred_error.legend()
    example_ax_pred_disagr.legend()

    ## Set plot title
    test_ax_disagr.set_title(f"Mean model ensemble disagreeement along test trajectories")
    test_ax_pred_error.set_title(f"Mean prediction error along test trajectories")
    test_ax_pred_disagr.set_title(f"Mean prediction error vs model ensemble disagreement on test trajectories")

    example_ax_disagr.set_title(f"Mean model ensemble disagreeement along example trajectories")
    example_ax_pred_error.set_title(f"Mean prediction error along example trajectories")
    example_ax_pred_disagr.set_title(f"Mean prediction error vs model ensemble disagreement on example trajectories")
    ## Save fig
    test_fig_disagr.savefig(f"{args.dump_path}/{args.environment}_test_trajectories_disagr",
                            bbox_inches='tight')
    test_fig_pred_error.savefig(f"{args.dump_path}/{args.environment}_test_trajectories_pred_error",
                            bbox_inches='tight')
    test_fig_pred_disagr.savefig(f"{args.dump_path}/{args.environment}_test_trajectories_pred_disagr",
                            bbox_inches='tight')

    example_fig_disagr.savefig(f"{args.dump_path}/{args.environment}_example_trajectories_disagr",
                               bbox_inches='tight')
    example_fig_pred_error.savefig(f"{args.dump_path}/{args.environment}_example_trajectories_pred_error",
                               bbox_inches='tight')
    example_fig_pred_disagr.savefig(f"{args.dump_path}/{args.environment}_example_trajectories_pred_disagr",
                               bbox_inches='tight')
    # plt.show()
