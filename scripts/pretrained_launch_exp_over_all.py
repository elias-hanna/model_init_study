## Call this piece of code from the top folder containing all env data ##
if __name__ == '__main__':
    # Local imports
    from model_init_study.visualization.test_trajectories_visualization \
        import TestTrajectoriesVisualization

    from model_init_study.models.dynamics_model \
        import DynamicsModel

    from model_init_study.controller.nn_controller \
        import NeuralNetworkController

    ## Separator classes
    from model_init_study.visualization.fetch_pick_and_place_separator \
        import FetchPickAndPlaceSeparator
    from model_init_study.visualization.ant_separator \
        import AntSeparator
    from model_init_study.visualization.ball_in_cup_separator \
        import BallInCupSeparator
    from model_init_study.visualization.redundant_arm_separator \
        import RedundantArmSeparator
    from model_init_study.visualization.fastsim_separator \
        import FastsimSeparator

    ## Visualization classes
    from model_init_study.visualization.state_space_repartition_visualization \
        import StateSpaceRepartitionVisualization
    from model_init_study.visualization.test_trajectories_visualization \
        import TestTrajectoriesVisualization
    from model_init_study.visualization.n_step_error_visualization \
        import NStepErrorVisualization
    
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
    
    module_path = os.path.dirname(model_init_study.__file__)

    parser = argparse.ArgumentParser(description='Process run parameters.')

    parser.add_argument('--init-methods', nargs="*",
                        type=str, default=['brownian-motion', 'levy-flight', 'colored-noise-beta-1', 'colored-noise-beta-2', 'random-actions'])

    parser.add_argument('--init-episodes', nargs="*",
                        type=int, default=[5, 10, 20])

    parser.add_argument('--pred-err-plot-upper-lim', type=float, default=5.)
    
    parser.add_argument('--dump-path', type=str, default='default_dump/')

    parser.add_argument('--environment', '-e', type=str, default='ball_in_cup')

    parser.add_argument('--pretrained-data-path', type=str)

    parser.add_argument('--no-training', action='store_true')

    args = parser.parse_args()

    sup_args = ""
    if args.no_training:
        sup_args = "--no-training"

    dynamics_model = DynamicsModel

    env_name = args.environment

    env_register_id = 'BallInCup3d-v0'
    gym_args = {}
    if args.environment == 'ball_in_cup':
        env_register_id = 'BallInCup3d-v0'
        separator = BallInCupSeparator
        ss_min = -0.4
        ss_max = 0.4
        x_idx = 0
        y_idx = 1
        z_idx = 2
    elif args.environment == 'redundant_arm':
        env_register_id = 'RedundantArmPos-v0'
        separator = RedundantArmSeparator
        ss_min = -1
        ss_max = 1
        x_idx = -2
        y_idx = -1
    elif args.environment == 'redundant_arm_no_walls':
        env_register_id = 'RedundantArmPosNoWalls-v0'
        separator = RedundantArmSeparator
        ss_min = -1
        ss_max = 1
        x_idx = -2
        y_idx = -1
    elif args.environment == 'redundant_arm_no_walls_no_collision':
        env_register_id = 'RedundantArmPosNoWallsNoCollision-v0'
        separator = RedundantArmSeparator
        ss_min = -1
        ss_max = 1
        x_idx = -2
        y_idx = -1
    elif args.environment == 'redundant_arm_no_walls_limited_angles':
        env_register_id = 'RedundantArmPosNoWallsLimitedAngles-v0'
        separator = RedundantArmSeparator
        ss_min = -1
        ss_max = 1
        gym_args['dof'] = 100
        x_idx = -2
        y_idx = -1
    elif args.environment == 'fastsim_maze':
        env_register_id = 'FastsimSimpleNavigationPos-v0'
        separator = FastsimSeparator
        ss_min = -10
        ss_max = 10
        x_idx = 0
        y_idx = 1
    elif args.environment == 'fastsim_maze_traps':
        env_register_id = 'FastsimSimpleNavigationPos-v0'
        separator = FastsimSeparator
        ss_min = -10
        ss_max = 10
        gym_args['physical_traps'] = True
        x_idx = 0
        y_idx = 1
    elif args.environment == 'fetch_pick_and_place':
        env_register_id = 'FetchPickAndPlaceDeterministic-v1'
        separator = FetchPickAndPlaceSeparator
        ss_min = -1
        ss_max = 1
    elif args.environment == 'ant':
        env_register_id = 'AntBulletEnvDeterministicPos-v0'
        separator = AntSeparator
        ss_min = -10
        ss_max = 10
    else:
        raise ValueError(f"{args.environment} is not a defined environment")
    
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

    controller_params = \
    {
        'controller_input_dim': obs_dim,
        'controller_output_dim': act_dim,
        'n_hidden_layers': 2,
        'n_neurons_per_hidden': 10
    }
    dynamics_model_params = \
    {
        'obs_dim': obs_dim,
        'action_dim': act_dim,
        'dynamics_model_type': 'prob', # possible values: prob, det
        'ensemble_size': 4, # only used if dynamics_model_type == prob
        'layer_size': 500,
        'batch_size': 512,
        'learning_rate': 1e-3,
        'train_unique_trans': False,
    }
    params = \
    {
        'obs_dim': obs_dim,
        'action_dim': act_dim,

        'separator': separator,
        
        'n_init_episodes': args.init_episodes,
        # 'n_test_episodes': int(.2*args.init_episodes), # 20% of n_init_episodes
        'n_test_episodes': 2,
        
        'controller_type': NeuralNetworkController,
        'controller_params': controller_params,

        'dynamics_model_params': dynamics_model_params,

        'action_min': -1,
        'action_max': 1,
        'action_init': 0,

        ## Random walks parameters
        'step_size': 0.1,

        'state_min': ss_min,
        'state_max': ss_max,
        
        'policy_param_init_min': -5,
        'policy_param_init_max': 5,
        
        'dump_path': args.dump_path,
        # 'path_to_test_trajectories': 'examples/'+args.environment+'_example_trajectories.npz',
        'path_to_test_trajectories': path_to_examples,

        'env': env,
        'env_max_h': max_step,
    }

    dynamics_model = DynamicsModel(params)

    rep_folders = next(os.walk(f'.'))[1]

    rep_folders = [x for x in rep_folders if (x.isdigit())]

    # rep_data = np.load(f'{rep_folders[0]}/{args.environment}_{args.init_methods[0]}_{args.init_episodes[0]}_data.npz')

    # tmp_data = rep_data['test_pred_trajs']

    ## Get parameters (should do a params.npz... cba)
    # trajs_per_rep = len(tmp_data)
    trajs_per_rep = 2 # 2 example trajs iirc
    n_total_trajs = trajs_per_rep*len(rep_folders)
    task_h = max_step

    n_init_method = len(args.init_methods) # 2
    init_methods = args.init_methods #['random-policies', 'random-actions']
    n_init_episodes = len(args.init_episodes) # 4
    init_episodes = args.init_episodes #[5, 10, 15, 20]
    
    example_pred_trajs = np.empty((n_init_method, n_init_episodes, n_total_trajs, task_h, obs_dim))
    example_pred_errors = np.empty((n_init_method, n_init_episodes, n_total_trajs, task_h))

    example_1_step_trajs = np.empty((n_total_trajs, task_h, obs_dim))
    example_1_step_pred_errors = np.empty((n_total_trajs, task_h, obs_dim))

    example_20_step_trajs = np.empty((n_total_trajs, task_h, obs_dim))
    example_20_step_pred_errors = np.empty((n_total_trajs, task_h, obs_dim))

    ## Do a new plot
    ## Will contain all means on a single plot
    # Create 4 figs
    example_fig_traj_plot = plt.figure()
    example_ax_traj_plot = fig_traj_plot.add_subplot(111)
    
    example_fig_pred_error = plt.figure()
    example_ax_pred_error = example_fig_pred_error.add_subplot(111)

    example_fig_cum_pred_error = plt.figure()
    example_ax_cum_pred_error = example_fig_cum_pred_error.add_subplot(111)
    if args.environment == 'ball_in_cup':
        example_fig_cum_pred_error = plt.figure()
        example_ax_cum_pred_error = example_fig_cum_pred_error.add_subplot(111, projection='3d')

    # Init limits for each fig
    example_limits_pred_error = [0, max_step,
                                 0, 0]

    example_limits_traj_plot = [ss_min, ss_max,
                                ss_min, ss_max]
    
    if args.environment == 'ball_in_cup':
        example_limits_traj_plot = [ss_min, ss_max,
                                    ss_min, ss_max,
                                    ss_min, ss_max]
        
    # Init labels for each fig
    example_labels_pred_error = ['Number of steps on environment', 'Mean prediction error']
    example_labels_cum_pred_error = ['Number of steps on environment', 'Cumulated mean prediction error (1-step predictions)']
    example_labels_traj_plot = ['X axis', 'Y axis']
    if args.environment == 'ball_in_cup':
        example_labels_traj_plot = ['X axis', 'Y axis', 'Z axis']

    ## Colors for plots
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
    cum_pred_errors = np.zeros((n_init_method, n_init_episodes, task_h))

    ## Plot table with mean prediction error for n step predictions    
    column_headers = [init_method for init_method in init_methods]
    row_headers = [init_episode for init_episode in init_episodes]
    cell_text_1_step = [["" for _ in range(len(column_headers))] for _ in range(len(row_headers))]
    cell_text_20_step = [["" for _ in range(len(column_headers))] for _ in range(len(row_headers))]
    cell_text_full = [["" for _ in range(len(column_headers))] for _ in range(len(row_headers))]
    rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))

    for j in range(n_init_episodes):
        init_episode = init_episodes[j]
        for i in range(n_init_method):
            init_method =  init_methods[i]
            rep_cpt = 0
            for rep_path in rep_folders:
                ## For each loaded model, relaunch the procedure to evaluate error on trajs
                ## TODO: Get the path of the file, should be able to get it
                ## from init method, init rep, and that's all -> need to refactor the model.pth obtained
                ## Surement un truc du genre je regarderais demain
                if args.pretrained_data_path is not None:
                    data_path = args.pretrained_data_path
                    path = os.path.join(data_path,
                                        f'{args.environment}{sup_args}_daqd_results/{rep_cpt+1}/{init_method}_{init_episode}_energy_minimization_random_10_results/trained_model.pth')
                    dynamics_model.load(path)
                else:
                    raise ValueError("No data path for pretrained data has been given")
                
                ## Execute each visualizer routines
                params['model'] = dynamics_model # to pass down to the visualizer routines
                test_traj_visualizer = TestTrajectoriesVisualization(params)
                n_step_visualizer = NStepErrorVisualization(params)

                ## Visualize n step error and disagreement ###

                n_step_visualizer.set_n(1)

                examples_1_step_trajs, examples_1_step_disagrs, examples_1_step_pred_errors = n_step_visualizer.dump_plots(
                    args.environment,
                    init_method,
                    init_episode,
                    'examples', dump_separate=True, no_sep=True)

                n_step_visualizer.set_n(20)

                examples_20_step_trajs, examples_20_step_disagrs, examples_20_step_pred_errors = n_step_visualizer.dump_plots(
                    args.environment,
                    init_method,
                    init_episode,
                    'examples', dump_separate=True, no_sep=True)

                ### Full recursive prediction visualizations ###
                examples_pred_trajs, examples_disagrs, examples_pred_errors = test_traj_visualizer.dump_plots(
                    args.environment,
                    init_method,
                    init_episode,
                    'examples', dump_separate=True, no_sep=True)

                ### Fill the rep data with collected results ###
                
                example_pred_trajs[i,j,rep_cpt*trajs_per_rep:
                                   rep_cpt*trajs_per_rep
                                   + trajs_per_rep] = examples_pred_trajs
                example_pred_errors[i,j,rep_cpt*trajs_per_rep:
                                    rep_cpt*trajs_per_rep
                                    + trajs_per_rep] = examples_pred_errors
                
                ### For N step vis ###
                example_1_step_trajs[rep_cpt*trajs_per_rep:
                                     rep_cpt*trajs_per_rep
                                     + trajs_per_rep] = examples_1_step_trajs
                example_1_step_pred_errors[rep_cpt*trajs_per_rep:
                                           rep_cpt*trajs_per_rep
                                           + trajs_per_rep] = examples_1_step_pred_errors

                example_20_step_trajs[rep_cpt*trajs_per_rep:
                                     rep_cpt*trajs_per_rep
                                     + trajs_per_rep] = examples_20_step_trajs
                example_20_step_pred_errors[rep_cpt*trajs_per_rep:
                                           rep_cpt*trajs_per_rep
                                           + trajs_per_rep] = examples_20_step_pred_errors

                rep_cpt += 1

                print(f"\n Finished processing {rep_cpt}th rep of {init_method}{sup_args} with {init_episode} init budget\n")


            ## Compute mean trajectory for each example traj
            # example_pred_trajs shape: (n_init_method, n_init_episodes, n_total_trajs, task_h, obs_dim)

            mean_example_traj_x = np.nanmean(example_pred_trajs[i,j,:2:,:,x_idx], axis=0)
            mean_example_traj_y = np.nanmean(example_pred_trajs[i,j,:2:,:,y_idx], axis=0)
            if args.environment == 'ball_in_cup':
                mean_example_traj_z = np.nanmean(example_pred_trajs[i,j,:2:,:,z_idx], axis=0)

            ## Mean pred error on full length recursive prediction on example trajs
            pred_error_vals = []

            for pred_errors in example_pred_errors[i,j]:
                max_non_nan_idx = (~np.isnan(pred_errors)).cumsum().argmax()
                pred_error_vals.append(pred_errors[max_non_nan_idx])

            mean_pred_errors[i, j, 2] = np.nanmean(np.absolute(pred_error_vals))
            std_pred_errors[i, j, 2] = np.nanstd(np.absolute(pred_error_vals))

            cell_text_full[j][i] = f"{round(mean_pred_errors[i,j,2],1)} \u00B1 {round(std_pred_errors[i,j,2],1)}"
            #### N step Disagreement and Prediction Error ####
            #### We only do it on example trajs as test trajs changes... between reps ####

            run_name = f'{env_name}_{init_method}_{init_episode}'
            
            fig_path_pred_error = os.path.join(args.dump_path, f'{run_name}/pred_error')
            os.makedirs(fig_path_pred_error, exist_ok=True)
        
            ## n = 1

            mean_pred_errors[i, j, 0] = np.nanmean(np.absolute(example_1_step_pred_errors))
            std_pred_errors[i, j, 0] = np.nanstd(np.absolute(example_1_step_pred_errors))
            cell_text_1_step[j][i] = f"{round(mean_pred_errors[i,j,0],3)} \u00B1 {round(std_pred_errors[i,j,0],3)}"

            ## n = 20

            mean_pred_errors[i, j, 1] = np.nanmean(np.absolute(example_20_step_pred_errors))
            std_pred_errors[i, j, 1] = np.nanstd(np.absolute(example_20_step_pred_errors))
            cell_text_20_step[j][i] = f"{round(mean_pred_errors[i,j,1],3)} \u00B1 {round(std_pred_errors[i,j,1],3)}"

            cum_pred_error = np.empty((task_h))

            ## Compute mean cumulated pred error on trajectories
            mean_along_traj = np.nanmean(np.absolute(example_1_step_pred_errors), axis=(0,2))
            for t in range(0, task_h):
                cum_pred_error[t] = np.nansum(mean_along_traj[:t])

            cum_pred_errors[i, j] = cum_pred_error

            #### Mean Disagreement, Mean Prediction Error, Prediction Error vs Disagreement ####
            ## Do small trick for label because data has a wrong name lol
            label_method = f'{init_method}'
            if label_method == 'brownian-motion':
                label_method = 'uniform-random-walk'
            if label_method == 'colored-noise-beta-0':
                label_method = 'brownian-motion'
            label = f'{label_method}_{init_episode}'

            ## On example trajs
            ## Compute mean and stddev of trajs prediction error
            example_mean_pred_error = np.nanmean(example_pred_errors[i,j], axis=0)
            example_std_pred_error = np.nanstd(example_pred_errors[i,j], axis=0)

            ## Update plot params
            if min(example_mean_pred_error) < example_limits_pred_error[2]:
                example_limits_pred_error[2] = min(example_mean_pred_error)
            if max(example_mean_pred_error) > example_limits_pred_error[3]:
                example_limits_pred_error[3] = max(example_mean_pred_error)
            ## Figure for pred_error
            example_ax_pred_error.plot(range(max_step), example_mean_pred_error,
                                       color=colors.to_rgba(i),
                                       linestyle=linestyles[i],
                                       label=label)

            ## Figure for cumulated pred_error
            example_ax_cum_pred_error.plot(range(max_step), cum_pred_error,
                                           color=colors.to_rgba(i),
                                           linestyle=linestyles[i],
                                           label=label)

            ## Figure for plotting trajectories
            example_ax_traj_plot.plot(mean_example_traj_x,
                                      mean_example_traj_y,
                                      color=colors.to_rgba(i),
                                      linestyle=linestyles[i],
                                      label=label)
            if args.environment == 'ball_in_cup':
                example_ax_traj_plot.plot(mean_example_traj_x,
                                          mean_example_traj_y,
                                          mean_example_traj_z,
                                          color=colors.to_rgba(i),
                                          linestyle=linestyles[i],
                                          label=label)
            
            print(f"\nPlotted for init_method {init_method} and init_episode {init_episode}\n")
            ## init method = i; init episode = j

        example_limits_pred_error = [0, max_step,
                                     0, args.pred_err_plot_upper_lim]

        ## Plot example traj
        ## Figure for plotting trajectories
        example_ax_traj_plot.plot(n_step_visualizer.test_trajectories[0, :, x_idx],
                                  n_step_visualizer.test_trajectories[0, :, y_idx],
                                  color='black',
                                  linestyle=linestyles[i+1],
                                  label='ground_truth')
        if args.environment == 'ball_in_cup':
            example_ax_traj_plot.plot(n_step_visualizer.test_trajectories[0, :, x_idx],
                                      n_step_visualizer.test_trajectories[0, :, y_idx],
                                      n_step_visualizer.test_trajectories[0, :, z_idx],
                                      color='black',
                                      linestyle=linestyles[i+1],
                                      label='ground_truth')

        
        ## Plot params
        # Set plot labels
        example_ax_pred_error.set_xlabel(example_labels_pred_error[0])
        example_ax_pred_error.set_ylabel(example_labels_pred_error[1])

        example_ax_cum_pred_error.set_xlabel(example_labels_cum_pred_error[0])
        example_ax_cum_pred_error.set_ylabel(example_labels_cum_pred_error[1])

        example_ax_traj_plot.set_xlabel(example_labels_traj_plot[0])
        example_ax_traj_plot.set_ylabel(example_labels_traj_plot[1])

        ## Set log scale if fastsim
        if args.environment == 'fastsim_maze' or args.environment == 'fastsim_maze_traps':
            example_ax_pred_error.set_yscale('log')
                               
            example_ax_cum_pred_error.set_yscale('log')

        ## Set plot limits
        x_min = example_limits_pred_error[0]; x_max = example_limits_pred_error[1];
        y_min = example_limits_pred_error[2]; y_max = example_limits_pred_error[3]
        
        example_ax_pred_error.set_xlim(x_min,x_max)
        example_ax_pred_error.set_ylim(y_min,y_max)

        example_ax_cum_pred_error.set_xlim(x_min,x_max)
        example_ax_cum_pred_error.set_ylim(y_min,y_max)

        x_min = example_limits_traj_plot[0]; x_max = example_limits_traj_plot[1];
        y_min = example_limits_traj_plot[2]; y_max = example_limits_traj_plot[3]

        example_ax_traj_plot.set_xlim(x_min,x_max)
        example_ax_traj_plot.set_ylim(y_min,y_max)

        ## Set legend
        
        example_ax_pred_error.legend()
        example_ax_cum_pred_error.legend()
        example_ax_traj_plot.legend()

        # example_ax_cum_pred_error.legend()

        ## Set plot title
        example_ax_pred_error.set_title(f"Mean prediction error along example trajectories with {init_episode} budget")

        example_ax_cum_pred_error.set_title(f"Cumulated mean prediction error along example trajectories with {init_episode} budget")

        example_ax_traj_plot.set_title(f"Predicted trajectory vs example trajectory with {init_episode} budget")
        
        ## Save fig        
        example_fig_pred_error.savefig(f"{args.dump_path}/{args.environment}_example_trajectories_pred_error_{init_episode}",
                                       bbox_inches='tight')        

        example_fig_cum_pred_error.savefig(f"{args.dump_path}/{args.environment}_example_trajectories_cum_pred_error_{init_episode}",
                                       bbox_inches='tight')        

        example_fig_traj_plot.savefig(f"{args.dump_path}/{args.environment}_example_trajectories_cum_traj_plot_{init_episode}",
                                       bbox_inches='tight')        
        ## Clear figs
        example_ax_pred_error.cla()
        example_ax_cum_pred_error.cla()
        example_ax_cum_traj_plot.cla()

    ## Save aggregated data
    np.savez("pretrained_pred_error_data.npz",
             mean_pred_errors=mean_pred_errors,
             std_pred_errors=std_pred_errors,
             cum_pred_errors=cum_pred_errors)

    print(f'Saved file pred_error_data.npz')

    ###############################################################
    ###############################################################
    ##################### Pred error tables #######################
    ########################### below #############################
    ###############################################################

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
            
    # example_limits_pred_error = [0, max_step,
    #                              0, args.pred_err_plot_upper_lim]

    # ## Plot params
    # # Set plot labels
    # example_ax_pred_error.set_xlabel(example_labels_pred_error[0])
    # example_ax_pred_error.set_ylabel(example_labels_pred_error[1])

    # ## Set plot limits
    # x_min = example_limits_pred_error[0]; x_max = example_limits_pred_error[1];
    # y_min = example_limits_pred_error[2]; y_max = example_limits_pred_error[3]
    
    # example_ax_pred_error.set_xlim(x_min,x_max)
    # example_ax_pred_error.set_ylim(y_min,y_max)

    # ## Set legend
    # example_ax_pred_error.legend(prop={'size': 1})

    # ## Set plot title
    # example_ax_pred_error.set_title(f"Mean prediction error along example trajectories")
    # ## Save fig
    # example_fig_pred_error.savefig(f"{args.dump_path}/{args.environment}_example_trajectories_pred_error",
    #                            bbox_inches='tight')

    print()
    
