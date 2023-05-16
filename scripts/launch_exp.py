if __name__ == '__main__':
    # Local imports
    from model_init_study.models.dynamics_model \
        import DynamicsModel

    from model_init_study.controller.nn_controller \
        import NeuralNetworkController

    from model_init_study.initializers.random_policy_initializer \
        import RandomPolicyInitializer
    from model_init_study.initializers.random_actions_initializer \
        import RandomActionsInitializer
    from model_init_study.initializers.random_actions_random_policies_hybrid_initializer \
        import RARPHybridInitializer
    from model_init_study.initializers.brownian_motion \
        import BrownianMotion
    from model_init_study.initializers.levy_flight \
        import LevyFlight
    from model_init_study.initializers.colored_noise_motion \
        import ColoredNoiseMotion
    
    # from model_init_study.visualization.discretized_state_space_visualization \
        # import DiscretizedStateSpaceVisualization
    from model_init_study.visualization.state_space_repartition_visualization \
        import StateSpaceRepartitionVisualization
    from model_init_study.visualization.test_trajectories_visualization \
        import TestTrajectoriesVisualization
    from model_init_study.visualization.n_step_error_visualization \
        import NStepErrorVisualization
    from model_init_study.visualization.dynamics_visualization \
        import DynamicsVisualization

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
    
    # Env imports
    import gym
    import diversity_algorithms.environments.env_imports ## Contains deterministic ant + fetch
    import mb_ge ## Contains ball in cup
    import redundant_arm ## contains redundant arm

    # Utils imports
    import numpy as np
    import argparse
    import os
    import model_init_study
    import types
    
    module_path = os.path.dirname(model_init_study.__file__)
    
    parser = argparse.ArgumentParser(description='Process run parameters.')

    parser.add_argument('--init-method', type=str, default='random-policies')

    parser.add_argument('--init-episodes', type=int, default='10')

    parser.add_argument('--step-size', type=float, default='0.1')

    parser.add_argument('--action-lasting-steps', type=int, default='5')

    parser.add_argument('--dump-path', type=str, default='default_dump/')

    parser.add_argument('--environment', '-e', type=str, default='ball_in_cup')

    args = parser.parse_args()

    dynamics_model = DynamicsModel
    
    ## Framework methods
    noise_beta = 2
    if args.init_method == 'random-policies':
        Initializer = RandomPolicyInitializer
    elif args.init_method == 'random-actions':
        Initializer = RandomActionsInitializer
    elif args.init_method == 'rarph':
        Initializer = RARPHybridInitializer
    elif args.init_method == 'brownian-motion':
        Initializer = BrownianMotion
    elif args.init_method == 'levy-flight':
        Initializer = LevyFlight
    elif args.init_method == 'colored-noise-beta-0':
        Initializer = ColoredNoiseMotion
        noise_beta = 0
    elif args.init_method == 'colored-noise-beta-1':
        Initializer = ColoredNoiseMotion
        noise_beta = 1
    elif args.init_method == 'colored-noise-beta-2':
        Initializer = ColoredNoiseMotion
        noise_beta = 2
    else:
        # raise Exception(f"Warning {args.init_method} isn't a valid initializer")
        raise Exception("Warning {} isn't a valid initializer".format(args.init_method))

    env_register_id = 'BallInCup3d-v0'
    gym_args = {}
    if args.environment == 'ball_in_cup':
        env_register_id = 'BallInCup3d-v0'
        separator = BallInCupSeparator
        ss_min = -0.4
        ss_max = 0.4
    elif args.environment == 'redundant_arm':
        env_register_id = 'RedundantArmPos-v0'
        separator = RedundantArmSeparator
        ss_min = -1
        ss_max = 1
    elif args.environment == 'redundant_arm_no_walls':
        env_register_id = 'RedundantArmPosNoWalls-v0'
        separator = RedundantArmSeparator
        ss_min = -1
        ss_max = 1
    elif args.environment == 'redundant_arm_no_walls_no_collision':
        env_register_id = 'RedundantArmPosNoWallsNoCollision-v0'
        separator = RedundantArmSeparator
        ss_min = -1
        ss_max = 1
    elif args.environment == 'redundant_arm_no_walls_limited_angles':
        env_register_id = 'RedundantArmPosNoWallsLimitedAngles-v0'
        separator = RedundantArmSeparator
        ss_min = -1
        ss_max = 1
        gym_args['dof'] = 100
    elif args.environment == 'fastsim_maze':
        env_register_id = 'FastsimSimpleNavigationPos-v0'
        separator = FastsimSeparator
        ss_min = -10
        ss_max = 10
    elif args.environment == 'fastsim_maze_traps':
        env_register_id = 'FastsimSimpleNavigationPos-v0'
        separator = FastsimSeparator
        ss_min = -10
        ss_max = 10
        gym_args['physical_traps'] = True
    elif args.environment == 'cartpole' or args.environment == 'pusher' \
       or args.environment == 'reacher':
        from dmbrl.config import create_config
        from dotmap import DotMap

        env_name = args.environment
        ctrl_args = []
        overrides = []
        logdir = args.dump_path

        ctrl_type = 'MPC'
        ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})

        cfg = create_config(env_name, ctrl_type, ctrl_args, overrides, logdir)

        env = cfg.ctrl_cfg.env

    else:
        # raise ValueError(f"{args.environment} is not a defined environment")
        raise ValueError("{} is not a defined environment".format(args.environment))
    
    
    # if args.environment == 'fetch_pick_and_place':
    #     env_register_id = 'FetchPickAndPlaceDeterministic-v1'
    #     separator = FetchPickAndPlaceSeparator
    #     ss_min = -1
    #     ss_max = 1
    # if args.environment == 'ant':
    #     env_register_id = 'AntBulletEnvDeterministicPos-v0'
    #     separator = AntSeparator
    #     ss_min = -10
    #     ss_max = 10
        
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
        'step_size': args.step_size,
        'noise_beta': noise_beta,
        
        'action_lasting_steps': args.action_lasting_steps,

        'state_min': ss_min,
        'state_max': ss_max,
        
        'policy_param_init_min': -5,
        'policy_param_init_max': 5,
        
        'dump_path': args.dump_path,
        # 'path_to_test_trajectories': 'examples/'+args.environment+'_example_trajectories.npz',
        'path_to_test_trajectories': path_to_examples,

        'env': env,
        'env_max_h': max_step,

        ## Dynamics visualizer specific params
        'sample_hor': 1,
        'sample_budget': 100,
        'num_cores': 10,
    }
    
    work_dir = os.getcwd()
    dump_dir = os.path.join(work_dir, args.dump_path)
    os.makedirs(dump_dir, exist_ok=True)
    
    ## Instanciate the initializer
    initializer = Initializer(params)

    ## Execute the initializer policies on the environment
    transitions = initializer.run()

    ## Separate training and test
    train_transitions = transitions[:-params['n_test_episodes']]
    test_transitions = transitions[-params['n_test_episodes']:]

    ## Format train actions and trajectories
    # Actions
    train_actions = np.empty((params['n_init_episodes'],
                              params['env_max_h'],
                              act_dim))
    train_actions[:] = np.nan

    for i in range(params['n_init_episodes']):
        traj_len = params['env_max_h'] if params['env_max_h'] < len(train_transitions[i]) \
                   else len(train_transitions[i])
        for j in range(traj_len):
            train_actions[i, j, :] = train_transitions[i][j][0]
    # Trajectories
    train_trajectories = np.empty((params['n_init_episodes'],
                                   params['env_max_h'],
                                   obs_dim))
    train_trajectories[:] = np.nan

    for i in range(params['n_init_episodes']):
        traj_len = params['env_max_h'] if params['env_max_h'] < len(train_transitions[i]) \
                   else len(train_transitions[i])
        for j in range(traj_len):
            train_trajectories[i, j, :] = train_transitions[i][j][1]


    ## Train the model
    trained = False
                    
    while not trained:
        dynamics_model = DynamicsModel(params)
        # Add data to replay buffer
        for i in range(len(train_transitions)):
            dynamics_model.add_samples_from_transitions(train_transitions[i])

        # Actually train the model
        stats = dynamics_model.train()
        if stats['Model Holdout Loss'] < 0:
            trained = True

    # work_dir = os.getcwd()
    # dump_dir = os.path.join(work_dir, args.dump_path)

    # wnb_path = os.path.join(dump_dir, f'{args.environment}_{args.init_method}_{args.init_episodes}_model_wnb.pt')
    wnb_path = os.path.join(dump_dir, '{}_{}_{}_model_wnb.pt'.format(args.environment, args.init_method, args.init_episodes))
    dynamics_model.save(wnb_path)

    ## Just for test
    # dynamics_model = DynamicsModel(params)
    # dynamics_model.load(wnb_path)
    
    ## Execute each visualizer routines
    params['model'] = dynamics_model # to pass down to the visualizer routines
    test_traj_visualizer = TestTrajectoriesVisualization(params)
    n_step_visualizer = NStepErrorVisualization(params)
    dynamics_visualizer = DynamicsVisualization(params)
    # env.set_state = types.MethodType(env.env.set_state.__func__, env)
    dynamics_visualizer.dump_plots(0)

    ## Visualize state space repartition (no need we plot it afterwards)
    # ssr_visualizer = StateSpaceRepartitionVisualization(params)
    # ssr_visualizer.set_trajectories(train_trajectories)
    # ssr_visualizer.dump_plots(args.environment, args.init_method, args.init_episodes, 'train')


    ## Visualize example trajectories
    # discretized_ss_visualizer = DiscretizedStateSpaceVisualization(params)
    ## Visualize n step error and disagreement ###

    n_step_visualizer.set_n(1)
    
    examples_1_step_trajs, examples_1_step_disagrs, examples_1_step_pred_errors = n_step_visualizer.dump_plots(
        args.environment,
        args.init_method,
        args.init_episodes,
        'examples', dump_separate=True, no_sep=True)

    # n_step_visualizer.set_n(5)

    # examples_5_step_trajs, examples_5_step_disagrs, examples_5_step_pred_errors = n_step_visualizer.dump_plots(
    #     args.environment,
    #     args.init_method,
    #     args.init_episodes,
    #     'examples', dump_separate=True, no_sep=True)

    # n_step_visualizer.set_n(10)

    # examples_10_step_trajs, examples_10_step_disagrs, examples_10_step_pred_errors = n_step_visualizer.dump_plots(
    #     args.environment,
    #     args.init_method,
    #     args.init_episodes,
    #     'examples', dump_separate=True, no_sep=True)

    n_step_visualizer.set_n(20)
    
    examples_20_step_trajs, examples_20_step_disagrs, examples_20_step_pred_errors = n_step_visualizer.dump_plots(
        args.environment,
        args.init_method,
        args.init_episodes,
        'examples', dump_separate=True, no_sep=True)

    ### Full recursive prediction visualizations ###
    examples_pred_trajs, examples_disagrs, examples_pred_errors = test_traj_visualizer.dump_plots(
        args.environment,
        args.init_method,
        args.init_episodes,
        'examples', dump_separate=True, no_sep=True)

    ## Format test trajectories
    # Trajectories
    test_trajectories = np.empty((params['n_test_episodes'],
                                  params['env_max_h'],
                                  obs_dim))
    test_trajectories[:] = np.nan

    for i in range(params['n_test_episodes']):
        traj_len = params['env_max_h'] if params['env_max_h'] < len(test_transitions[i]) \
                   else len(test_transitions[i])
        for j in range(traj_len):
            test_trajectories[i, j, :] = test_transitions[i][j][1]
    # Actions
    test_actions = np.empty((params['n_test_episodes'],
                                  params['env_max_h'],
                                  act_dim))
    test_actions[:] = np.nan

    for i in range(params['n_test_episodes']):
        traj_len = params['env_max_h'] if params['env_max_h'] < len(test_transitions[i]) \
                   else len(test_transitions[i])
        for j in range(traj_len):
            test_actions[i, j, :] = test_transitions[i][j][0]

    ## Visualize test trajectories

    # ## Visualize n step error and disagreement
    # n_step_visualizer.set_test_trajectories(test_trajectories)
    # test_n_step_trajs, test_n_step_disagrs, test_n_step_pred_errors = n_step_visualizer.dump_plots(
    #     args.environment,
    #     args.init_method,
    #     args.init_episodes,
    #     'test', dump_separate=True, no_sep=True)

    ### Full recursive prediction visualizations ###
    test_traj_visualizer.set_test_trajectories(test_trajectories)
    test_pred_trajs, test_disagrs, test_pred_errors = test_traj_visualizer.dump_plots(
        args.environment,
        args.init_method,
        args.init_episodes,
        'test', dump_separate=True, no_sep=True)

    data_path = os.path.join(
        args.dump_path,
        # f'{args.environment}_{args.init_method}_{args.init_episodes}_data.npz')
        '{}_{}_{}_data.npz'.format(args.environment, args.init_method, args.init_episodes))

    
    np.savez(data_path,
             test_pred_trajs=test_pred_trajs,
             test_disagrs=test_disagrs,
             test_pred_errors=test_pred_errors,
             examples_pred_trajs=examples_pred_trajs,
             examples_disagrs=examples_disagrs,
             examples_pred_errors=examples_pred_errors,
             # test_n_step_trajs=test_n_step_trajs,
             # test_n_step_disagrs=test_n_step_disagrs,
             # test_n_step_pred_errors=test_n_step_pred_errors,
             examples_1_step_trajs=examples_1_step_trajs,
             examples_1_step_disagrs=examples_1_step_disagrs,
             examples_1_step_pred_errors=examples_1_step_pred_errors,
             # examples_5_step_trajs=examples_5_step_trajs,
             # examples_5_step_disagrs=examples_5_step_disagrs,
             # examples_5_step_pred_errors=examples_5_step_pred_errors,
             # examples_10_step_trajs=examples_10_step_trajs,
             # examples_10_step_disagrs=examples_10_step_disagrs,
             # examples_10_step_pred_errors=examples_10_step_pred_errors,
             examples_20_step_trajs=examples_20_step_trajs,
             examples_20_step_disagrs=examples_20_step_disagrs,
             examples_20_step_pred_errors=examples_20_step_pred_errors,
             train_trajs=train_trajectories,
             train_actions=train_actions,
             test_trajs=test_trajectories,
             test_actions=test_actions,)

    print('\n###############################################################################\n')
    # print(f'Finished cleanly for {args.environment} environment with {args.init_method} init method and {args.init_episodes} episodes budget')
    print('Finished cleanly for {} environment with {} init method and {} episodes budget'.format(args.environment, args.init_method, args.init_episodes))
    print('\n###############################################################################\n')
    
    exit(0)
