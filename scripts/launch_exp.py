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

    from model_init_study.visualization.discretized_state_space_visualization \
        import DiscretizedStateSpaceVisualization
    from model_init_study.visualization.test_trajectories_visualization \
        import TestTrajectoriesVisualization

    # Env imports
    import diversity_algorithms.environments.env_imports ## Contains deterministic ant + fetch
    import mb_ge ## Contains ball in cup
    import redundant_arm ## contains redundant arm
    
    # Gym imports
    import gym

    # Argparse imports
    import argparse
    
    parser = argparse.ArgumentParser(description='Process run parameters.')

    parser.add_argument('--init-method', type=str, default='random-policies')

    parser.add_argument('--init-episodes', type=int, default='10')

    parser.add_argument('--action-lasting-steps', type=int, default='5')

    parser.add_argument('--dump-path', type=str, default='default_dump/')

    parser.add_argument('--environment', '-e', type=str, default='ball_in_cup')

    args = parser.parse_args()

    dynamics_model = DynamicsModel
    
    ## Framework methods
    
 
    if args.init_method == 'random-policies':
        Initializer = RandomPolicyInitializer
    elif args.init_method == 'random-actions':
        Initializer = RandomActionsInitializer

    env_register_id = 'BallInCup3d-v0'
    if args.environment == 'ball_in_cup':
        env_register_id = 'BallInCup3d-v0'
        ss_min = -0.4
        ss_max = 0.4
    if args.environment == 'redundant_arm':
        env_register_id = 'RedundantArm-v0'
        ss_min = -1
        ss_max = 1
    if args.environment == 'fetch_pick_and_place':
        env_register_id = 'FetchPickAndPlaceDeterministic-v1'
        ss_min = -1
        ss_max = 1
    if args.environment == 'ant':
        env_register_id = 'AntBulletEnvDeterministic-v0'
        ss_min = -10
        ss_max = 10
        
    env = gym.make(env_register_id)

    controller_params = \
    {
        'controller_input_dim': env.observation_space.shape[0],
        'controller_output_dim': env.action_space.shape[0],
        'n_hidden_layers': 2,
        'n_neurons_per_hidden': 10
    }
    dynamics_model_params = \
    {
        'obs_dim': env.observation_space.shape[0],
        'action_dim': env.action_space.shape[0],
        'dynamics_model_type': 'prob', # possible values: prob, det
        'ensemble_size': 4, # only used if dynamics_model_type == prob
        'layer_size': 500,
        'batch_size': 512,
        'learning_rate': 1e-3,
        'train_unique_trans': False,
    }
    params = \
    {
        'n_init_episodes': args.init_episodes,
        'n_test_episodes': int(.2*args.init_episodes), # 20% of n_init_episodes
        
        'controller_type': NeuralNetworkController,
        'controller_params': controller_params,

        'dynamics_model_params': dynamics_model_params,

        'action_min': -1,
        'action_max': 1,
        'action_lasting_steps': args.action_lasting_steps,

        'state_min': ss_min,
        'state_max': ss_max,
        
        'policy_param_init_min': -5,
        'policy_param_init_max': 5,
        
        'dump_path': args.dump_path,
        'path_to_test_trajectories': 'examples/'+args.environment+'_example_trajectories.npz',

        'env': env,
        'env_max_h': env._max_episode_steps,
    }
    
    ## Instanciate the initializer
    initializer = Initializer(params)

    ## Execute the initializer policies on the environment
    transitions = initializer.run()

    ## Separate training and test
    train_transitions = transitions[:-params['n_test_episodes']]
    test_transitions = transitions[-params['n_test_episodes']:]

    ## Train the model
    dynamics_model = DynamicsModel(params)
    # Add data to replay buffer
    for i in range(len(train_transitions)):
        dynamics_model.add_samples_from_transitions(train_transitions[i])

    # Actually train the model
    dynamics_model.train()
    ## Execute each visualizer routines
    params['model'] = dynamics_model # to pass down to the visualizer routines
    test_traj_visualizer = TestTrajectoriesVisualization(params)
    # discretized_ss_visualizer = DiscretizedStateSpaceVisualization(params)

    exit(0)
