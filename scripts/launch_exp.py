if __name__ == '__main__':
    # Local imports

    # Gym imports
    import gym

    # Argparse imports
    import argparse
    
    parser = argparse.ArgumentParser(description='Process run parameters.')

    parser.add_argument('--algorithm', type=str, default='ge')

    parser.add_argument('--budget', type=int, default=100000)

    parser.add_argument('--variable-horizon', action='store_true')

    args = parser.parse_args()

    dynamics_model = DynamicsModel
    
    ## Framework methods
    env = gym.make('BallInCup3d-v0')
        
    controller_params = \
    {
        'controller_input_dim': 6,
        'controller_output_dim': 3,
        'n_hidden_layers': 2,
        'n_neurons_per_hidden': 50
    }
    dynamics_model_params = \
    {
        'obs_dim': 6,
        'action_dim': 3,
        'dynamics_model_type': 'prob', # possible values: prob, det
        'ensemble_size': 4, # only used if dynamics_model_type == prob
        'layer_size': 500,
        'batch_size': 512,
        'learning_rate': 1e-3,
        'train_unique_trans': False,
    }
    params = \
    {
        'controller_type': NeuralNetworkController,
        'controller_params': controller_params,

        'action_min': -1,
        'action_max': 1,
        
        'policy_param_init_min': -5,
        'policy_param_init_max': 5,

        'dynamics_model_params': dynamics_model_params,
        
        'dump_path': args.dump_path,
        'dump_rate': dump_rate, # unused if dump_checkpoints used
        'dump_checkpoints': [10000, 20000, 50000, 100000, 200000, 500000, 1000000],
        'nb_of_samples_per_state':10,
        'dump_all_transitions': False,
        'env_max_h': env.max_steps,
    }
    

    ## Instanciate the initializer

    ## Execute the initializer policies on the environment

    ## Train the model

    ## Execute each visualizer routines
    
    pass
