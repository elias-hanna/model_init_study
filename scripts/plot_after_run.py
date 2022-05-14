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
    module_path = os.path.dirname(model_init_study.__file__)

    parser = argparse.ArgumentParser(description='Process run parameters.')

    parser.add_argument('--init-method', type=str, default='random-policies')

    parser.add_argument('--init-episodes', type=int, default='10')

    parser.add_argument('--dump-path', type=str, default='default_dump/')

    parser.add_argument('--environment', '-e', type=str, default='ball_in_cup')

    args = parser.parse_args()

    dynamics_model = DynamicsModel
    
    ## Framework methods
 
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

        'n_init_episodes': args.init_episodes,
        # 'n_test_episodes': int(.2*args.init_episodes), # 20% of n_init_episodes
        'n_test_episodes': 2,
        
        'controller_type': NeuralNetworkController,
        'controller_params': controller_params,

        'dynamics_model_params': dynamics_model_params,

        'action_min': -1,
        'action_max': 1,

        'state_min': ss_min,
        'state_max': ss_max,
        
        'policy_param_init_min': -5,
        'policy_param_init_max': 5,
        
        'dump_path': args.dump_path,
        # 'path_to_test_trajectories': 'examples/'+args.environment+'_example_trajectories.npz',
        'path_to_test_trajectories': path_to_examples,

        'env': env,
        'env_max_h': env._max_episode_steps,
    }
    params['model'] = None
    
    rep_folders = next(os.walk(f'.'))[1]

    rep_folders = [x for x in rep_folders if (x.isdigit())]
    
    # import pdb; pdb.set_trace()

    rep_data = np.load(f'{rep_folders[0]}/{args.environment}_{args.init_method}_{args.init_episodes}_data.npz')

    tmp_data = rep_data['test_pred_trajs']

    ## Get parameters (should do a params.npz... cba)
    trajs_per_rep = len(tmp_data)
    n_total_trajs = trajs_per_rep*len(rep_folders)
    task_h = len(tmp_data[0])
    obs_dim = len(tmp_data[0][0])
    
    test_pred_trajs = np.empty((n_total_trajs, task_h, obs_dim))
    test_disagrs = np.empty((n_total_trajs, task_h))
    test_pred_errors = np.empty((n_total_trajs, task_h))

    examples_pred_trajs = np.empty((n_total_trajs, task_h, obs_dim))
    examples_disagrs = np.empty((n_total_trajs, task_h))
    examples_pred_errors = np.empty((n_total_trajs, task_h))


    rep_cpt = 0
    
    for rep_path in rep_folders:
        rep_data = np.load(f'{rep_path}/{args.environment}_{args.init_method}_{args.init_episodes}_data.npz')

        test_pred_trajs[rep_cpt*trajs_per_rep:
                        rep_cpt*trajs_per_rep + trajs_per_rep] = rep_data['test_pred_trajs']
        test_disagrs[rep_cpt*trajs_per_rep:
                     rep_cpt*trajs_per_rep + trajs_per_rep] = rep_data['test_disagrs']
        test_pred_errors[rep_cpt*trajs_per_rep:
                         rep_cpt*trajs_per_rep + trajs_per_rep] = rep_data['test_pred_errors']

        examples_pred_trajs[rep_cpt*trajs_per_rep:
                        rep_cpt*trajs_per_rep + trajs_per_rep] = rep_data['examples_pred_trajs']
        examples_disagrs[rep_cpt*trajs_per_rep:
                     rep_cpt*trajs_per_rep + trajs_per_rep] = rep_data['examples_disagrs']
        examples_pred_errors[rep_cpt*trajs_per_rep:
                         rep_cpt*trajs_per_rep + trajs_per_rep] = rep_data['examples_pred_errors']

        rep_cpt += 1
        
    test_model_trajs = (test_pred_trajs,
                        test_disagrs,
                        test_pred_errors,)
    
    examples_model_trajs = (examples_pred_trajs,
                            examples_disagrs,
                            examples_pred_errors,)

    test_traj_visualizer = TestTrajectoriesVisualization(params)

    test_traj_visualizer.dump_plots(args.environment,
                                    args.init_method,
                                    args.init_episodes,
                                    'all_test', model_trajs=test_model_trajs)

    test_traj_visualizer.dump_plots(args.environment,
                                    args.init_method,
                                    args.init_episodes,
                                    'all_examples', model_trajs=examples_model_trajs)

####################################################################################wip
    n_init_method = 2
    init_methods = ['random_policies', 'random_actions']
    n_init_episodes = 4
    init_episodes = [5, 10, 15, 20]
    
    test_pred_trajs = np.empty((n_init_method, n_init_episodes, n_total_trajs, task_h, obs_dim))
    test_disagrs = np.empty((n_init_method, n_init_episodes, n_total_trajs, task_h))
    test_pred_errors = np.empty((n_init_method, n_init_episodes, n_total_trajs, task_h))

    examples_pred_trajs = np.empty((n_init_method, n_init_episodes, n_total_trajs, task_h, obs_dim))
    examples_disagrs = np.empty((n_init_method, n_init_episodes, n_total_trajs, task_h))
    examples_pred_errors = np.empty((n_init_method, n_init_episodes, n_total_trajs, task_h))
    

    rep_cpt = 0
    
    for rep_path in rep_folders:
        for i in range(n_init_method):
            init_method =  init_methods[i]
            for j in range(n_init_episodes):
                init_episode = init_episodes[j]:
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
                
                examples_pred_trajs[i,j,rep_cpt*trajs_per_rep:
                                    rep_cpt*trajs_per_rep
                                    + trajs_per_rep] = rep_data['examples_pred_trajs']
                examples_disagrs[i,j,rep_cpt*trajs_per_rep:
                                 rep_cpt*trajs_per_rep
                                 + trajs_per_rep] = rep_data['examples_disagrs']
                examples_pred_errors[i,j,rep_cpt*trajs_per_rep:
                                     rep_cpt*trajs_per_rep
                                     + trajs_per_rep] = rep_data['examples_pred_errors']

        rep_cpt += 1
        
    test_model_trajs = (test_pred_trajs,
                        test_disagrs,
                        test_pred_errors,)
    
    examples_model_trajs = (examples_pred_trajs,
                            examples_disagrs,
                            examples_pred_errors,)

    test_traj_visualizer = TestTrajectoriesVisualization(params)

    test_traj_visualizer.dump_plots(args.environment,
                                    args.init_method,
                                    args.init_episodes,
                                    'all_test', model_trajs=test_model_trajs)

    test_traj_visualizer.dump_plots(args.environment,
                                    args.init_method,
                                    args.init_episodes,
                                    'all_examples', model_trajs=examples_model_trajs)
