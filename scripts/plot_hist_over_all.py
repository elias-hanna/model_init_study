## Call this piece of code from the top folder containing all env data ##
if __name__ == '__main__':
    # Local imports
    from model_init_study.visualization.state_space_repartition_visualization \
        import StateSpaceRepartitionVisualization

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

    parser.add_argument('--dump-path', type=str, default='default_dump/')

    parser.add_argument('--environment', '-e', type=str, default='ball_in_cup')

    args = parser.parse_args()

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

    init_methods = args.init_methods #['random-policies', 'random-actions']
    n_init_method = len(args.init_methods)
    init_episodes = args.init_episodes #[5, 10, 15, 20]
    n_init_episodes = len(args.init_episodes)
        
    params = {'obs_dim': obs_dim}

    ssr_vis = StateSpaceRepartitionVisualization(params)
    
    for init_episode in init_episodes:
        actions = [[], []]
        observations = [[], []]
        ## Create a folder for data corresponding to a single init_episode budget
        path = os.path.join(args.dump_path, f'{args.environment}_{init_episode}')
        os.makedirs(path, exist_ok=True)

        cpt_method = 0
        for init_method in init_methods:
            for rep_path in rep_folders:
                rep_data = np.load(f'{rep_path}/{args.environment}_{init_method}_{init_episode}_data.npz')
                actions[cpt_method].append(rep_data['train_actions'])
                observations[cpt_method].append(rep_data['train_trajs'])
            cpt_method += 1
            if cpt_method >= 2:
                # Only works for 2 concurrent init methods as of now
                break
        ## Format data of actions and observations
        n_actions = 0
        n_obs = 0
        for i in range(len(actions[0])):
            for j in range(len(actions[0][i])):
                n_actions += len(actions[0][i][j])
                n_obs += len(observations[0][i][j])

        n_c_actions = 0
        n_c_obs = 0
        for i in range(len(actions[1])):
            for j in range(len(actions[1][i])):
                n_c_actions += len(actions[1][i][j])
                n_c_obs += len(observations[1][i][j])

        ## Fill numpy arrays
        form_actions = np.empty((n_actions, act_dim)) 
        form_obs = np.empty((n_obs, obs_dim))
        curr_ptr = 0
        for i in range(len(actions[0])):
            for j in range(len(actions[0][i])):
                form_actions[curr_ptr:curr_ptr+len(actions[0][i][j])] = actions[0][i][j]
                form_obs[curr_ptr:curr_ptr+len(observations[0][i][j])] = observations[0][i][j]
                curr_ptr += len(actions[0][i][j])

        form_c_actions = np.empty((n_c_actions, act_dim)) 
        form_c_obs = np.empty((n_c_obs, obs_dim))
        curr_ptr = 0
        for i in range(len(actions[1])):
            for j in range(len(actions[1][i])):
                form_c_actions[curr_ptr:curr_ptr+len(actions[1][i][j])] = actions[1][i][j]
                form_c_obs[curr_ptr:curr_ptr+len(observations[1][i][j])] = observations[1][i][j]
                curr_ptr += len(actions[1][i][j])
                
        ssr_vis.set_trajectories(form_actions)
        ssr_vis.set_concurrent_trajectories(form_c_actions)
        
        fig_path = os.path.join(path, f'{args.environment}_repartition_actions_{init_episode}')
        ssr_vis.dump_plots(args.environment, '', init_episode, 'train',
                           spe_fig_path=fig_path, use_concurrent_trajs=True, legends=args.init_methods)

        ssr_vis.set_trajectories(form_obs)
        ssr_vis.set_concurrent_trajectories(form_c_obs)
        
        fig_path = os.path.join(path, f'{args.environment}_repartition_obs_{init_episode}')
        ssr_vis.dump_plots(args.environment, '', init_episode, 'train',
                           spe_fig_path=fig_path, use_concurrent_trajs=True, legends=args.init_methods)
