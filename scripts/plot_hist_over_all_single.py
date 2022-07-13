import numpy as np
from multiprocessing import cpu_count
from scipy.spatial import cKDTree as KDTree
import random
import time
import copy

def format_data(actions, observations):
    ## Format data of actions and observations
    n_actions = 0
    n_obs = 0
    n_trans = 0
    for i in range(len(actions)):
        for j in range(len(actions[i])):
            n_actions += len(actions[i][j])
            n_obs += len(observations[i][j])
            n_trans += len(observations[i][j]) - 1
            
    ## Fill numpy arrays
    form_actions = np.empty((n_actions, act_dim)) 
    form_obs = np.empty((n_obs, obs_dim))
    form_ds = np.empty((n_trans, obs_dim))
    curr_ptr = 0
    curr_ds_ptr = 0
    for i in range(len(actions)):
        for j in range(len(actions[i])):
            form_actions[curr_ptr:curr_ptr+len(actions[i][j])] = actions[i][j]
            form_obs[curr_ptr:curr_ptr+len(observations[i][j])] = observations[i][j]
            curr_ptr += len(actions[i][j])
            form_ds[curr_ds_ptr:curr_ds_ptr+len(observations[i][j])-1] = observations[i][j][1:] - observations[i][j][:-1]
            curr_ds_ptr += len(observations[i][j])-1

    # Do some cleaning on the trajectories data
    form_actions = form_actions[np.isfinite(form_actions).any(axis=1)]
    form_obs = form_obs[np.isfinite(form_obs).any(axis=1)]
    form_ds = form_ds[np.isfinite(form_ds).any(axis=1)]

    return form_actions, form_obs, form_ds

## Call this piece of code from the top folder containing all env data ##
if __name__ == '__main__':
    # Local imports
    from model_init_study.visualization.state_space_repartition_visualization \
        import StateSpaceRepartitionVisualization

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
    import os
    import argparse
    import model_init_study
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    module_path = os.path.dirname(model_init_study.__file__)

    parser = argparse.ArgumentParser(description='Process run parameters.')

    parser.add_argument('--init-methods', nargs="*",
                        type=str, default=['brownian-motion', 'levy-flight', 'colored-noise-beta-1', 'colored-noise-beta-2', 'random-actions'])

    parser.add_argument('--init-episodes', nargs="*",
                        type=int, default=[5, 10, 20])
    
    parser.add_argument('--dump-path', type=str, default='default_dump/')

    parser.add_argument('--environment', '-e', type=str, default='ball_in_cup')

    args = parser.parse_args()

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
    else:
        raise ValueError(f"{args.environment} is not a defined environment")

    env = gym.make(env_register_id, **gym_args)

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

    init_methods = args.init_methods
    n_init_method = len(args.init_methods)
    init_episodes = args.init_episodes
    n_init_episodes = len(args.init_episodes)
        
    params = {'obs_dim': obs_dim}

    ssr_vis = StateSpaceRepartitionVisualization(params)

    ########################
    #### MAIN PLOT LOOP ####
    ########################
    for init_episode in init_episodes:
        actions_mins = None
        actions_maxs = None
        obs_mins = None
        obs_maxs = None
        all_actions = []
        all_observations = []
        #################################
        #### DETERMINE MINS AND MAXS ####
        #################################
        for init_method in init_methods:
            actions = []
            observations = []
            ## Create a folder for data corresponding to a single init_episode budget
            path = os.path.join(args.dump_path, f'{args.environment}_{init_episode}')
            os.makedirs(path, exist_ok=True)

            ## Aggregate data over all reps ##
            for rep_path in rep_folders:
                rep_data = np.load(f'{rep_path}/{args.environment}_{init_method}_{init_episode}_data.npz')
                actions.append(rep_data['train_actions'])
                observations.append(rep_data['train_trajs'])

            form_actions, form_obs, _ = format_data(actions, observations)
            
            if actions_mins is None:
                actions_mins = np.min(form_actions, axis=0)
            else:
                loc_actions_mins = np.min(form_actions, axis=0)
                action_mins = np.array([min(actions_mins[idx], loc_actions_mins[idx])
                               for idx in range(len(actions_mins))])
            if actions_maxs is None:
                actions_maxs = np.max(form_actions, axis=0)
            else:
                loc_actions_maxs = np.max(form_actions, axis=0)
                action_maxs = np.array([max(actions_maxs[idx], loc_actions_maxs[idx])
                               for idx in range(len(actions_maxs))])
            if obs_mins is None:
                obs_mins = np.min(form_obs, axis=0)
            else:
                loc_obs_mins = np.min(form_obs, axis=0)
                obs_mins = np.array([min(obs_mins[idx], loc_obs_mins[idx])
                               for idx in range(len(obs_mins))])
            if obs_maxs is None:
                obs_maxs = np.max(form_obs, axis=0)
            else:
                loc_obs_maxs = np.max(form_obs, axis=0)
                obs_maxs = np.array([max(obs_maxs[idx], loc_obs_maxs[idx])
                               for idx in range(len(obs_mins))])
                
        for init_method in init_methods:
            actions = []
            observations = []
            ## Create a folder for data corresponding to a single init_episode budget
            path = os.path.join(args.dump_path, f'{args.environment}_{init_episode}')
            os.makedirs(path, exist_ok=True)

            ## Aggregate data over all reps ##
            for rep_path in rep_folders:
                rep_data = np.load(f'{rep_path}/{args.environment}_{init_method}_{init_episode}_data.npz')
                actions.append(rep_data['train_actions'])
                observations.append(rep_data['train_trajs'])

            form_actions, form_obs, _ = format_data(actions, observations)

            # ### PLOT DATA REPARTITION HISTOGRAM ###
            # ssr_vis.set_trajectories(form_actions)
        
            # fig_path = os.path.join(path, f'{args.environment}_repartition_actions_{init_episode}_{init_method}')
            # ssr_vis.dump_plots(args.environment, '', init_episode, 'train', dim_type='action',
            #                    spe_fig_path=fig_path, legends=[init_method],
            #                    mins=action_mins, maxs=action_maxs)
            
            # ssr_vis.set_trajectories(form_obs)
            
            # fig_path = os.path.join(path, f'{args.environment}_repartition_obs_{init_episode}_{init_method}')
            # ssr_vis.dump_plots(args.environment, '', init_episode, 'train', dim_type='state',
            #                    spe_fig_path=fig_path, legends=[init_method],
            #                    mins=obs_mins, maxs=obs_maxs)

            all_actions.append(copy.copy(form_actions))
            all_observations.append(copy.copy(form_obs))

        ## Plot all on same plot

        ssr_vis.set_trajectories(all_actions)

        fig_path = os.path.join(path, f'{args.environment}_repartition_actions_{init_episode}')
        ssr_vis.dump_plots(args.environment, '', init_episode, 'train', dim_type='action',
                           spe_fig_path=fig_path, legends=init_methods,
                           mins=actions_mins, maxs=actions_maxs, plot_all=True)

        ssr_vis.set_trajectories(all_observations)

        fig_path = os.path.join(path, f'{args.environment}_repartition_obs_{init_episode}')
        ssr_vis.dump_plots(args.environment, '', init_episode, 'train', dim_type='state',
                           spe_fig_path=fig_path, legends=init_methods,
                           mins=obs_mins, maxs=obs_maxs, plot_all=True)
