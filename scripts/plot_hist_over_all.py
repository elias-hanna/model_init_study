import numpy as np
from multiprocessing import cpu_count

import time

def KLdivergence(x, y):
  """Compute the Kullback-Leibler divergence between two multivariate samples.
  Parameters
  ----------
  x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
  y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.
  Returns
  -------
  out : float
    The estimated Kullback-Leibler divergence D(P||Q).
  References
  ----------
  Pérez-Cruz, F. Kullback-Leibler divergence estimation of
  continuous distributions IEEE International Symposium on Information
  Theory, 2008.
  """
  # x=x[:1000]
  # y=y[:1000]
  from scipy.spatial import cKDTree as KDTree
  # Check the dimensions are consistent
  x = np.atleast_2d(x)
  y = np.atleast_2d(y)

  n,d = x.shape
  m,dy = y.shape

  assert(d == dy)

  # Build a KD tree representation of the samples and find the nearest neighbour
  # of each point in x.
  xtree = KDTree(x)
  ytree = KDTree(y)

  # Get the first two nearest neighbours for x, since the closest one is the
  # sample itself.
  nn_x = xtree.query(x, k=2, eps=.01, p=2, workers=cpu_count()-1)
  nn_y = ytree.query(x, k=1, eps=.01, p=2, workers=cpu_count()-1)
  r = nn_x[0][:,1]
  s = nn_y[0]

  # r = xtree.query(x, k=2, eps=.01, p=2, workers=cpu_count()-1)[0][:,1]
  # s = ytree.query(x, k=1, eps=.01, p=2, workers=cpu_count()-1)[0]

  # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
  # on the first term of the right hand side.
  ## quick fix to prevent infs put them at 0 so they are ignored...
  r[r==np.inf] = 0
  s[s==np.inf] = 0
  ## quick fix to prevent division by zero
  ## while keeping the identical samples between the two distributions
  r += 0.0000000001
  s += 0.0000000001
  return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))


def JSdivergence(x, y):
    from scipy.spatial import cKDTree as KDTree
  
    x=x[:1000]
    y=y[:1000]
    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    
    n,d = x.shape
    m,dy = y.shape
    
    assert(d == dy)

    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    xtree = KDTree(x)
    ytree = KDTree(y)
    
    ### Do x || m
    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    nn_x = xtree.query(x, k=2, eps=.01, p=2, workers=cpu_count()-1)
    nn_y = ytree.query(x, k=1, eps=.01, p=2, workers=cpu_count()-1)
    r = nn_x[0][:,1]
    s = nn_y[0]
    
    # r = xtree.query(x, k=2, eps=.01, p=2, workers=cpu_count()-1)[0][:,1]
    # s = ytree.query(x, k=1, eps=.01, p=2, workers=cpu_count()-1)[0]
    
    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.
    ## quick fix to prevent infs put them at 0 so they are ignored...
    r[r==np.inf] = 0
    s[s==np.inf] = 0
    ## quick fix to prevent division by zero
    ## while keeping the identical samples between the two distributions
    r += 0.0000000001
    s += 0.0000000001

    mu = 1/2*(r+s)

    kl_x_m = -np.log(r/mu).sum() * d / n + np.log(m / (n - 1.))

    ### Do y || m
    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    nn_x = xtree.query(y, k=1, eps=.01, p=2, workers=cpu_count()-1)
    nn_y = ytree.query(y, k=2, eps=.01, p=2, workers=cpu_count()-1)
    r = nn_x[0]
    s = nn_y[0][:,1]
    
    # r = xtree.query(x, k=2, eps=.01, p=2, workers=cpu_count()-1)[0][:,1]
    # s = ytree.query(x, k=1, eps=.01, p=2, workers=cpu_count()-1)[0]
    
    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.
    ## quick fix to prevent infs put them at 0 so they are ignored...
    r[r==np.inf] = 0
    s[s==np.inf] = 0
    ## quick fix to prevent division by zero
    ## while keeping the identical samples between the two distributions
    r += 0.0000000001
    s += 0.0000000001

    mu = 1/2*(r+s)

    kl_y_m = -np.log(s/mu).sum() * d / n + np.log(m / (n - 1.))
    
    return 1/2*(kl_y_m + kl_x_m)
    
############################################################################################
############################################################################################
############################################################################################

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

    ## Load 10 NS (all evaluated individuals) data
    ns_all_eval_data = np.load(f'{args.environment}_trajectories_all.npz')
    ns_all_eval_trajs = ns_all_eval_data['trajectories']
    # ns_all_eval_actions = ns_all_eval_data['actions']
    ns_all_eval_params = ns_all_eval_data['params']

    ## Format data of actions and observations
    # n_actions = ns_all_eval_actions.shape[0] * ns_all_eval_actions.shape[1]
    n_obs = ns_all_eval_trajs.shape[0] * ns_all_eval_trajs.shape[1]
    n_trans = ns_all_eval_trajs.shape[0] * (ns_all_eval_trajs.shape[1] - 1)

    # form_ns_actions = np.empty((n_actions, act_dim)) 
    form_ns_obs = np.empty((n_obs, obs_dim))
    form_ns_ds = np.empty((n_trans, obs_dim))
    curr_ptr = 0
    curr_ds_ptr = 0

    for i in range(len(ns_all_eval_trajs)):
        # form_ns_actions[curr_ptr:curr_ptr+len(ns_all_eval_actions[i])] = ns_all_eval_actions[i]
        form_ns_obs[curr_ptr:curr_ptr+len(ns_all_eval_trajs[i])] = ns_all_eval_trajs[i]
        curr_ptr += len(ns_all_eval_trajs[i])
        form_ns_ds[curr_ds_ptr:curr_ds_ptr+len(ns_all_eval_trajs[i])-1] = ns_all_eval_trajs[i][1:] - ns_all_eval_trajs[i][:-1]
        curr_ds_ptr += len(ns_all_eval_trajs[i])-1

    ## Plot table with delta S mean and stddev (prediction target) + delta A mean and stddev    
    column_headers = [init_method for init_method in init_methods]
    row_headers = [init_episode for init_episode in init_episodes]
    cell_text = [["" for _ in range(len(column_headers))] for _ in range(len(row_headers))]
    rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))

    ## Plot table with Jensen Shannon divergence for NS vs considered data distribution
    cell_text_js = [["" for _ in range(len(column_headers))] for _ in range(len(row_headers))]
    rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))

    cpt_tab = 0
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
        n_trans = 0
        for i in range(len(actions[0])):
            for j in range(len(actions[0][i])):
                n_actions += len(actions[0][i][j])
                n_obs += len(observations[0][i][j])
                n_trans += len(observations[0][i][j]) - 1

        n_c_actions = 0
        n_c_obs = 0
        n_c_trans = 0
        for i in range(len(actions[1])):
            for j in range(len(actions[1][i])):
                n_c_actions += len(actions[1][i][j])
                n_c_obs += len(observations[1][i][j])
                n_c_trans += len(observations[1][i][j]) - 1
                
        ## Fill numpy arrays
        form_actions = np.empty((n_actions, act_dim)) 
        form_obs = np.empty((n_obs, obs_dim))
        form_ds = np.empty((n_trans, obs_dim))
        curr_ptr = 0
        curr_ds_ptr = 0
        for i in range(len(actions[0])):
            for j in range(len(actions[0][i])):
                form_actions[curr_ptr:curr_ptr+len(actions[0][i][j])] = actions[0][i][j]
                form_obs[curr_ptr:curr_ptr+len(observations[0][i][j])] = observations[0][i][j]
                curr_ptr += len(actions[0][i][j])
                form_ds[curr_ds_ptr:curr_ds_ptr+len(observations[0][i][j])-1] = observations[0][i][j][1:] - observations[0][i][j][:-1]
                curr_ds_ptr += len(observations[0][i][j])-1

        form_c_actions = np.empty((n_c_actions, act_dim)) 
        form_c_obs = np.empty((n_c_obs, obs_dim))
        form_c_ds = np.empty((n_c_trans, obs_dim))
        curr_ptr = 0
        curr_ds_ptr = 0
        for i in range(len(actions[1])):
            for j in range(len(actions[1][i])):
                form_c_actions[curr_ptr:curr_ptr+len(actions[1][i][j])] = actions[1][i][j]
                form_c_obs[curr_ptr:curr_ptr+len(observations[1][i][j])] = observations[1][i][j]
                curr_ptr += len(actions[1][i][j])
                form_c_ds[curr_ds_ptr:curr_ds_ptr+len(observations[1][i][j])-1] = observations[1][i][j][1:] - observations[1][i][j][:-1]
                curr_ds_ptr += len(observations[0][i][j])-1

        
        ### PLOT DATA REPARTITION HISTOGRAM ###
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

        ### PLOT DELTA S AND A MEAN AND STDDEV ###
        # First init method
        mean_ds = round(np.mean(np.mean(form_ds, axis=0)), 6)
        std_ds = round(np.mean(np.std(form_ds, axis=0)), 6)
        
        cell_text[cpt_tab][0] = f"\u0394 s = {mean_ds} \u00B1 {std_ds}"

        print(f'{row_headers[cpt_tab]} ; {column_headers[0]} -> {cell_text[cpt_tab][0]}')

        # Second init method
        mean_ds = round(np.mean(np.mean(form_c_ds, axis=0)), 7)
        std_ds = round(np.mean(np.std(form_c_ds, axis=0)), 7)
        
        cell_text[cpt_tab][1] = f"\u0394 s = {mean_ds} \u00B1 {std_ds}"
        
        print(f'{row_headers[cpt_tab]} ; {column_headers[1]} -> {cell_text[cpt_tab][1]}')

        ### PLOT JENSEN SHANNON DIVERGENCE BETWEEN NS DATA DISTRIB AND CONSIDERED INIT ###
        # First init method
        js = JSdivergence(form_ns_obs, form_obs)
        
        cell_text_js[cpt_tab][0] = f"{js}"

        # Second init method
        js_c = JSdivergence(form_ns_obs, form_c_obs)
        
        cell_text_js[cpt_tab][1] = f"{js_c}"
        
        cpt_tab += 1
        
    ## Plot delta s and delta a variance
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    the_table = plt.table(cellText=cell_text,
                          rowLabels=row_headers,
                          rowColours=rcolors,
                          rowLoc='right',
                          colColours=ccolors,
                          colLabels=column_headers,
                          loc='center')
    fig.tight_layout()
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    plt.title(f'State and action mean and standard deviation on {args.environment} environment', y=.7)
    
    plt.savefig(f"{args.environment}_delta_s_a", dpi=300, bbox_inches='tight')
    ## Plot JS divergence
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    the_table = plt.table(cellText=cell_text_js,
                          rowLabels=row_headers,
                          rowColours=rcolors,
                          rowLoc='right',
                          colColours=ccolors,
                          colLabels=column_headers,
                          loc='center')
    fig.tight_layout()
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    plt.title(f'Jensen Shannon divergence between \n initialization methods data distributions and Novelty Search \n obtained data distribution on {args.environment} environment', y=.7)
    
    plt.savefig(f"{args.environment}_jensen_shannon_divergence", dpi=300, bbox_inches='tight')

    # plt.show()
