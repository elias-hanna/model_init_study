def process_args(args):
    environments = args.environments
    return environments
    
def main(args):
    environments = process_args(args)

    ## Get root working directory (from where py script is launched)
    root_wd = os.getcwd()

    envs_jsd_mean_divs = np.empty((len(environments)))
    envs_jsd_mean_divs[:] = np.nan
    envs_jsd_std_divs = np.empty((len(environments)))
    envs_jsd_std_divs[:] = np.nan
    ## Iterate over environments
    for (i, env) in zip(range(len(environments)), environments):
        ## Get env folder path
        env_path = os.path.join(root_wd, f"{env}_dynamics_uniformity_results")
        ## Get folders in env_path
        reps_path = next(os.walk(env_path))[1]
        ## Reconstruct absolute paths for repetitions folders
        reps_path = [os.path.join(env_path, rep_path) for rep_path in reps_path] 
        ## Iterate over the repetitions folder in the environment
        interest_inds = []
        if 'maze' in env:
            interest_inds = [0,1] ## x, y
        elif env == 'ball_in_cup':
            interest_inds = [0,1] ## x, y
        elif env == 'cartpole':
            interest_inds = [0,1] ## x, theta pole
        elif env == 'reacher':
            interest_inds = [i for i in range(17)]
        elif env == 'pusher':
            interest_inds = [i for i in range(20)]
            
        ## Init data structures
        env_jsd_mean_divs = np.empty((len(reps_path), len(interest_inds)))
        env_jsd_mean_divs[:] = np.nan
        env_jsd_std_divs = np.empty((len(reps_path), len(interest_inds)))
        env_jsd_std_divs[:] = np.nan
        for (j, rep_path) in zip(range(len(reps_path)), reps_path):
            jsd_data = np.load(f'{rep_path}/{env}_jsd_all_trans_vs_median.npz')
            jsd_divs_per_action = jsd_data['scipy_divs_per_action']
            jsd_mean_divs = jsd_data['scipy_mean_div']
            jsd_std_divs = jsd_data['scipy_std_div']
            ## Get the repetition jsd mean values and store it
            env_jsd_mean_divs[j] = jsd_mean_divs[interest_inds]
            env_jsd_std_divs[j] = jsd_std_divs[interest_inds]

        ## Perform mean of mean and std over repetitions for env and store it
        envs_jsd_mean_divs[i] = np.nanmean(env_jsd_mean_divs)
        envs_jsd_std_divs[i] = np.nanmean(env_jsd_std_divs)

    ## Order the environments from most uniform to least uniform
    ## (from low to high jsd value)
    sorted_inds = envs_jsd_mean_divs.argsort()
    envs_jsd_mean_divs = envs_jsd_mean_divs[sorted_inds]
    envs_jsd_std_divs = envs_jsd_std_divs[sorted_inds]
    environments = [environments[i] for i in sorted_inds]
    ## Plot boxplots of uniformity
    fig, ax = plt.subplots()

    x = [i for i in range(len(environments))]
    ax.errorbar(x, envs_jsd_mean_divs, yerr=envs_jsd_std_divs, fmt='o', capsize=4)

    plt.xticks(x, environments)

    ax.set_xlabel('Environments', fontsize=15)
    ax.set_ylabel('Uniformity measure (0 to 1, lower is more uniform)', fontsize=15)
    plt.title('Dynamics uniformity for various robotics and control environments', fontsize=20)
    plt.show()
    ## Save figure
    

        
            
if __name__ == '__main__':
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Process run parameters.')

    parser.add_argument('--environments', nargs="*", default=[])

    args = parser.parse_args()
    main(args)
