# Env imports
import gym
import diversity_algorithms.environments.env_imports ## Contains deterministic ant + fetch
import mb_ge ## Contains ball in cup
import redundant_arm ## contains redundant arm

# env_name = 'ball_in_cup'
# env_name = 'redundant_arm'
# env_name = 'fastsim_maze'
env_name = 'fastsim_maze_traps'

gym_args = {}
if env_name == 'ball_in_cup':
    env_register_id = 'BallInCup3d-v0'
if env_name == 'redundant_arm':
    env_register_id = 'RedundantArmPosNoWallsLimitedAngles-v0'
if env_name == 'fastsim_maze':
    env_register_id = 'FastsimSimpleNavigationPos-v0'
    # gym_args['render'] = True

if env_name == 'fastsim_maze_traps':
    env_register_id = 'FastsimSimpleNavigationPos-v0'
    gym_args['physical_traps'] = True
    # gym_args['render'] = True

    
env = gym.make(env_register_id, **gym_args)

try:
    max_step = env._max_episode_steps
except:
    try:
        max_step = env.max_steps
    except:
        raise AttributeError("Env does not allow access to _max_episode_steps or to max_steps")

import model_init_study
import os
from model_init_study.controller.nn_controller \
        import NeuralNetworkController

module_path = os.path.dirname(model_init_study.__file__)
    
path_to_examples = os.path.join(module_path,
                                'examples/',
                                env_name+'_example_trajectories.npz')
obs = env.reset()
if isinstance(obs, dict):
        obs_dim = env.observation_space['observation'].shape[0]
else:
    obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
    
import numpy as np

examples_data = np.load(path_to_examples)
params = examples_data['params']

controller_params = \
{
    'controller_input_dim': obs_dim,
    'controller_output_dim': act_dim,
    'n_hidden_layers': 2,
    'n_neurons_per_hidden': 10
}
run_params = \
{
    'obs_dim': obs_dim,
    'action_dim': act_dim,
    
    'controller_type': NeuralNetworkController,
    'controller_params': controller_params,
    
    'action_min': -1,
    'action_max': 1,
    'action_init': 0,
    
    'policy_param_init_min': -5,
    'policy_param_init_max': 5,
    
    'path_to_test_trajectories': path_to_examples,
    
    'env': env,
    'env_max_h': max_step,
}

import time

for param in params:
    controller = NeuralNetworkController(run_params)
    controller.set_parameters(param)
    obs = env.reset()
    if 'fastsim_maze' in env_name:
        env = gym.make(env_register_id, **gym_args)
        obs = env.reset()
        env.enable_display()
    for t in range(max_step):
        a = controller(obs)
        # a = env.action_space.sample()
        obs, rew, done, info = env.step(a)
        env.render()
        if done:
            break
        print(t)
        time.sleep(0.001)
