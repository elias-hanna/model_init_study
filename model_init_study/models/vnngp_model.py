############################## 1 #######################################
import tqdm
import math
import torch
import gpytorch
from matplotlib import pyplot as plt

# Make plots inline
# %matplotlib inline

############################## 2 #######################################
import urllib.request
import os
from scipy.io import loadmat
from math import floor


# this is for running the notebook in our testing framework

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
    raise Exception(f"Warning {args.init_method} isn't a valid initializer")

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
else:
    raise ValueError(f"{args.environment} is not a defined environment")


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



reshaped_train_trajectories = train_trajectories.reshape((params['n_init_episodes']*
                                                          params['env_max_h'], obs_dim))
reshaped_train_actions = train_actions.reshape((params['n_init_episodes']*
                                                params['env_max_h'], act_dim))
reshaped_test_trajectories = test_trajectories.reshape((params['n_test_episodes']*
                                                        params['env_max_h'], obs_dim))
reshaped_test_actions = test_actions.reshape((params['n_test_episodes']*
                                              params['env_max_h'], act_dim))


# smoke_test = ('CI' in os.environ)

# if not smoke_test and not os.path.isfile('../elevators.mat'):
#     print('Downloading \'elevators\' UCI dataset...')
#     urllib.request.urlretrieve('https://drive.google.com/uc?export=download&id=1jhWL3YUHvXIaftia4qeAyDwVxo6j1alk', '../elevators.mat')


# if smoke_test:  # this is for running the notebook in our testing framework
#     X, y = torch.randn(100, 3), torch.randn(100)
# else:
#     data = torch.Tensor(loadmat('../elevators.mat')['data'])
#     X = data[:1000, :-1]
#     X = X - X.min(0)[0]
#     X = 2 * (X / X.max(0)[0].clamp_min(1e-6)) - 1
#     y = data[:1000, -1]
#     y = y.sub(y.mean()).div(y.std())
    
# train_n = int(floor(0.8 * len(X)))
# train_x = X[:train_n, :].contiguous()
# train_y = y[:train_n].contiguous()

# test_x = X[train_n:, :].contiguous()
# test_y = y[train_n:].contiguous()

train_x_obs = reshaped_train_trajectories[:-1,:]
train_x_act = reshaped_train_actions[:-1,:]
assert train_x_obs.shape[0] == train_x_act.shape[0]
train_x = np.empty((train_x_obs.shape[0], act_dim + obs_dim))
train_x[:, :act_dim] = train_x_act
train_x[:, act_dim:act_dim + obs_dim] = train_x_obs
train_y = reshaped_train_trajectories[1:,:] - reshaped_train_trajectories[:-1, :]

test_x_obs = reshaped_test_trajectories[:-1,:]
test_x_act = reshaped_test_actions[:-1,:]
assert test_x_obs.shape[0] == test_x_act.shape[0]
test_x = np.empty((test_x_obs.shape[0], act_dim + obs_dim))
test_x[:, :act_dim] = test_x_act
test_x[:, act_dim:act_dim + obs_dim] = test_x_obs
test_y = reshaped_test_trajectories[1:,:] - reshaped_test_trajectories[:-1, :]

## 1 dimensional output
train_x = train_x
train_y = train_y[:, 2]

test_x = test_x
test_y = test_y[:, 2]

## Numpify
train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)

test_x = torch.from_numpy(test_x)
test_y = torch.from_numpy(test_y)

if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()

############################## 3 #######################################
from gpytorch.models import ApproximateGP
from gpytorch.variational.nearest_neighbor_variational_strategy import NNVariationalStrategy


class GPModel(ApproximateGP):
    def __init__(self, inducing_points, likelihood, k=256, training_batch_size=256):

        m, d = inducing_points.shape
        self.m = m
        self.k = k

        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(m)

        if torch.cuda.is_available():
            inducing_points = inducing_points.cuda()

        variational_strategy = NNVariationalStrategy(self, inducing_points, variational_distribution, k=k,
                                                     training_batch_size=training_batch_size)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d)

        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, prior=False, **kwargs):
        if x is not None:
            if x.dim() == 1:
                x = x.unsqueeze(-1)
        return self.variational_strategy(x=x, prior=False, **kwargs)
smoke_test = False
if smoke_test:
    k = 32
    training_batch_size = 32
else:
    k = 256
    training_batch_size = 64

k = 5
likelihood = gpytorch.likelihoods.GaussianLikelihood()
# Note: one should use full training set as inducing points!
model = GPModel(inducing_points=train_x, likelihood=likelihood, k=k, training_batch_size=training_batch_size)

if torch.cuda.is_available():
    likelihood = likelihood.cuda()
    model = model.cuda()

############################## 4 #######################################
num_epochs = 1 if smoke_test else 20
num_batches = model.variational_strategy._total_training_batches


model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Our loss object. We're using the VariationalELBO
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))


# epochs_iter = tqdm.notebook.tqdm(range(num_epochs), desc="Epoch")
epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
for epoch in epochs_iter:
    # minibatch_iter = tqdm.notebook.tqdm(range(num_batches), desc="Minibatch", leave=False)
    minibatch_iter = tqdm.tqdm(range(num_batches), desc="Minibatch", leave=False)

    for i in minibatch_iter:
        optimizer.zero_grad()
        output = model(x=None)
        # Obtain the indices for mini-batch data
        current_training_indices = model.variational_strategy.current_training_indices
        # Obtain the y_batch using indices. It is important to keep the same order of train_x and train_y
        # import pdb; pdb.set_trace()
        y_batch = train_y[...,current_training_indices]
        # y_batch = train_y[current_training_indices, ...]
        if torch.cuda.is_available():
            y_batch = y_batch.cuda()
        loss = -mll(output, y_batch)
        minibatch_iter.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()

############################## 5 #######################################
from torch.utils.data import TensorDataset, DataLoader


test_dataset = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

############################## 6 #######################################
model.eval()
likelihood.eval()
means = torch.tensor([0.])
test_mse = 0
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        preds = model(x_batch)
        means = torch.cat([means, preds.mean.cpu()])

        diff = torch.pow(preds.mean - y_batch, 2)
        diff = diff.sum(dim=-1) / test_x.size(0) # sum over bsz and scaling
        diff = diff.mean() # average over likelihood_nsamples
        test_mse += diff
means = means[1:]
test_rmse = test_mse.sqrt().item()

############################## 7 #######################################
from torch.utils.data import TensorDataset, DataLoader


test_dataset = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

############################## 8 #######################################
model.eval()
likelihood.eval()
means = torch.tensor([0.])
test_mse = 0
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        preds = model(x_batch)
        means = torch.cat([means, preds.mean.cpu()])

        diff = torch.pow(preds.mean - y_batch, 2)
        diff = diff.sum(dim=-1) / test_x.size(0) # sum over bsz and scaling
        diff = diff.mean() # average over likelihood_nsamples
        test_mse += diff
means = means[1:]
test_rmse = test_mse.sqrt().item()

############################## 9 #######################################
print(test_rmse)

