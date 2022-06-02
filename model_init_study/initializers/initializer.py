## Abstract layer imports
from abc import abstractmethod

## Multiprocessing imports
from multiprocessing import cpu_count
from multiprocessing import Pool

# Other imports
import copy

class Initializer:
    def __init__(self, params):
        """
        Input: params (dict) consisting of the various parameters for the initializer
        Important parameters are: number of episodes; environment to run on
        Warning: this works only for vectorized observation and action spaces 
        """
        self._action_min = params['action_min']
        self._action_max = params['action_max']
        self._n_init_episodes = params['n_init_episodes']
        self._n_test_episodes = params['n_test_episodes']
        self._env_max_h = params['env_max_h']
        self._env = params['env']
        self._init_obs = self._env.reset()
        self._is_goal_env = False
        if isinstance(self._init_obs, dict):
            self._is_goal_env = True
            self._init_obs = self._init_obs['observation']
        self._obs_dim = params['obs_dim']
        self._act_dim = params['action_dim']
        self.nb_thread = cpu_count() - 1 or 1
        self.policies = None

    @abstractmethod
    def _get_action(self, obs, t):
        """
        Input: obs (observation vector, list), t (timesteps, int)
        Output: action (action vector, list)
        """
        raise NotImplementedError
    
    def _eval_policy(self, idx):
        """
        Input: idx (int) the index of the policy to eval in the Initializer
        Output: Trajectory and actions taken
        """
        env = copy.copy(self._env) ## need to verify this works
        obs = env.reset()
        transitions = []
        # traj = []
        # actions = []
        cum_rew = 0
        ## WARNING: need to get previous obs
        for t in range(self._env_max_h):
            if self._is_goal_env:
                obs = obs['observation']
            action = self._get_action(idx, obs, t)
            action[action>self._action_max] = self._action_max
            action[action<self._action_min] = self._action_min
            transitions.append((action, obs))
            obs, reward, done, info = env.step(action)
            cum_rew += reward
            if done:
                break
            # traj.append(obs)
            # actions.append(action)
        if self._is_goal_env:
            obs = obs['observation']
        transitions.append((None, obs))
        ## Return transitions
        return transitions
    
    def run(self):
        ## Setup multiprocessing pool
        pool = Pool(processes=self.nb_thread)
        ret = pool.starmap(self._eval_policy,
                           zip(range(self._n_init_episodes + self._n_test_episodes)))
        return ret
