## Abstract layer imports
from abc import abstractmethod

## Multiprocessing imports
from multiprocessing import cpu_count
from multiprocessing import Pool

class Initializer:
    def __init__(self, params):
        """
        Input: params (dict) consisting of the various parameters for the initializer
        Important parameters are: number of episodes; environment to run on
        Warning: this works only for vectorized observation and action spaces 
        """
        self._env = params['env']
        self._init_obs = self._env.reset()
        self._obs_dim = self._env.observation_space.shape[0]
        self._act_dim = self._evn.action_space.shape[0]
        self._env_max_h = params['env_max_h']
        self._n_episodes = params['n_episodes']
        self._controller = params['controller_type'](params)
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
        ## Create a copy of the controller
        controller = self._controller.copy()
        ## Verify that x and controller parameterization have same size
        # assert len(x) == len(self.controller.get_parameters())
        ## Set controller parameters
        controller.set_parameters(x)

        env = copy.copy(self._env) ## need to verify this works
        obs = env.reset()
        traj = []
        actions = []
        cum_rew = 0
        ## WARNING: need to get previous obs
        for _ in range(self._env_max_h):
            action = self._get_action(idx, obs, t)
            obs, reward, done, info = env.step(action)
            cum_rew += reward
            traj.append(obs)
            actions.append(action)
        ## Return what?
        pass
    
    def eval_all_policies(self):
        ## Setup multiprocessing pool
        pool = Pool(processes=self.nb_thread)
        ## Get policy reprensation size
        policy_representation_dim = len(self.controller.get_parameters())
        ## Inits
        to_evaluate = []
        ## Random policy parametrizations creation
        for _ in range(self.nb_eval):
            ## Create a random policy parametrization 
            x = np.random.uniform(low=self.policy_param_init_min,
                                  high=self.policy_param_init_max,
                                  size=policy_representation_dim)
            to_evaluate += [x]
        # env_map_list = [gym_env_or_model for _ in range(self.nb_eval)]
        # env_map_list = [copy.deepcopy(gym_env_or_model._dynamics_model) for _ in range(self.nb_eval)]
        # env_map_list = [0. for _ in range(self.nb_eval)]
        ## Evaluate all generated policies on given environment
        elements = []
        # import pdb;pdb.set_trace()
        if eval_on_model:
            elements = self._eval_all_elements_on_model(to_evaluate, gym_env_or_model, prev_element)
            # elements = pool.starmap(eval_func, zip(to_evaluate, env_map_list,
                                                   # repeat(prev_element)))
            # for xx in to_evaluate:
                # elements.append(eval_func(xx, gym_env_or_model, prev_element))
        else:
            elements = pool.starmap(eval_func, zip(to_evaluate, repeat(gym_env_or_model),
                                                   repeat(prev_element))
