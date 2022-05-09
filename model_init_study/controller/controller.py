from abc import abstractmethod

from copy import deepcopy

class Controller():
    def __init__(self, params=None):
        self._process_params(params)
        # self._init_controller(params)
        self.controller = None
        self.parameters = None ## Warning, this is the policy parameters not the class parameters uwu
        self.n_params = None
        
    def _process_params(self, params):
        if 'controller_params' in params:
            controller_params = params['controller_params']

            if 'controller_input_dim' in controller_params:
                self.input_dim = controller_params['controller_input_dim']
            else:
                raise ValueError('Controller _process_params error: no value passed for controller_input_dim')

            if 'controller_output_dim' in controller_params:
                self.output_dim = controller_params['controller_output_dim']
            else:
                raise ValueError('Controller _process_params error: no value passed for controller_output_dim')            
        else:
            raise Exception('Controller _process_params error: controller_params not in params')

    def copy(self):
        """
        Returns:
            copy: complete copy of the object
        """
        return deepcopy(self)
    
    @abstractmethod
    def compute_action(self, obs):
        """
        Args:
            obs: what the controller takes as input to compute actions, must be of dimensions equal to self.input_dim
        
        Returns:
            action: action vector, must be of dimensions equal to self.output_dim
        """
        raise NotImplementedError

    @abstractmethod
    def set_parameters(self, parameters):
        """
        Sets the controller parameters
        Args:
            parameters: 1 dimensionnal parameters vector
        """
        raise NotImplementedError

    @abstractmethod
    def get_parameters(self):
        """
        Gets the controller parameters
        Returns:
            parameters: 1 dimensionnal parameters vector
        """
        raise NotImplementedError
    
    def __call__(self, obs):
        return self.compute_action(obs)
