# Model Dependancies
from model_init_study.models.deterministic_model import DeterministicDynModel
from model_init_study.models.probabilistic_ensemble import ProbabilisticEnsemble
from model_init_study.utils.simple_replay_buffer import SimpleReplayBuffer
import model_init_study.utils.pytorch_util as ptu

# torch import
import torch

# Other includes
import numpy as np
import copy

class DynamicsModel():
    def __init__(self, params, dynamics_model=None):
        if dynamics_model is not None:
            self._dynamics_model = dynamics_model
            return
        
        self._process_params(params)
        
        ## INIT MODEL ##
        if self._dynamics_model_type == "prob":
            from model_init_study.models.mbrl import MBRLTrainer
            variant = dict(
                mbrl_kwargs=dict(
                    ensemble_size=self._ensemble_size,
                    layer_size=self._layer_size,
                    learning_rate=self._learning_rate,
                    batch_size=self._batch_size,
                )
            )
            M = variant['mbrl_kwargs']['layer_size']
            dynamics_model = ProbabilisticEnsemble(
                ensemble_size=variant['mbrl_kwargs']['ensemble_size'],
                obs_dim=self._obs_dim,
                action_dim=self._action_dim,
                hidden_sizes=[M, M])
            dynamics_model_trainer = MBRLTrainer(ensemble=dynamics_model,
                                                 **variant['mbrl_kwargs'],)

        # ensemble somehow cant run in parallel evaluations
        elif dynamics_model_type == "det":
            from model_init_study.models.mbrl_det import MBRLTrainer
            dynamics_model = DeterministicDynModel(obs_dim=self._obs_dim,
                                                   action_dim=self._action_dim,
                                                   hidden_size=self._layer_size)
            dynamics_model_trainer = MBRLTrainer(model=dynamics_model,
                                                 batch_size=self._batch_size,)
        self._dynamics_model = dynamics_model
        self._dynamics_model_trainer = dynamics_model_trainer

        # initialize replay buffer
        self._replay_buffer = SimpleReplayBuffer(max_replay_buffer_size=1000000,
                                                 observation_dim=self._obs_dim,
                                                 action_dim=self._action_dim,
                                                 env_info_sizes=dict(),)

    def _process_params(self, params):
        if 'dynamics_model_params' in params:
            if 'obs_dim' in params['dynamics_model_params']:
                self._obs_dim = params['dynamics_model_params']['obs_dim']
            else:
                raise Exception('DynamicsModel _process_params error: obs_dim not in params')
            if 'action_dim' in params['dynamics_model_params']:
                self._action_dim = params['dynamics_model_params']['action_dim']
            else:
                raise Exception('DynamicsModel _process_params error: action_dim not in params')
            if 'dynamics_model_type' in params['dynamics_model_params']:
                self._dynamics_model_type = params['dynamics_model_params']['dynamics_model_type']
            else:
                raise Exception('DynamicsModel _process_params error: dynamics_model_type not in params')
            if 'ensemble_size' in params['dynamics_model_params']:
                self._ensemble_size = params['dynamics_model_params']['ensemble_size']
            else:
                raise Exception('DynamicsModel _process_params error: ensemble_size not in params')
            if 'layer_size' in params['dynamics_model_params']:
                self._layer_size = params['dynamics_model_params']['layer_size']
            else:
                raise Exception('DynamicsModel _process_params error: layer_size not in params')
            if 'batch_size' in params['dynamics_model_params']:
                self._batch_size = params['dynamics_model_params']['batch_size']
            else:
                raise Exception('DynamicsModel _process_params error: batch_size not in params')
            if 'learning_rate' in params['dynamics_model_params']:
                self._learning_rate = params['dynamics_model_params']['learning_rate']
            else:
                raise Exception('DynamicsModel _process_params error: learning_rate not in params')
            if 'train_unique_trans' in params['dynamics_model_params']:
                self._train_unique_trans = params['dynamics_model_params']['train_unique_trans']
            else:
                raise Exception('DynamicsModel _process_params error: train_unique_trans not in params')
        else:
            raise Exception('DynamicsModel _process_params error: dynamics_model_params not in params')

    def forward_multiple(self, A, S, mean=True, disagr=True):
        ## Takes a list of actions A and a list of states S we want to query the model from
        ## Returns a list of the return of a forward call for each couple (action, state)
        assert len(A) == len(S)
        batch_len = len(A)
        ens_size = self._dynamics_model.ensemble_size
        
        S_0 = np.empty((batch_len*ens_size, S.shape[1]))
        A_0 = np.empty((batch_len*ens_size, A.shape[1]))

        batch_cpt = 0
        for a, s in zip(A, S):
            S_0[batch_cpt*ens_size:batch_cpt*ens_size+ens_size,:] = \
            np.tile(s,(self._dynamics_model.ensemble_size, 1))
            # np.tile(copy.deepcopy(s),(self._dynamics_model.ensemble_size, 1))

            A_0[batch_cpt*ens_size:batch_cpt*ens_size+ens_size,:] = \
            np.tile(a,(self._dynamics_model.ensemble_size, 1))
            # np.tile(copy.deepcopy(a),(self._dynamics_model.ensemble_size, 1))
            batch_cpt += 1
        # import pdb; pdb.set_trace()
        return self.forward(A_0, S_0, mean=mean, disagr=disagr, multiple=True)

        return batch_pred_delta_ns, batch_disagreement

    def forward(self, a, s, mean=True, disagr=True, multiple=False):
        s_0 = copy.deepcopy(s)
        a_0 = copy.deepcopy(a)

        if not multiple:
            s_0 = np.tile(s_0,(self._dynamics_model.ensemble_size, 1))
            a_0 = np.tile(a_0,(self._dynamics_model.ensemble_size, 1))

        s_0 = ptu.from_numpy(s_0)
        a_0 = ptu.from_numpy(a_0)
        
        # a_0 = a_0.repeat(self._dynamics_model.ensemble_size,1)

        # if probalistic dynamics model - choose output mean or sample
        if disagr:
            if not multiple:
                pred_delta_ns, disagreement = self._dynamics_model.sample_with_disagreement(
                    torch.cat((
                        self._dynamics_model._expand_to_ts_form(s_0),
                        self._dynamics_model._expand_to_ts_form(a_0)), dim=-1
                    ), disagreement_type="mean" if mean else "var")
                pred_delta_ns = ptu.get_numpy(pred_delta_ns)
                return pred_delta_ns, disagreement
            else:
                pred_delta_ns_list, disagreement_list = \
                self._dynamics_model.sample_with_disagreement_multiple(
                    torch.cat((
                        self._dynamics_model._expand_to_ts_form(s_0),
                        self._dynamics_model._expand_to_ts_form(a_0)), dim=-1
                    ), disagreement_type="mean" if mean else "var")
                for i in range(len(pred_delta_ns_list)):
                    pred_delta_ns_list[i] = ptu.get_numpy(pred_delta_ns_list[i])
                return pred_delta_ns_list, disagreement_list
        else:
            pred_delta_ns = self._dynamics_model.output_pred_ts_ensemble(s_0, a_0, mean=mean)
        return pred_delta_ns, 0

    def train(self, verbose=True):
        if verbose:
            import time
            start = time.time()
        torch.set_num_threads(24)
        self._dynamics_model_trainer.train_from_buffer(self._replay_buffer,
                                                       holdout_pct=0.1,
                                                       max_grad_steps=100000,
                                                       # epochs_since_last_update=10,
                                                       use_unique_transitions=self._train_unique_trans)
        self._dynamics_model_trainer.end_epoch(0) ## weird idk why they did it that way

        stats = self._dynamics_model_trainer.get_diagnostics()
        if verbose:
            print("=========================================\nDynamics Model Trainer statistics:")
            for name, value in zip(stats.keys(), stats.values()):
                print(name, ": ", value)
            model_train_time = time.time() - start
            print(f"Model train time: {model_train_time} seconds")
            print("=========================================\n")
            return stats
        
    def add_samples_from_transitions(self, transitions):
        A = []
        S = []
        NS = []
        for i in range(len(transitions) - 1):
            A.append(copy.copy(transitions[i][0]))
            S.append(copy.copy(transitions[i][1]))
            # NS.append(copy.copy(transitions[i+1][1] - transitions[i][1]))
            NS.append(copy.copy(transitions[i+1][1]))
        self.add_samples(S, A, NS)
        
    def add_samples(self, S, A, NS):
        for s, a, ns in zip(S, A, NS):
            self._replay_buffer.add_sample(s, a, 0, 0, ns, {})
            
    # Utils methods
    def normalize_standard(self, vector, mean_vector, std_vector):
        return [(vector[i] - mean_vector[i])/std_vector[i] for i in range(len(vector))]
    
    def rescale_standard(self, vector, mean_vector, std_vector):
        return [vector[i]*std_vector[i] + mean_vector[i] for i in range(len(vector))]
