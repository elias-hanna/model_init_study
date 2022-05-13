from abc import abstractmethod
import numpy as np


class TrajectorySeparator():
    def __init__(self):
        pass

    def separate_traj(self, traj):
        """
        Separate each element of the trajectory into several subsets corresponding 
        to different parts of the state
        """
        separation = self.get_separation()
        separated_traj = []
        total_sep = 0
        for sep in separation:
            separated_traj.append(traj[:, total_sep:total_sep+sep])
            total_sep += sep
        return separated_traj
                       
    @abstractmethod
    def get_separation(self):
        """
        Returns the size of each subset of an element of a trajectory 
        note: same size for all elements
        [size_1, size_2, ..., size_n]
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_labels(self):
        """
        Returns the labels of each subset of an element of a trajectory 
        note: same labels for all elements
        [label_1, label_2, ..., label_n]
        """
        raise NotImplementedError
    
    def separate_trajs(self, trajs_to_sep):
        """
        trajs_to_sep in the np format (nb_of_trajs, nb_of_steps, state_dim)
        """
        separation = self.get_separation()
        separated_trajs = []
        for sep in separation:
            separated_trajs.append(np.empty((trajs_to_sep.shape[0],
                                             trajs_to_sep.shape[1],
                                             sep)))
        
        for i in range(len(trajs_to_sep)):
            separated_traj = self.separate_traj(trajs_to_sep[i])
            for j in range(len(separation)):
                separated_trajs[j][i] = separated_traj[j]
        labels = self.get_labels()
        
        return separated_trajs, labels
