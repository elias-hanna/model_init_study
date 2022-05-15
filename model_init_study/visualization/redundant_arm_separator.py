# import traj_separators
from model_init_study.visualization.traj_separators import TrajectorySeparator

# class RedundantArmSeparator(traj_separators.TrajectorySeparator):
class RedundantArmSeparator(TrajectorySeparator):
    def __init__(self):
        pass

    def get_separation(self):
        """
        Returns the size of each subset of an element of a trajectory 
        note: same size for all elements
        [size_1, size_2, ..., size_n]
        """
        return [25]
        
    def get_labels(self):
        """
        Returns the labels of each subset of an element of a trajectory 
        note: same labels for all elements
        [label_1, label_2, ..., label_n]
        """
        return ['articulations_angles']
