# import traj_separators
from model_init_study.visualization.traj_separators import TrajectorySeparator

# class AntSeparator(traj_separators.TrajectorySeparator):
class AntSeparator(TrajectorySeparator):
    def __init__(self):
        pass

    def get_separation(self):
        """
        Returns the size of each subset of an element of a trajectory 
        note: same size for all elements
        [size_1, size_2, ..., size_n]
        """
        return [1, 2, 3, 22]
        
    def get_labels(self):
        """
        Returns the labels of each subset of an element of a trajectory 
        note: same labels for all elements
        [label_1, label_2, ..., label_n]
        """
        return ['z', 'angle_to_target', 'xyz_body_speed', 'other_obs']
