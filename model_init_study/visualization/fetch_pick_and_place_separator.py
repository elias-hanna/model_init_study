import traj_separators

class FetchPickAndPlaceSeparator(traj_separators.TrajectorySeparator):
    def __init__(self):
        pass

    def get_separation(self):
        """
        Returns the size of each subset of an element of a trajectory 
        note: same size for all elements
        [size_1, size_2, ..., size_n]
        """
        return [3, 3, 19]
        
    def get_labels(self):
        """
        Returns the labels of each subset of an element of a trajectory 
        note: same labels for all elements
        [label_1, label_2, ..., label_n]
        """
        return ['gripper_xyz', 'object_xyz', 'other_obs']
