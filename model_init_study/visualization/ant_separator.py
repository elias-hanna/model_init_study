class AntSeparator():
    def __init__(self):
        pass

    def get_separation(self):
        """
        Returns the size of each subset of an element of a trajectory 
        note: same size for all elements
        [size_1, size_2, ..., size_n]
        """
        raise NotImplementedError
    
    def get_labels(self):
        """
        Returns the labels of each subset of an element of a trajectory 
        note: same labels for all elements
        [label_1, label_2, ..., label_n]
        """
        raise NotImplementedError
