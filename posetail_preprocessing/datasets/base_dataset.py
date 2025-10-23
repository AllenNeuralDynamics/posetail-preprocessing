from abc import ABC, abstractmethod

class BaseDataset(ABC): 

    def __init__(self, dataset_path, dataset_outpath):
        self.dataset_path = dataset_path 
        self.dataset_outpath = dataset_outpath

    @abstractmethod
    def load_calibration(self):
        pass

    @abstractmethod
    def load_pose3d(self):
        pass 

    @abstractmethod
    def generate_metadata(self):
        pass

    @abstractmethod
    def select_train_set(self):
        pass 

    @abstractmethod
    def select_test_set(self):  
        pass 

    @abstractmethod
    def generate_dataset(self): 
        pass 

    @abstractmethod
    def get_metadata(self):
        pass 