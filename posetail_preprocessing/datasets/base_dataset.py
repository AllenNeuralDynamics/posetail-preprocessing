from abc import ABC, abstractmethod

class BaseDataset(ABC): 

    def __init__(self, dataset_path, dataset_outpath):
        self.dataset_path = dataset_path 
        self.dataset_outpath = dataset_outpath
        self.metadata = None

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
    def select_splits(self):
        pass 

    @abstractmethod
    def generate_dataset(self): 
        pass 

    def get_metadata(self):
        return self.metadata

    def set_metadata(self, df): 
        self.metadata = df

    def _subset_pose_dict(self, pose_dict, start_frame = 0, n_frames = None): 
        
        # subset coords to correspond to the portion of the dataset used
        pose = pose_dict['pose']

        if n_frames: 
            pose = pose[:, start_frame:start_frame + n_frames, :, :]
            pose_dict['pose'] = pose

        return pose_dict

    def _select_subset_for_split(self, split, n = None, random_state = 3): 

        # check number of videos in the split
        split_mask = self.metadata['split'] == split
        n_rows = len(self.metadata.loc[split_mask])

        # handle the case where None is passed for n, 
        # use all the data
        if n is None: 
            n = n_rows

        # only sample when we have more rows available than the number 
        # we are sampling 
        if n_rows and n_rows > n: 
            self.metadata.loc[split_mask, 'include'] = False
            split_ixs = self.metadata.loc[split_mask].sample(n = n, random_state = random_state).index
            self.metadata.loc[split_ixs, 'include'] = True