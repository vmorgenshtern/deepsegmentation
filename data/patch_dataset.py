import sys
sys.path.append("..")

import numpy as np
from torch.utils.data import Dataset

from utils.mode import Mode

class Patch_dataset(Dataset):
    """
    This class is used to handle point cloud datasets. During software execution, only a reference to all point clouds
    and the current point cloud needs to be stored. The object returns in particular point coordinates and
    the ground truth segment for each point (if mode is not PREDICT).
    The user can choose a subset of the available point clouds.

    Refer: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, patch_list, mode=Mode.TRAIN, seed=1):
        """
        Inputs:
            patch_list: list of patches
            dataset_name (string): Folder name of the dataset.
            mode (enumeration Mode):
                Nature of operation to be done with the data.
                Possibe inputs are Mode.PREDICT, Mode.TRAIN, Mode.TEST
                Default value: Mode.TRAIN
            seed
                Seed used for random functions
                Default:1
        """

        self.mode = mode
        self.patch_list = patch_list
        self.patch2_idx = None

    def __len__(self):
        '''Returns length of the dataset'''
        return self.patch_list.shape[0]

    def __getitem__(self, idx):
        '''
        Return combinations of patches

        Inputs:
            idx(int): index of patch that shall be returned together with self.patch2_idx
        '''

        'get a patch via idx'
        patch1 = self.patch_list[idx, :]
        patch1_coordinates = np.expand_dims(patch1[0], axis=0)

        'get second patch from outside'
        patch2 = self.patch_list[self.patch2_idx, :]
        patch2_coordinates = np.expand_dims(patch2[0], axis=0)

        'use relative shift or absolute shift by commenting respectively'
        shift = patch1[1] - patch2[1]
        #shift = np.abs(patch1[1] - patch2[1])

        if self.mode != Mode.PREDICT: #cannot assign label in predict mode
            if patch1[2] == patch2[2]:
                label = 1
            else:
                label = 0

            return patch1_coordinates, patch2_coordinates, shift, label

        else:
            return patch1_coordinates, patch2_coordinates, shift