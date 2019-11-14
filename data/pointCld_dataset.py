import sys
sys.path.append('../')

import numpy as np
from torch.utils.data import Dataset
import torch

from structureHelpers import *
from utils.mode import Mode
import pipelines

class PointCld_dataset(Dataset):
    """
    This class is used to handle point cloud datasets. During software execution, only a reference to all point clouds
    and the current point cloud needs to be stored. The object returns in particular point coordinates and
    the ground truth segment for each point (if mode is not PREDICT).
    The user can choose a subset of the available point clouds.

    Refer: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """
    DEFAULT_DATA_PATH = read_config('./config.json')['input_data_path']

    def __init__(self, dataset_name, size, cfg_path, mode=Mode.TRAIN, dataset_parent_path=DEFAULT_DATA_PATH
                 , augmentation=None, seed=1, batch_size=1):
        """
        Inputs:
            dataset_name (string): Folder name of the dataset.
            size (int): Nr of point clouds used in the dataset
            mode (enumeration Mode):
                Nature of operation to be done with the data.
                Possibe inputs are Mode.PREDICT, Mode.TRAIN, Mode.TEST
                Default value: Mode.TRAIN
            dataset_parent_path (string):
                Path of the folder where the dataset folder is present
                Default: DEFAULT_DATA_PATH as per config.json
            cfg_path (string):
                Config file path of the experiment

            augmentation(Augmentation object):
                Augmentation to be applied on the dataset. #TODO: Provision only
            seed
                Seed used for random functions
                Default:1
            batch_size: Batch size used for loader. For point cloud dataset, use 1
        """
        params = read_config(cfg_path)
        self.cfg_path = params['cfg_path']

        self.mode = mode
        self.dataset_path = os.path.join(dataset_parent_path, dataset_name)
        self.dataset_name = dataset_name
        self.size = size
        self.augmentation = augmentation #TODO: Provision only
        self.batch_size = batch_size

        #Provision
        self.point_cloud_list = []
        self.use_preprocessed_patches = False

        # Initialize Database in order to get list of point clouds
        self._init_dataset(dataset_name, seed, params)

    def __len__(self):
        '''Returns length of the dataset'''
        return self.size

    def __getitem__(self, idx):
        '''
        Using self.point_cloud_list and the argument idx, return point clouds and labels if not predict mode.
        The point clouds and labels are returned in torch tensor format

        If preprocessed patches are available, only the name of the point cloud is returned, such that the
        corresponding patches can be found and loaded.

        Inputs:
            idx(int): index of point cloud that shall be returned.

        Outputs:
            point_cloud_coordinates, point_cloud_segments(tuple)
            point_cloud_coordinates, point_cloud_segments(tuple)
            file_name(str): Name of currently loaded point cloud according its filename
        '''

        file_name = self.point_cloud_list[idx]
        print('Loading ' + file_name)

        'if preprocessed patches available, only return the current point cloud name. Otherwise, load the point cloud'
        if self.use_preprocessed_patches:
            point_cloud = 0 #point cloud not used if patches available
            return point_cloud, file_name
        else:
            # Read point clouds using name. Note: Point cloud is array of Points where each point has
            # coordinates and a segment number
            point_cloud = np.load(os.path.join(self.dataset_path, self.point_cloud_list[idx]), allow_pickle=True)


            point_cloud_coordinates = np.array([i.getCoordinates() for i in point_cloud])


            # If mode is PREDICT, convert point cloud coordinates to tensor and return the point cloud coordinates
            if self.mode == Mode.PREDICT:
                # Convert point cloud coordinates to tensor and return it
                point_cloud_coordinates = torch.from_numpy(
                    point_cloud_coordinates).float()

                'return with additional brackets to have same dimension as in else case'
                return [point_cloud_coordinates], file_name

            # If mode is not PREDICT,  return point cloud coordinates and label (segment information)
            else:

                # Convert point cloud coordinates and segments to tensor and return them.
                # NOTE: Actually a matrix with relations between all points of the cloud would be required here,
                #   but storing matrix with size e.g. 100k x100k does not work. Matrix of relations will be calculated
                #   in each patch processing step.
                #point_cloud_coordinates = torch.from_numpy(point_cloud_coordinates).float()
                point_cloud_segments = np.array([i.getSegment() for i in point_cloud])

                return tuple((point_cloud_coordinates, point_cloud_segments)), file_name

    def _init_dataset(self, dataset_name, seed, params):
        '''
        Initialize Dataset: Get the list of point clouds from the dataset folder.

        If the dataset is found, the size is checked against the number of point clouds in the folder and
        if size is more than the number of point clouds in the folder, randomly pick point clouds from the folder so
        that number of point clouds is equal to the user specified size.

        If the dataset is not found, the simulator is run and a folder containing simulated point clouds
        is created with the given dataset name.

        Final point cloud list is stored into self.img_list

        Inputs:
            see in __init__()

        '''

        # Check if the dataset directory exists
        if os.path.isdir(self.dataset_path):

            # If dataset directory found: Collect the file names of all point clouds and store them to a list
            point_cloud_list = [name for name in os.listdir(self.dataset_path) if name.endswith('.npy')]
            point_cloud_count = len(point_cloud_list)
            # Compare point_cloud_count with size provided by user and then assign to list to self member

            # If number of point clouds < size: inform user about the available point clouds count
            # and change value of self.size to number of available point clouds

            if point_cloud_count < self.size:
                print(
                    "There are only " + str(point_cloud_count) + " point clouds in the folder.\n" +
                    str(point_cloud_count) + " point clouds are used.")
                self.point_cloud_list = point_cloud_list
                self.size = point_cloud_count


            # if number of point clouds > size : inform user about the available point clouds count
            # Randomly select point clouds from list and  assign them to self member such that self.point_clouds_list
            # would contain number of point clouds as specified by user.

            elif point_cloud_count > self.size:
                print("There are more than " + str(self.size) + " point clouds in the folder. " + str(
                    point_cloud_count) + " point clouds are available.\n" + str(
                    self.size) + " point clouds are randomly chosen from the available point clouds.")

                'use seed to randomize choice'
                np.random.seed(seed)

                # return self.size randomly chosen indices in the range from 1 to point_cloud_count
                rdm_indices = np.random.choice(point_cloud_count, self.size)
                self.point_cloud_list = [point_cloud_list[i] for i in rdm_indices]

            # If number of point clouds == size -> no changes
            else:
                self.point_cloud_list = point_cloud_list

        else:
            'Dataset does not exist. Call pipeline'
            print('Create new dataset.')
            pipelines.simulation_pipeline(params=params, batch_size=self.size, dataset_name=dataset_name,
                                          mode=self.mode, seed=seed)
            self.point_cloud_list = [name for name in os.listdir(self.dataset_path) if name.endswith('.npy')]

        # CODE FOR CONFIG FILE TO RECORD DATASETS USED
        # Save the dataset information by writing to config file
        if self.mode == Mode.TRAIN:
            params = read_config(self.cfg_path)
            params['Network']['total_dataset_number'] += 1
            dataset_key = 'Training_Dataset_' + str(params['Network']['total_dataset_number'])
            # If augmentation is applied
            if self.augmentation:
                #TODO: augmentation_applied = [i.__class__.__name__ for i in self.augmentation.augmentation_list]
                print('_init_dataset: self.augmentation should not have a value.')
            else:
                augmentation_applied = None
            dataset_info = {
                'name': dataset_name,
                'path': self.dataset_path,
                'size': self.size,
                'augmentation_applied': augmentation_applied,
                'seed': seed,
                'batch_size': self.batch_size
            }
            params[dataset_key] = dataset_info
            write_config(params, params['cfg_path'], sort_keys=True)
