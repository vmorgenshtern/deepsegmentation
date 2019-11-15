from tensorboardX import SummaryWriter
from torch.autograd import Variable
import datetime
import time
import csv

import random
import matplotlib.pyplot as plt

from structureHelpers import *
from Network.preprocessing import *
from Network.network import *
from Network.postprocessing import *

from utils.mode import Mode
from data import patch_dataset
from utils.probability_map import *
import utils.eval as eval

class Training:
    """
    This class is used to execute the training process of the network.
    """

    def __init__(self, cfg_path, cuda_device_id=0, torch_seed=None):
        """
        Inputs: cfg_path(string): Path to experiment's config file
                cuda_device_id(int): GPU number that shall be used. Range [0,...,7]
                torch_seed(int): Seed used for random generators in pytorch functions
        """
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path


        self.model_info = self.params['Network']
        self.model_info['seed'] = torch_seed or self.model_info['seed']
        if 'trained_time' in self.model_info:
            self.raise_training_complete_exception()

        'Setup the Cuda device'
        self.setup_cuda(cuda_device_id)
        self.writer = SummaryWriter(os.path.join(self.params['tf_logs_path']))

        'training steps'
        self.steps = 0

        'Provisions for model parameters. Are initialized via function setup_model.'
        self.model = None
        self.optimiser = None
        self.loss_function = None
        self.mode = None
        self.batch_size = 1
        self.start_time = 0

        self.preprocessing_finished = 0

    def setup_cuda(self, cuda_device_id):
        '''Setup the CUDA device'''
        torch.backends.cudnn.fastest = True
        torch.cuda.set_device(cuda_device_id)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.manual_seed_all(self.model_info['seed'])
        torch.manual_seed(self.model_info['seed'])

    def setup_model(self, model, convModel, weight_ratio, optimiser, optimiser_params, loss_function, nr_voxels_per_dim, batch_size):
        """
        Setup the model by defining the model, optimiser,loss function and learning rate.

        Inputs: model(network object): Network that is used for training
                convModel(str): Name of convolutional model(alexnet or resnet18)
                weight_ratio(int or float): weight ratio between related patches and not related patches
                optimiser: Optimizer object
                optimiser_params(Dictionary): Parameters of the optimiser
                loss_function: loss function
                nr_voxels_per_dim(int): voxelization parameter
                batch_size(int): batch_size used for patch_database
        """
        self.batch_size = batch_size
        self.add_tensorboard_graph(model(convModel, 1, 2, nr_voxels_per_dim, self.device), nr_voxels_per_dim)
        self.model = model(convModel, 1, 2, nr_voxels_per_dim, self.device)
        #if torch.cuda.device_count() > 1:
        #    self.model = nn.DataParallel(self.model, device_ids=[0, 1, 2, 3])
        self.model = self.model.to(self.device)
        self.optimiser = optimiser(self.model.parameters(), **optimiser_params)

        wr = weight_ratio
        loss_weight = torch.tensor([1, wr], dtype=torch.float).to(self.device)
        self.loss_function = loss_function(weight=loss_weight, reduction='sum')


        # Load model if retrain key is present in model info
        if 'retrain' in self.model_info and self.model_info['retrain'] == True:
            self.load_pretrained_model()

        'CODE FOR CONFIG FILE TO RECORD MODEL PARAMETERS'
        'Save the model, optimiser,loss function name for writing to config file'
        self.model_info['model_name'] = model.__name__
        self.model_info['weight_ratio'] = wr
        self.model_info['optimiser'] = optimiser.__name__
        self.model_info['loss_function'] = loss_function.__name__
        self.model_info['optimiser_params'] = optimiser_params
        self.model_info['nr_voxels_per_dim'] = nr_voxels_per_dim
        self.model_info['Patch_database_batch_size'] = self.batch_size
        self.model_info['relation_accept_threshold'] = 0.5 #the probability threshold to accept an entry as a valid relation
        self.model_info['Nr_compare_patches'] = 50 #number of patches to compare all patches with. To reduce training duration
        self.params['Network'] = self.model_info
        write_config(self.params, self.cfg_path, sort_keys=True)

    def add_tensorboard_graph(self, model, nr_voxels_per_dim):
        '''
        Creates a tensor board graph for network visualisation
        '''
        # Tensorboard Graph for network visualization

        'dummy variables to show tensor sizes in graph'
        dummy_input1 = Variable(torch.rand(self.batch_size, 1, nr_voxels_per_dim, nr_voxels_per_dim, nr_voxels_per_dim))
        dummy_input2 = Variable(torch.rand(self.batch_size, 1, nr_voxels_per_dim, nr_voxels_per_dim, nr_voxels_per_dim))
        dummy_shift = torch.rand(self.batch_size, 3, 1)
        self.writer.add_graph(model, (dummy_input1, dummy_input2, dummy_shift), True)

    def load_pretrained_model(self):
        'Load pre trained model to the using  pretrain_model_path parameter from config file'
        self.model.load_state_dict(torch.load(self.model_info['pretrain_model_path']))

    def raise_training_complete_exception(self):
        raise Exception("Model has already been trained on {}. \n"
                        "1.To use this model as pre trained model and train again\n "
                        "create new experiment using create_retrain_experiment function.\n\n"
                        "2.To start fresh with same experiment name, delete the experiment  \n"
                        "using delete_experiment function and create experiment "
                        "               again.".format(self.model_info['trained_time']))

    def execute_training(self, train_loader, validate_loader, num_epochs=None, do_preprocessing=False,
                         validate_frequency=1):
        """
        Execute training by running training for defined number of epochs.

        Inputs:
            train_loader, test_loader(dataset loader objects)
            do_preprocessing(bool): True, if all point clouds shall be preprocessed in advance. False, if point clouds
                    shall be preprocessed in each step again
            num_epochs(int): Number of epochs for which model shall be trained
            validate_frequency(int): Number of epochs after which model shall be evaluated on validation set.
        """

        'read param file again to include changes if any'
        self.params = read_config(self.cfg_path)

        'Check if already trained'
        if 'trained_time' in self.model_info:
            self.raise_training_complete_exception

        self.model_info = self.params['Network']
        self.model_info['num_epochs'] = num_epochs or self.model_info['num_epochs']

        'RUN THE TRAINING'
        print("Training started")
        self.start_time = time.time()

        for self.epoch in range(num_epochs):
            self.train_epoch(train_loader, do_preprocessing=do_preprocessing, save_to_disc=True)
            if self.epoch % validate_frequency == 0:
                self.validate_epoch(validate_loader, dataset_identifier=1, save_to_disc=True, visualize=False)
        print("Training finished")

        'Save model using path from model parameters and print path to console'
        torch.save(self.model.state_dict(),
                   os.path.join(self.params['network_output_path'], self.params['trained_model_name']))
        print('Model saved at: ' + str(
            os.path.join(self.params['network_output_path'], self.params['trained_model_name'])))

        'Update the config file with training information'
        self.model_info['num_steps'] = self.steps
        self.model_info['trained_time'] = "{:%B %d, %Y, %H:%M:%S}".format(datetime.datetime.now())
        if do_preprocessing:
            self.params['trained_with_preprocessing'] = True
        else:
            self.params['trained_with_preprocessing'] = False
        self.params['Network'] = self.model_info

        write_config(self.params, self.cfg_path, sort_keys=True)


    def train_epoch(self, train_loader, do_preprocessing, save_to_disc=False):
        """
        Train using one single iteration over all point clouds.

        Inputs:
            train_loader(torch.utils.data.DataLoader object): Training dataset
            do_preprocessing(bool): Whether to build patches everytime a point cloud is loaded or use precomputed patches
            save_to_disc(bool): Whether to store statistics on disc.
        """
        mode = Mode.TRAIN

        'print the epoch number and total epochs'
        print("Epoch [{}/{}] \n".format(self.epoch + 1, self.model_info['num_epochs']))

        'Create list to store loss value to display statistics'
        loss_list = []
        build_ref_matrix = True

        if do_preprocessing and self.preprocessing_finished == 0:
            'If path exists, write to config. Else: Preprocess voxelized patches and store to disk.'
            prep_path = os.path.join(train_loader.dataset.dataset_path, 'Preprocessed_patches')
            if os.path.isdir(prep_path):
                self.params["preprocessed_patches_path"] = prep_path
                write_config(self.params, self.cfg_path, sort_keys=True)
            else:
                print('Did not find preprocessed patches. Processing started.')
                self.params["preprocessed_patches_path"] = prep_path
                write_config(self.params, self.cfg_path, sort_keys=True)
                os.makedirs(self.params["preprocessed_patches_path"])

                for i, (point_cloud, file_name) in enumerate(train_loader):
                    file_name = file_name[0]
                    'extract patches and relations from given point cloud'
                    pc_preprocessor = Point_Cloud_Processing(Mode.TRAIN)
                    _, reference_matrix, _, voxelized_patches, _ = pc_preprocessor.apply_preprocessing(point_cloud,
                                                                                                    self.model_info[
                                                                                                        "nr_voxels_per_dim"],
                                                                                                    self.batch_size,
                                                                                                    build_ref_matrix=build_ref_matrix)

                    patches_path = os.path.join(self.params["preprocessed_patches_path"], file_name)
                    reference_matrix_path = os.path.join(self.params["preprocessed_patches_path"],
                                                         file_name[:-len('.npy')] + '_ref_matrix.npy')
                    'store to disk'
                    np.save(patches_path, voxelized_patches)
                    np.save(reference_matrix_path, reference_matrix)

            'After all point clouds are processed, set variable in dataset loader to only return file names'
            train_loader.dataset.use_preprocessed_patches = True

            'set variable to avoid calling preprocessing function again'
            self.preprocessing_finished = 1
            print('Preprocessed patches are available.')

        'iterate over the whole training dataset'
        for i, (point_cloud, file_name) in enumerate(train_loader):
            file_name = file_name[0]
            'if no preprocessed patches available, prepare the patches. If available, load from disk.'
            if not do_preprocessing:
                'extract patches and relations from given point cloud'
                pc_preprocessor = Point_Cloud_Processing(Mode.TRAIN)
                patch_loader, reference_matrix, _, _, _ = pc_preprocessor.apply_preprocessing(point_cloud,
                                                                                     self.model_info["nr_voxels_per_dim"],
                                                                                     self.batch_size,
                                                                                     build_ref_matrix=build_ref_matrix)
            else:
                voxelized_patches = np.load(os.path.join(self.params["preprocessed_patches_path"], file_name), allow_pickle=True)
                reference_matrix = np.load(os.path.join(self.params["preprocessed_patches_path"],
                                                        file_name[:-len('.npy')] + '_ref_matrix.npy'), allow_pickle=True)
                patch_data = patch_dataset.Patch_dataset(patch_list=voxelized_patches, mode=mode)
                patch_loader = torch.utils.data.DataLoader(dataset=patch_data,
                                                           batch_size=self.batch_size,
                                                           shuffle=True, num_workers=4)
            if reference_matrix is not None:
                print("Reference matrix has {} elements. Of those are {} elements 1, the others are 0.".format(reference_matrix.shape[0]**2, np.count_nonzero(reference_matrix)))

            'randomly sample indices that are used to find comparison patches. Used to avoid too lengthy training.'
            typical_length = self.params['Network']['Nr_compare_patches']  #worked well with 50 on small training sets. Found by experiment.
            nr_indices = min(typical_length, patch_loader.dataset.patch_list.shape[0])
            indices = random.sample(list(range(0, patch_loader.dataset.patch_list.shape[0])), nr_indices)

            for index in indices:
                patch_loader.dataset.patch2_idx = index
                loss = 0

                for j, (patch1, patch2, shift, label) in enumerate(patch_loader):

                    patch1 = patch1.to(self.device).float()
                    patch2 = patch2.to(self.device).float()
                    shift = shift.to(self.device).float()
                    label = label.to(self.device).long()

                    compare_patches = self.model(patch1, patch2, shift)

                    loss += self.loss_function(input=compare_patches,
                                              target=label)

                # Backward and optimize
                loss = loss / (patch_loader.dataset.patch_list.shape[0])
                loss_list.append(loss)
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

            print('Finished point cloud ' + str(i))
            'Save model after certain number of steps as specified in params[network_save_freq]'
            # Note: Prefix default filename with step number Eg: step_5_trained_model.pth
            if self.steps % self.params['network_save_freq'] == 0:
                torch.save(self.model.state_dict(), os.path.join(self.params['network_output_path'],
                                                                 'step_' + str(self.steps) + '_' + self.params[
                                                                     'trained_model_name']))

            'Print loss statistics after certain number of steps as specified in params[display_stats_freq].'
            if self.steps % self.params['display_stats_freq'] == 0:
                avg, maxL, minL = eval.calculate_loss_stats(loss_list, is_train=True,
                                                            writer=self.writer, steps=self.steps)
                'Print stats with 6 digits precision'
                print("Statistics in step %d: Average Loss: %.6f Maximum Loss: %.6f Minimum Loss: %.6f" % (
                self.steps, avg, maxL, minL))

                'save statistics to file'
                if save_to_disc:
                    if not os.path.exists(os.path.join(self.params["output_data_path"], 'Train_Statistics.csv')):
                        with open(os.path.join(self.params["output_data_path"], 'Train_Statistics.csv'), mode='w') as file:
                            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            writer.writerow(['Epoch', 'MinLoss', 'AvgLoss', 'MaxLoss', 'TimeElapsed'])

                    with open(os.path.join(self.params["output_data_path"], 'Train_Statistics.csv'), mode='a') as file:
                        writer = csv.writer(file, delimiter=',', quotechar='"',
                                            quoting=csv.QUOTE_MINIMAL)
                        hours, rem = divmod(time.time() - self.start_time, 3600)
                        minutes, seconds = divmod(rem, 60)
                        writer.writerow([str(self.epoch),
                                         str(minL),
                                         str(avg),
                                         str(maxL),
                                         str("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
                                         ])

                'reset loss list after number of steps as specified in params[display_stats_freq]'
                loss_list = []

            self.steps = self.steps + 1

    def validate_epoch(self, validate_loader, dataset_identifier, save_to_disc=True, visualize=False):
        """
        Test model after an epoch and calculate loss on validation dataset

        Inputs:
            validate_loader: dataset with test data
            dataset_identifier: 0 if function used with training set, 1 if function used with validation set
            save_to_disc(bool): whether to store statistics to disc
            visualize(bool): True if correlation matrix shall be plotted
        """
        mode = Mode.TEST
        if dataset_identifier == 0:
            print('Testing step with Training set')
        else:
            print('Testing step with Validation set')

        'Set model to evaluation mode. Refer : https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html'
        self.model.eval()

        'variables needed to decide if correlation matrix shall be visualized'
        dataset_indices = np.arange(validate_loader.dataset.size)
        if dataset_identifier == 0:
            'choose maximum 1 point cloud from training set for visualization'
            nr_choose = min(dataset_indices.size, 1)
        else:
            nr_choose = min(dataset_indices.size, 3) #not used
        idx_choose = np.random.choice(dataset_indices, nr_choose)

        file_counter = 0

        'Validation step.'
        with torch.no_grad():
            loss_list = []
            'accuracy lists for testing different thresholds'
            error_list_y = [[] for i in range(21)]
            error_list_x = np.arange(0, 1.05, 0.05)
            for i, (point_cloud, file_name) in enumerate(validate_loader):
                file_name = file_name[0]
                if i not in idx_choose and dataset_identifier == 0:
                    'test only small number of point cloud on training set'
                    continue
                'extract patches and relations from given point cloud'
                pc_preprocessor = Point_Cloud_Processing(Mode.TEST)
                build_ref_matrix = True
                patch_loader, reference_matrix, patches, voxelized_patches, _ = \
                    pc_preprocessor.apply_preprocessing(point_cloud, self.model_info["nr_voxels_per_dim"],
                                                        self.batch_size, build_ref_matrix=build_ref_matrix)

                relation_matrix = (torch.empty(voxelized_patches.shape[0], voxelized_patches.shape[0], 2).float()).to(self.device)
                reference_matrix = (torch.tensor(reference_matrix)).to(self.device)

                'loop through all patch combinations and apply network'
                for patch_idx in range(patch_loader.dataset.patch_list.shape[0]):
                    patch_loader.dataset.patch2_idx = patch_idx
                    loss = 0
                    intermediate = 0
                    for j, (patch1, patch2, shift, label) in enumerate(patch_loader):
                        # print('index: ' +str(index) + 'patch1 batch size: ' + str(patch1.shape[0]))

                        patch1 = patch1.to(self.device).float()
                        patch2 = patch2.to(self.device).float()
                        shift = shift.to(self.device).float()
                        label = label.to(self.device).long()

                        compare_patches = self.model(patch1, patch2, shift)

                        loss += self.loss_function(input=compare_patches,
                                                   target=label)

                        relation_matrix[patch_idx, intermediate:intermediate+patch2.shape[0]] = compare_patches
                        intermediate += patch2.shape[0]

                    loss = loss / patch_loader.dataset.patch_list.shape[0]
                    loss_list.append(loss)

                avg, maxL, minL = eval.calculate_loss_stats(loss_list, is_train=False,
                                                            writer=self.writer, steps=self.steps)
                'Print stats with 6 digits precision'
                print("Statistics in step %d: Average Loss: %.6f Maximum Loss: %.6f Minimum Loss: %.6f" % (
                    self.steps, avg, maxL, minL))
                loss_list = []

                relation_matrix = get_probability_map(relation_matrix)
                'get rid of NaN values that result from softmax. Attention: Do not use on backpropagated values.'
                relation_matrix[torch.isnan(relation_matrix)] = 0

                if save_to_disc:
                    relation_matrix = np.array(relation_matrix.tolist())
                    reference_matrix = np.array(reference_matrix.tolist())

                    'iterating over different thresholds to find best threshold'
                    counter = 0
                    test_relation_matrix = copy.deepcopy(relation_matrix)
                    for threshold in error_list_x:

                        test_relation_matrix[relation_matrix >= threshold] = 1
                        test_relation_matrix[relation_matrix < threshold] = 0
                        test_difference_matrix = np.abs(reference_matrix - test_relation_matrix)

                        error = np.count_nonzero(test_difference_matrix) / reference_matrix.size
                        error_list_y[counter].append(error)

                        counter += 1

                    relation_matrix[relation_matrix >= self.params['Network']['relation_accept_threshold']] = 1
                    relation_matrix[relation_matrix < self.params['Network']['relation_accept_threshold']] = 0
                    difference_matrix = np.abs(reference_matrix - relation_matrix)
                    accuracy = 1 - np.count_nonzero(difference_matrix) / reference_matrix.size

                    path = os.path.join(self.params['output_data_path'], 'relation_matrices_validate_' + str(self.epoch) + '_' +
                                         str(file_counter) + '.png')
                    while os.path.exists(path):
                        file_counter += 1
                        path = os.path.join(self.params['output_data_path'], 'relation_matrices_validate_' + str(self.epoch) + '_' +
                                             str(file_counter) + '.png')

                    print("Relation matrix {}, Reference matrix {}, difference matrix {}".format(str(relation_matrix.shape), str(reference_matrix.shape), str(difference_matrix.shape)))
                    if np.allclose(reference_matrix, reference_matrix.T, rtol=1e-05, atol=1e-05):
                        print("Reference matrix is symmetric")
                    else:
                        print("Reference matrix is not symmetric")
                    print("Classified {} relations. There were {} wrong classifications.".format(relation_matrix.size, np.count_nonzero(difference_matrix)))

                    'conversions before plotting to have only 1 and 0 as value. Otherwise' \
                    'uint8 difference is circular'
                    relation_matrix = np.asarray(relation_matrix, dtype="uint8")
                    reference_matrix = np.asarray(reference_matrix, dtype="uint8")
                    difference_matrix = np.asarray(difference_matrix, dtype="uint8")

                    f, axarr = plt.subplots(1, 3)
                    if (dataset_identifier == 0):
                        plt.title('Test on Training Set')
                    else:
                        plt.title('Test on Validation Set')
                    axarr[0].imshow(relation_matrix)
                    axarr[0].set_title('Estimated matrix')
                    axarr[1].imshow(reference_matrix)
                    axarr[1].set_title('Reference matrix')
                    axarr[2].imshow(difference_matrix)
                    axarr[2].set_title('Difference matrix')
                    plt.savefig(path)

                    if visualize:
                        plt.show()
                    plt.close()

                    if not os.path.exists(os.path.join(self.params["output_data_path"], 'Validate_Statistics.csv')):
                        with open(os.path.join(self.params["output_data_path"], 'Validate_Statistics.csv'),
                                  mode='w') as file:
                            writer = csv.writer(file, delimiter=',', quotechar='"',
                                                quoting=csv.QUOTE_MINIMAL)
                            writer.writerow(['Epoch', 'Name', 'NrPoints', 'NrPatches', 'NrRelations',
                                             'NrWrongRelationsAfterNetwork', 'AccuracyAfterNetwork',
                                             'MinLoss', 'AvgLoss', 'MaxLoss', 'TimeElapsed'])

                    with open(os.path.join(self.params["output_data_path"], 'Validate_Statistics.csv'), mode='a') as file:
                        writer = csv.writer(file, delimiter=',', quotechar='"',
                                                     quoting=csv.QUOTE_MINIMAL)
                        hours, rem = divmod(time.time() - self.start_time, 3600)
                        minutes, seconds = divmod(rem, 60)
                        writer.writerow([str(self.epoch),
                                         file_name,
                                         str(len(point_cloud[0][0])),
                                         str(reference_matrix.shape[0]),
                                         str(relation_matrix.size),
                                         str(np.count_nonzero(difference_matrix)),
                                         str(accuracy),
                                         str(minL),
                                         str(avg),
                                         str(maxL),
                                         str("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
                                         ])

            error_list_y_mean = [np.mean(np.array(i)) for i in error_list_y]
            if save_to_disc:
                if not os.path.exists(os.path.join(self.params["output_data_path"], 'Validate_Epoch_Statistics.csv')):
                    with open(os.path.join(self.params["output_data_path"], 'Validate_Epoch_Statistics.csv'),
                              mode='w') as file:
                        writer = csv.writer(file, delimiter=',', quotechar='"',
                                            quoting=csv.QUOTE_MINIMAL)
                        writer.writerow(['Epoch', 'AvgAccuracyAfterNetwork', 'TimeElapsed'])

                relation_accept_threshold_idx = \
                    np.where(np.array(error_list_x) == self.params['Network']['relation_accept_threshold'])[0][0]
                with open(os.path.join(self.params["output_data_path"], 'Validate_Epoch_Statistics.csv'), mode='a') as file:
                    writer = csv.writer(file, delimiter=',', quotechar='"',
                                        quoting=csv.QUOTE_MINIMAL)
                    hours, rem = divmod(time.time() - self.start_time, 3600)
                    minutes, seconds = divmod(rem, 60)
                    writer.writerow([str(self.epoch),
                                     str(1-error_list_y_mean[relation_accept_threshold_idx]),
                                     str("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
                                     ])

            plt.stem(error_list_x, error_list_y_mean)
            if save_to_disc:
                plt.savefig(os.path.join(self.params['output_data_path'], 'threshold_epoch_' + str(self.epoch)))
            if visualize:
                plt.show()
            plt.close()

        'Save model after every validation step'
        # Note: Prefix default filename with step number Eg: epoch_5_trained_model.pth
        torch.save(self.model.state_dict(), os.path.join(self.params['network_output_path'],
                                                        'epoch_' + str(self.epoch) + '_' + self.params[
                                                                                    'trained_model_name']))

        'set model to train mode'
        self.model.train()