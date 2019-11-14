from structureHelpers import *
from Network.preprocessing import *
from utils.mode import Mode
from training import get_probability_map
import torch

class Prediction:
    def __init__(self, cfg_path, cuda_device_id=0):
        'Provisions'
        self.device = None

        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.setup_cuda(cuda_device_id)


    def setup_cuda(self, cuda_device_id=0):
        '''Setup the CUDA device'''
        torch.backends.cudnn.fastest = True
        torch.cuda.set_device(cuda_device_id)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def setup_model(self, model, convModel, model_file_name=None):
        '''
        Setup the model by defining the model, load the model from the pth file saved during training.
        '''
        model_file_name = model_file_name or self.params['trained_model_name']

        self.model = model(convModel, 1, 2, self.params["Network"]["nr_voxels_per_dim"], self.device).to(self.device)

        self.model.load_state_dict(torch.load(os.path.join(self.params['network_output_path'], model_file_name)))

    def predict(self, predict_data_loader):
        """
        Compute relation probability for all combinations of patches. Save them to disk, so postprocessing can be applied.
        """

        # Read params to check if any params have been changed by user
        self.params = read_config(self.cfg_path)
        self.model.eval()

        batch_size = self.params["Network"]["Patch_database_batch_size"]

        if not os.path.exists(os.path.join(self.params['output_data_path'], 'Predictions')):
            self.params["predictions_path"] = os.path.join(self.params['output_data_path'], 'Predictions')
            write_config(self.params, self.cfg_path, sort_keys=True)
            os.makedirs(self.params["predictions_path"])

        with torch.no_grad():
            for i, (point_cloud, file_name) in enumerate(predict_data_loader):
                file_name = file_name[0]

                'extract patches and relations from given point cloud'
                pc_preprocessor = Point_Cloud_Processing(Mode.PREDICT)

                patch_loader, _, patches, voxelized_patches, normalization_factor = \
                    pc_preprocessor.apply_preprocessing(point_cloud, self.params["Network"]["nr_voxels_per_dim"],
                                                        batch_size, build_ref_matrix=False)

                relation_matrix = (torch.empty(voxelized_patches.shape[0], voxelized_patches.shape[0], 2).float()).to(self.device)

                'loop through all patch combinations and apply network'
                for patch_idx in range(patch_loader.dataset.patch_list.shape[0]):
                    patch_loader.dataset.patch2_idx = patch_idx

                    intermediate = 0
                    for j, (patch1, patch2, shift) in enumerate(patch_loader):
                        # print('index: ' +str(index) + 'patch1 batch size: ' + str(patch1.shape[0]))

                        patch1 = patch1.to(self.device).float()
                        patch2 = patch2.to(self.device).float()
                        shift = shift.to(self.device).float()

                        compare_patches = self.model(patch1, patch2, shift)

                        relation_matrix[patch_idx, intermediate:intermediate+patch2.shape[0]] = compare_patches
                        intermediate += patch2.shape[0]


                relation_matrix = get_probability_map(relation_matrix)
                'get rid of NaN values that result from softmax. Attention: Do not use on backpropagated values.'
                relation_matrix[torch.isnan(relation_matrix)] = 0

                relation_matrix = np.array(relation_matrix.tolist())
                patches = patches * normalization_factor

                'save information to disc'
                file_name = file_name[:-len('.npy')]
                relation_matrix_path = os.path.join(self.params['predictions_path'], file_name + '_relation_matrix_network.npy')
                patches_path = os.path.join(self.params['predictions_path'], file_name + '_patches.npy')
                np.save(relation_matrix_path, relation_matrix)
                np.save(patches_path, patches)
