import csv
import time
import matplotlib.pyplot as plt
from numba import jit

from structureHelpers import *
from Network.preprocessing import *
from Network.network import *
from Network.postprocessing import *
import utils.eval as eval
from utils.probability_map import *


class Evaluation:
    """
    This class is used to execute the evaluation process of the network. It is executed after training is finished.
    """
    def __init__(self, cfg_path, cuda_device_id=0):
        """
        Inputs: cfg_path(string): Path to experiment's config file
                cuda_device_id(int): GPU number that shall be used. Range [0,...,7]
        """
        self.device = None

        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.setup_cuda(cuda_device_id)

    def setup_cuda(self, cuda_device_id):
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

    def evaluation(self, test_loader, save_to_disc=False, visualize=False):
        """
        Test model with a separate test dataset

        Inputs:
            test_loader: dataset with test data
            save_to_disc(bool): True if statistics shall be plotted
        """

        print('Begin evaluation')
        start_time = time.time()
        storage_path = os.path.join(self.params['output_data_path'], 'Test_Results')

        'If no test results folder exists, compute patch relations. Otherwise, only do postprocessing'
        if not os.path.exists(storage_path):
            print('Relation matrices not available. Starting to compute relation matrices.')
            self.params["test_path"] = storage_path
            write_config(self.params, self.cfg_path, sort_keys=True)
            os.makedirs(self.params["test_path"])

            self.compute_patch_relations(test_loader, storage_path=storage_path)

        'do postprocessing'
        if os.path.exists(os.path.join(self.params["test_path"], 'Test_Statistics.csv')):
            os.remove(os.path.join(self.params["test_path"], 'Test_Statistics.csv'))

        print('Relation matrices available. Begin postprocessing.')

        'use relation matrices from network as identifier'
        identifiers = [name[:-(len('_relation_matrix_network.npy'))] for name in \
                           os.listdir(storage_path) if name.endswith('_relation_matrix_network.npy')]

        'iterate over all identifiers to process the point clouds stepwise'
        avg_accuracy = []
        for identifier in identifiers:
            print('Starting point cloud ' + identifier)

            'get data'
            relation_matrix_soft = np.load(os.path.join(storage_path, identifier + '_relation_matrix_network.npy'),
                                           allow_pickle=True)
            reference_matrix = np.load(os.path.join(storage_path, identifier + '_reference_matrix.npy'),
                                           allow_pickle=True)
            patches = np.load(os.path.join(storage_path, identifier + '_patches.npy'),
                                           allow_pickle=True)
            'postprocess data'
            relation_matrix_hard = np.zeros(relation_matrix_soft.shape)
            relation_matrix_hard[relation_matrix_soft >= self.params['Network']['relation_accept_threshold']] = 1
            relation_matrix_hard[relation_matrix_soft < self.params['Network']['relation_accept_threshold']] = 0
            relation_matrix_hard = relation_matrix_hard.astype(np.int8)

            consistent_matrices_names = ['after_network', 'hard_noML_m14', 'soft_noML_m14', 'hard_ML_m14',
                                         'soft_ML_m14', 'hard_ML_oracle', 'soft_ML_oracle']
            consistent_relation_matrices = [relation_matrix_hard]

            'hard values after network, no MatchLift, fix m'
            relation_matrix_out, _ = self.postprocessing(relation_matrix_hard, patches=patches,
                                                                   oracle_m=14, apply_MatchLift=False)
            consistent_relation_matrices.append(relation_matrix_out)

            'soft values after network, no MatchLift, fix m'
            relation_matrix_out, _ = self.postprocessing(relation_matrix_soft, patches=patches,
                                                                   oracle_m=14, apply_MatchLift=False)
            consistent_relation_matrices.append(relation_matrix_out)

            'hard values after network, MatchLift, fix m'
            relation_matrix_out, _ = self.postprocessing(relation_matrix_hard, patches=patches,
                                                                   oracle_m=14, apply_MatchLift=True)
            consistent_relation_matrices.append(relation_matrix_out)

            'soft values after network, MatchLift, fix m'
            relation_matrix_out, clustered_point_cloud = self.postprocessing(relation_matrix_soft, patches=patches,
                                                                   oracle_m=14, apply_MatchLift=True)
            consistent_relation_matrices.append(relation_matrix_out)

            'Oracle Assignments'
            oracle_dict = self.oracle_dict()
            m = oracle_dict[identifier[:identifier.find('_')]]
            print(identifier + ' has m: ' + str(m))

            'hard values after network, MatchLift, oracle m'
            relation_matrix_out, _ = self.postprocessing(relation_matrix_hard, patches=patches,
                                                         oracle_m=m, apply_MatchLift=True)
            consistent_relation_matrices.append(relation_matrix_out)

            'soft values after network, MatchLift, oracle m'
            relation_matrix_out, _ = self.postprocessing(relation_matrix_soft, patches=patches,
                                                                             oracle_m=m, apply_MatchLift=True)
            consistent_relation_matrices.append(relation_matrix_out)

            if save_to_disc:

                self.visualize(clustered_point_cloud=clustered_point_cloud, name=identifier, jupyter=False,
                                         auto_open=False)

                accuracy_methods = []
                for i in range(len(consistent_relation_matrices)):
                    difference_matrix = np.abs(reference_matrix - consistent_relation_matrices[i])
                    title = identifier + '_' + consistent_matrices_names[i]
                    self.plot_matrices(consistent_relation_matrices[i], reference_matrix,
                                       title=title, visualize=visualize)
                    accuracy = 1 - np.count_nonzero(difference_matrix) / reference_matrix.size
                    accuracy_methods.append(accuracy)

                avg_accuracy.append(np.array(accuracy_methods))

                if not os.path.exists(os.path.join(self.params["test_path"], 'Test_Statistics.csv')):
                    with open(os.path.join(self.params["test_path"], 'Test_Statistics.csv'),
                              mode='w') as file:
                        writer = csv.writer(file, delimiter=',', quotechar='"',
                                            quoting=csv.QUOTE_MINIMAL)
                        writer.writerow(['Name', 'NrPatches', 'NrRelations',
                                         'Accuracy_'+consistent_matrices_names[0],
                                         'Accuracy_'+consistent_matrices_names[1],
                                         'Accuracy_'+consistent_matrices_names[2],
                                         'Accuracy_'+consistent_matrices_names[3],
                                         'Accuracy_' + consistent_matrices_names[4],
                                         'Accuracy_' + consistent_matrices_names[5],
                                         'Accuracy_' + consistent_matrices_names[6],
                                         'TimeElapsed'])

                with open(os.path.join(self.params["test_path"], 'Test_Statistics.csv'),
                          mode='a') as file:
                    writer = csv.writer(file, delimiter=',', quotechar='"',
                                        quoting=csv.QUOTE_MINIMAL)
                    hours, rem = divmod(time.time() - start_time, 3600)
                    minutes, seconds = divmod(rem, 60)
                    writer.writerow([identifier,
                                     str(reference_matrix.shape[0]),
                                     str(reference_matrix.size),
                                     str(accuracy_methods[0]),
                                     str(accuracy_methods[1]),
                                     str(accuracy_methods[2]),
                                     str(accuracy_methods[3]),
                                     str(accuracy_methods[4]),
                                     str(accuracy_methods[5]),
                                     str(accuracy_methods[6]),
                                     #str(minL),
                                     #str(avg),
                                     #str(maxL),
                                     str("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
                                     ])

            print('Finished point cloud ' + identifier)

        'Postprocessing finished. Calculate mean over all results'
        avg_accuracy = np.mean(np.stack(avg_accuracy, axis=0), axis=0)

        with open(os.path.join(self.params["test_path"], 'Test_Avg_Statistics.csv'),
                  mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Avg_Accuracy_' + consistent_matrices_names[0],
                             'Avg_Accuracy_' + consistent_matrices_names[1],
                             'Avg_Accuracy_' + consistent_matrices_names[2],
                             'Avg_Accuracy_' + consistent_matrices_names[3],
                             'Avg_Accuracy_' + consistent_matrices_names[4],
                             'Avg_Accuracy_' + consistent_matrices_names[5],
                             'Avg_Accuracy_' + consistent_matrices_names[6],
                            ])

        with open(os.path.join(self.params["test_path"], 'Test_Avg_Statistics.csv'),
                  mode='a') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
            writer.writerow([
                str(avg_accuracy[0]),
                str(avg_accuracy[1]),
                str(avg_accuracy[2]),
                str(avg_accuracy[3]),
                str(avg_accuracy[4]),
                str(avg_accuracy[5]),
                str(avg_accuracy[6]),
                ])

    def compute_patch_relations(self, test_loader, storage_path):
        """
        Run the network for all point clouds in the test set and save results to a folder.
        
        Inputs:
            storage_path(str): Path, where to store for each point cloud:
                - unnormalized patches
                - relation matrix
                - reference matrix
        """

        mode = Mode.TEST

        'Set model to evaluation mode. Refer : https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html'
        self.model.eval()

        'Execute test step.'
        with torch.no_grad():

            for i, (point_cloud, file_name) in enumerate(test_loader):
                file_name = file_name[0]

                'extract patches and relations from given point cloud'
                pc_preprocessor = Point_Cloud_Processing(Mode.TEST)
                build_ref_matrix = True
                patch_loader, reference_matrix, patches, voxelized_patches, normalization_factor = \
                    pc_preprocessor.apply_preprocessing(point_cloud, self.params["Network"]["nr_voxels_per_dim"],
                                                        self.params["Network"]["Patch_database_batch_size"],
                                                        build_ref_matrix=build_ref_matrix)

                relation_matrix = (torch.empty(voxelized_patches.shape[0], voxelized_patches.shape[0], 2).float()).to(
                    self.device)
                reference_matrix = (torch.tensor(reference_matrix)).to(self.device)

                'loop through all patch combinations and apply network'
                for patch_idx in range(patch_loader.dataset.patch_list.shape[0]):
                    patch_loader.dataset.patch2_idx = patch_idx
                    intermediate = 0
                    for j, (patch1, patch2, shift, _) in enumerate(patch_loader):
                        patch1 = patch1.to(self.device).float()
                        patch2 = patch2.to(self.device).float()
                        shift = shift.to(self.device).float()

                        compare_patches = self.model(patch1, patch2, shift)

                        relation_matrix[patch_idx, intermediate:intermediate + patch2.shape[0]] = compare_patches
                        intermediate += patch2.shape[0]

                relation_matrix_soft = self.compute_network_output(relation_matrix, mode='soft')
                reference_matrix = np.array(reference_matrix.tolist())
                patches = patches * normalization_factor

                'save information to disc'
                file_name = file_name[:-len('.npy')]
                relation_matrix_path = os.path.join(storage_path, file_name +'_relation_matrix_network.npy')
                reference_matrix_path = os.path.join(storage_path, file_name + '_reference_matrix.npy')
                patches_path = os.path.join(storage_path, file_name + '_patches.npy')
                np.save(relation_matrix_path, relation_matrix_soft)
                np.save(reference_matrix_path, reference_matrix)
                np.save(patches_path, patches)

    def matrix_from_clusters(self, cluster_list):
        """
        This method gets a list of clusters and returns the corresponding relation matrix.

        Inputs:
            cluster_list(list of lists): list of lists with indices of patches

        Outputs:
            relation_matrix(np.array(NxN)): Binary matrix of patch relations.
                    Size is NxN, where N is maximum index from cluster_list.
        """

        'get size of matrix by counting elements in cluster list.'
        N = 0
        for i in range(len(cluster_list)):
            N += len(cluster_list[i])

        relation_matrix = np.zeros((N, N))

        'iterate over all cluster lists'
        for cluster in range(len(cluster_list)):
            'iterate over all combinations of patch indices inside a cluster'
            for patch_i_idx in cluster_list[cluster]:
                for patch_j_idx in cluster_list[cluster]:
                    'set element (i,j) to 1'
                    relation_matrix[patch_i_idx, patch_j_idx] = 1
                    'set element (j,i) to 1'
                    relation_matrix[patch_j_idx, patch_i_idx] = 1

        return relation_matrix

    def plot_matrices(self, estimated_matrix, ground_truth_matrix, title, visualize=False):
        """
        Save plots of matrices to disk and show them.

        Inputs:
            estimated_matrix(np.array): The estimated matrix
            ground_truth_matrix(np.array): The ground truth matrix
            title(str): The title of the plot
            visualize(bool): Whether to show the plot
        """

        'conversions before plotting to have only 1 and 0 as value. Otherwise' \
        'uint8 difference is circular'
        estimated_matrix = np.asarray(estimated_matrix, dtype="uint8")
        ground_truth_matrix = np.asarray(ground_truth_matrix, dtype="uint8")
        difference_matrix = np.asarray(np.abs(ground_truth_matrix-estimated_matrix), dtype="uint8")

        plt.figure()
        f, axarr = plt.subplots(1, 3)

        plt.title(title)

        axarr[0].imshow(estimated_matrix)
        axarr[0].set_title('Estimated matrix')
        axarr[1].imshow(ground_truth_matrix)
        axarr[1].set_title('Reference matrix')
        axarr[2].imshow(difference_matrix)
        axarr[2].set_title('Difference matrix')

        path = os.path.join(self.params['test_path'], title)
        plt.savefig(path, dpi=600)

        if visualize:
            plt.show()
        plt.close()

    def postprocessing(self, relation_matrix_in, patches, oracle_m=None, apply_MatchLift=True):
        """
        Apply convex optimization and rounding.

        Inputs:
            relation_matrix_in(np.array): Noisy input relation matrix
            patches(np.array): list of points
            oracle_m(int): Number of faces can be given by oracle or as fix value. If None, it will be estimated.
            apply_MatchLift(bool): Whether to apply convex optimization of not

        Outputs:
            relation_matrix_out(np.array): Consistent binary matrix with patch indices that belong together
            point_cloud_recovered: List of lists: Points in one list element belong to the same cluster
        """

        post_processor = Postprocessor(self.cfg_path, mode='test', oracle_m=oracle_m)
        post_processor.relation_matrix = relation_matrix_in
        post_processor.patches = patches

        if apply_MatchLift:
            relation_matrix_cvx = post_processor.apply_MatchLift()
        else:
            relation_matrix_cvx = relation_matrix_in #noisy relation matrix

        clusters = post_processor.enforce_consistent_clusters(relation_matrix_cvx)

        relation_matrix_out = self.matrix_from_clusters(cluster_list=clusters)

        point_cloud_recovered = post_processor.recover_point_cloud(clusters=clusters)

        return copy.deepcopy(relation_matrix_out), copy.deepcopy(point_cloud_recovered)

    def oracle_dict(self):
        """
        This is a dictionary that contains the number of faces for different point clouds.
        Required for oracle assignment.
        """

        dict_nr_faces = {
            "Tetrahedron": 4,
            "Octahedron": 8,
            "Dodecahedron": 12,
            "Truncated Tetrahedron": 8,
            "Cuboctahedron": 14,
            "Truncated Octahedron": 14,
            "Triangular Prism": 5,
            "Pentagonal Prism": 7,
            "Hexagonal Prism": 8,
            "Heptagonal Prism": 9,
            "Square Pyramid": 5,
            "Triangular Cupola": 8,
            "Square Cupola": 10,
            "Cube Cutoff": 6,
            "Cutoff Parallelogram Offset": 8,
            "Cube": 6,
            "Octagonal Prism": 10,
            "Pentagonal Pyramid": 6,
        }

        return dict_nr_faces

    def compute_network_output(self, relation_matrix, mode='real'):
        """
        This function computes the relation matrix from the network output.
            - Softmax is applied
            - Then hard or soft values are returned

        Inputs:
            relation_matrix(torch.tensor([N,N,2)): Relation matrix from N patches
            mode(str): 'hard' or 'soft'

        Outputs:
            relation_matrix(np.array([N,N])): relation matrix with hard(binary) or soft(real) values
        """

        relation_matrix = get_probability_map(relation_matrix)
        'get rid of NaN values that result from softmax. Attention: Do not use on backpropagated values.'
        relation_matrix[torch.isnan(relation_matrix)] = 0

        relation_matrix = np.array(relation_matrix.tolist())

        if mode == 'hard':
            relation_matrix[relation_matrix >= self.params['Network']['relation_accept_threshold']] = 1
            relation_matrix[relation_matrix < self.params['Network']['relation_accept_threshold']] = 0
            relation_matrix = relation_matrix.astype(np.int8)

        return relation_matrix

    def visualize(self, clustered_point_cloud, name, jupyter=False, auto_open=False):
        """
        Visualizes a clustered point cloud using plotly.

        Inputs: clustered_point_cloud(list of lists): Clusters containing points
                name(str): Name of point cloud
                jupyter(bool): True if plot shall be shown in Jupyter notebook. False if plot shall be shown in browser.
                auto_open(bool): to define whether .html file shall be opened or just stored

        """

        if jupyter == True and auto_open == True:
            print('Function visualize: Parameter auto_open has no effect in jupyter notebooks.')

        data = []
        for facet_id in range(len(clustered_point_cloud)):
            x = y = z = []
            for idx in range(len(clustered_point_cloud[facet_id])):
                x = np.append(x, clustered_point_cloud[facet_id][idx][0])
                y = np.append(y, clustered_point_cloud[facet_id][idx][1])
                z = np.append(z, clustered_point_cloud[facet_id][idx][2])

            RGB_colour = mapRGB(facet_id)

            data.append(go.Scatter3d(
                name='facet #' + str(facet_id),
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    symbol='circle',
                    color=RGB_colour,
                    size=2,
                    line=dict(
                        width=0,
                        color=RGB_colour)
                )
            ))

        layout = go.Layout(
            # title='Point Cloud Scatter Plot',
            scene=dict(aspectmode='data',
                       xaxis=dict(showticklabels=False,
                                  gridcolor="black",
                                  title=''),
                       yaxis=dict(showticklabels=False,
                                  gridcolor="black",
                                  title=''),
                       zaxis=dict(showticklabels=False,
                                  gridcolor="black",
                                  title='')),
            # width=1000,
            # margin=dict(
            #    r=20, l=10,
            #    b=10, t=10))
        )

        fig = go.Figure(data=data, layout=layout)

        path = os.path.join(self.params['test_path'], name + '_clustered_point_cloud.html')

        if jupyter == True:
            py.iplot(fig, filename=path)
            'store plot'
            py.plot(fig, filename=path, auto_open=False)
        else:
            py.plot(fig, filename=path, auto_open=auto_open)