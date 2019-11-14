import sys
sys.path.append('../')

import cvxpy as cp
import numpy as np
from numpy import linalg as LA
import plotly.offline as py
import plotly.graph_objs as go
from utils.mapRGB import mapRGB
from numba import jit
import copy

from structureHelpers import *

class Point:
    """
    This class is used to store the properties of each point.
    """
    def __init__(self, coordinates=0, segment="NaN"):
        self.coordinates = coordinates

        'Note: each point has a segment that it belongs to'
        self.segment = segment

    def setCoordinates(self, coordinates):
        self.coordinates = coordinates

    def setSegment(self, segment):
        self.segment = segment

    def getCoordinates(self):
        return self.coordinates

    def getSegment(self):
        return self.segment

class Postprocessor:
    def __init__(self, cfg_path, mode='prediction', oracle_m=None):
        """
        Inputs:
            cfg_path(str): Path to configuration file
            mode(str): 'test' or 'prediction'
            oracle_m(int): Number of faces m can be assigned by oracle or as fix value. If None, it will be estimated.
        """

        self.m_estimated = oracle_m

        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.mode = mode
        self.error = 0

        self.point_cloud = None
        self.patches = None
        self.relation_matrix = None
        self.relation_matrix_after_enforce_consistency = None
        self.name = '0'

    def load_data(self, name):
        """
        Load estimated relation matrix and unnormalized patches from predictions path using their identifier.

        Inputs:
            name(str): point cloud name
        """

        self.name = name

        relation_matrix_path = os.path.join(self.params['predictions_path'], self.name + '_relation_matrix_network.npy')
        patches_path = os.path.join(self.params['predictions_path'], self.name + '_patches.npy')

        self.relation_matrix = np.load(relation_matrix_path, allow_pickle=True)
        self.patches = np.load(patches_path, allow_pickle=True)

        self.error = 0
        if self.patches.shape[0] != self.relation_matrix.shape[0]:
            self.error = 1
            'Note: Reason for error could not be identified. Error appears occasionally.'
            print('postprocessing.py/load_data: Attention: Number of patches differs with relation matrix. Evaluation cannot be done.')

    def recover_point_cloud(self, clusters):
        """
        Find consistent clusters and assign points in the patches according to those.

        Inputs:
            clusters(list of lists): List that contains clusters with patch indices
        """

        if self.error == 1:
            return

        print('Starting to recover point cloud')
        self.point_cloud = [[] for i in range(len(clusters))]

        'i: iterate over whole cluster list'
        for i in range(len(clusters)):
            'j: iterate over entries of each cluster list element'
            for j in range(len(clusters[i])):
                'k: iterate over all point coordinates of the patches that correspond to cluster list entry'
                points_in_patch = self.patches[clusters[i][j], 0].shape[0]
                for k in range(points_in_patch):
                    'assign coordinates and segment'
                    self.point_cloud[i].append(self.patches[clusters[i][j], 0][k])

        return copy.deepcopy(self.point_cloud)

    def enforce_consistent_clusters(self, relation_matrix):
        """
        Find consistent clusters in a non-greedy way.
        The columns of the relation matrix after MatchLift with maximum correlation are merged step-wise until there
        are no more two columns with an inner product greater than a certain threshold.

        Input:
            relation matrix: Relation matrix after MatchLift
        """

        if self.error == 1:
            return

        if self.m_estimated is None:    # No MatchLift performed
            self.m_estimated = MatchLift(relation_matrix, oracle_m=None).estimate_m()
        print('Enforce Consistency: m_estimated= ' + str(self.m_estimated))

        dim = relation_matrix.shape[0]
        n = dim/self.m_estimated

        'prepare lists with ascending numbers. One for each patch. Those are later merged.'
        clusters = []
        for i in range(dim):
            clusters.append([i])

        finished = 0

        'make a copy of the relation matrix to avoid changing it'
        B = copy.deepcopy(relation_matrix)
        while finished == 0:

            'A is the correlation matrix of B'
            A = np.matmul(B.T, B)
            'to avoid that same columns are merged'
            np.fill_diagonal(A, 0)

            idx_max_inner_prod = np.unravel_index(A.argmax(), A.shape)

            'if the two columns with maximum correlation have an inner product higher than a threshold, ' \
            'merge both associated cluster lists. If no two columns can be found, finish.'
            if A[idx_max_inner_prod] >= n/2:
                clusters[idx_max_inner_prod[0]].extend(clusters[idx_max_inner_prod[1]])
                del clusters[idx_max_inner_prod[1]]

                'Sum both columns and renormalize'
                B = np.zeros((B.shape[0], B.shape[1]-1))
                for i in range(B.shape[1]):
                    B[:, i] = np.sum(relation_matrix[clusters[i]], axis=0) / len(clusters[i])
            else:
                finished = 1
                print('Consistent mapping found')
                self.relation_matrix_after_enforce_consistency = relation_matrix
                return clusters

    def enforce_consistent_clusters_v2(self, relation_matrix):
        """
        Version of enforce_consistent_clusters with more functionality. On top of the basic functionality,
        a copy of the relation matrix is modified during the clustering process. When patches are assigned to the same
        cluster, the cells corresponding to those are assigned to 1.

        Example: Patch 1 and Patch 3 are assigned together. Those are found in the relation matrix in column 1 and 3.
        Then cell (1,3) and (3,1) can be set to 1.

        The process is repeated from beginning if a certain number of cells has been corrected to 1, i.e. a better
        matrix is available.
        """

        if self.error == 1:
            return

        if self.m_estimated is None: # No MatchLift performed
            self.m_estimated = MatchLift(relation_matrix, oracle_m=None).estimate_m()

        dim = relation_matrix.shape[0]
        n = dim/self.m_estimated

        'prepare lists with ascending numbers. One for each patch. Those are later merged.'
        clusters = []
        for i in range(dim):
            clusters.append([i])
        clusters_cpy = copy.deepcopy(clusters)

        finished = 0
        'make a copy of the relation matrix to avoid changing it'
        B = copy.deepcopy(relation_matrix)
        repeat_counter = 0
        repeat_threshold = 10

        while finished == 0:
            'A is the correlation matrix of B'
            A = np.matmul(B.T, B)
            'to avoid that same columns are merged'
            np.fill_diagonal(A, 0)

            idx_max_inner_prod = np.unravel_index(A.argmax(), A.shape)

            'if the two columns with maximum correlation have an inner product higher than a threshold, ' \
            'merge both associated cluster lists. If no two columns can be found, finish.'
            if A[idx_max_inner_prod] >= n/2:
                clusters[idx_max_inner_prod[0]].extend(clusters[idx_max_inner_prod[1]])
                del clusters[idx_max_inner_prod[1]]
                'patches now belong together -> improve mapping by setting all combinations of' \
                'elements that belong to patches of the currently fixed same cluster to 1 in relation matrix'
                for i in range(len(clusters[idx_max_inner_prod[0]])):
                    for j in range(len(clusters[idx_max_inner_prod[0]])):
                        idx1 = clusters[idx_max_inner_prod[0]][i]
                        idx2 = clusters[idx_max_inner_prod[0]][j]
                        if relation_matrix[idx1, idx2] != 1 or relation_matrix[idx2, idx1] != 1:
                            relation_matrix[idx1, idx2] = 1
                            relation_matrix[idx2, idx1] = 1

                            repeat_counter += 1
                            if repeat_counter == repeat_threshold:
                                'repeat from new in hope of finding better relations'
                                B = copy.deepcopy(relation_matrix)
                                clusters = copy.deepcopy(clusters_cpy)
                                break
                    if repeat_counter == repeat_threshold:
                        break
                if repeat_counter == repeat_threshold:
                    repeat_counter = 0
                    continue

                'Sum both columns and renormalize'
                B = np.zeros((B.shape[0], B.shape[1]-1))
                for i in range(B.shape[1]):
                    B[:, i] = np.sum(relation_matrix[clusters[i]], axis=0) / len(clusters[i])
            else:
                finished = 1
                print('Consistent mapping found')
                self.relation_matrix_after_enforce_consistency = relation_matrix
                return clusters

    def apply_MatchLift(self):
        """
        Applies match lift algorithm

        """
        if self.error == 1:
            return

        ml = MatchLift(self.relation_matrix, oracle_m=self.m_estimated)
        relation_matrix_after_matchLift, self.m_estimated = ml.matchLift()

        return copy.deepcopy(relation_matrix_after_matchLift)

    def visualize(self, jupyter=False, auto_open=False):
        """
        Visualizes a clustered point cloud using plotly.

        Inputs: jupyter(bool): True if plot shall be shown in Jupyter notebook. False if plot shall be shown in browser.
                auto_open(bool): to define whether .html file shall be opened or just stored

        """

        if self.error == 1:
            return

        if jupyter == True and auto_open == True:
            print('Function visualize: Parameter auto_open has no effect in jupyter notebooks.')

        data = []
        for facet_id in range(len(self.point_cloud)):
            x = y = z = []
            for idx in range(len(self.point_cloud[facet_id])):
                x = np.append(x, self.point_cloud[facet_id][idx][0])
                y = np.append(y, self.point_cloud[facet_id][idx][1])
                z = np.append(z, self.point_cloud[facet_id][idx][2])

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

        if self.mode == 'test':
            path = os.path.join(self.params['test_path'], self.name + '_clustered_point_cloud.html')
        else:
            path = os.path.join(self.params['predictions_path'], self.name + '_clustered_point_cloud.html')

        if jupyter == True:
            py.iplot(fig, filename=path)
            'store plot'
            py.plot(fig, filename=path, auto_open=False)
        else:
            py.plot(fig, filename=path, auto_open=auto_open)

class MatchLift:
    """
    This class is based on the descriptions given in
        [1] Near-Optimal Joint Object Matching via Convex Relaxation by Chen et al. See there for more details.
    """
    def __init__(self, X_in, oracle_m=None):
        """
        Inputs:
            X_in(np.array): Matrix with noisy relations
            oracle_m(int): Number of faces may be given via oracle. This input can also be used for fix m.
        """
        self.X_in = X_in

        self.vertex_degrees = []
        'each m_i has size 1. There is m_1 to m_n.'
        self.m_i_list = np.ones((1, self.X_in.shape[0]))

        self.m_estimated = oracle_m

    def trim_X_in(self):
        """
        Necessary for estimating m. Block-sparse matrix X_in must be trimmed to remove
        undesired bias effect caused by over-represented rows/columns.

        """

        X_in_trimmed = self.X_in
        """
        Note: Commented the whole trimming, because each S_i is a set with 1 element, as acc.
        page 4 of [1, X_ij is element{0,1}^{S_i x S_j} and X_ij is only 1 element.
        
        'find smallest vertex degree in graph'
        self.calculate_vertex_degrees()
        d_min = min(self.vertex_degrees)

        'A vertex is overrepresented if its vertex degree exceeds 2*d_min.' \
        'Randomly sample 2*d_min edges in this case and set others to 0'
        for i in range(len(self.vertex_degrees)):
            if self.vertex_degrees[i] > 2*d_min:
                probability_set_to_0 = (self.vertex_degrees[i] - 2*d_min)/self.vertex_degrees[i]

                for j in range(self.X_in_trimmed.shape[0]):
                        if self.X_in_trimmed[i, j] == 1: # the ith row
                            if np.random.binomial(1, probability_set_to_0):
                                self.X_in_trimmed[i, j] = 0

                        if i == j:
                            #to avoid that probability for main diagonal element to be set to 0 is twice as high
                            continue

                        if self.X_in_trimmed[j, i] == 1: # the ith column
                            if np.random.binomial(1, probability_set_to_0):
                                self.X_in_trimmed[j, i] = 0
        """

        return X_in_trimmed

    def calculate_vertex_degrees(self):
        """
        Each patch is considered a view. There are n patches that make n sets S_1 ... S_n.
        Each patch is thus a vertex and the vertex degree is given by the number of ones in the row and column
        corresponding to the given patch.
        """

        for i in range(self.X_in.shape[0]):
            'Note: -1 to avoid that X_ii is counted twice'
            self.vertex_degrees.append(np.count_nonzero(self.X_in[:, i]) + np.count_nonzero(self.X_in[i, :]) - 1)

    def estimate_m(self, X_in_trimmed):
        """
        Estimate size of universe m according to Algorithm 1 in [1].
        Here m is the number of clusters.
        """

        eigenvalues, eigenvectors_normalized = LA.eig(X_in_trimmed)

        M = int(np.max([2, np.max(self.m_i_list)]))
        N = X_in_trimmed.shape[0]
        'lambda_i in descending order'
        lambda_i_list = np.sort(eigenvalues)[::-1]

        m_estimator = np.array([])
        for i in range(M-1, N-2):
            'start at M-1, because [1] starts indexing with 1. Run until N-2, because N is excluded in [1]' \
            'and indexing in [1] starts at 1.'
            m_estimator = np.append(m_estimator, np.abs(lambda_i_list[i]-lambda_i_list[i+1]))
        'Return estimated m. Add M, because indexing in m_estimator starts at 0, ' \
        'but argmax in [1] starts at M.'
        m_estimated = np.argmax(m_estimator) + M

        return m_estimated

    def matchLift(self):
        """
        Apply Match Lift algorithm
        """

        X_in_trimmed = self.trim_X_in()

        if self.m_estimated is None:
            self.m_estimated = self.estimate_m(X_in_trimmed)
        else:
            print('postprocessing.py: Using oracle assignment or fix value for number of faces m')
        print('estimated m: ' +str(self.m_estimated))

        dim = X_in_trimmed.shape[0]
        n = dim
        #cardinality_edges = np.count_nonzero(np.around(X_in_trimmed, decimals=1))    #rounding necessary, if soft matrix is used
        #lambda_regularizer = np.sqrt(cardinality_edges)/(2*n)
        #print('Lambda Regularizer = ' + str(lambda_regularizer))
        lambda_regularizer = 0.5
        ones_matrix = np.ones(X_in_trimmed.shape)
        'prototype for semidefinite constraint'
        prototype_matrix = np.zeros((dim+1, dim+1))
        prototype_matrix[0, :] = np.ones(dim+1)
        prototype_matrix[:, 0] = np.ones(dim + 1)
        prototype_matrix[0, 0] = self.m_estimated

        X = cp.Variable(shape=(dim, dim), symmetric=True)
        prototype_matrix = cp.Variable(shape=(dim+1, dim+1))

        'Convex optimization problem'
        objective = cp.Minimize(-cp.trace(cp.matmul(X.T, X_in_trimmed)) + lambda_regularizer*cp.trace(cp.matmul(ones_matrix, X)))

        constraints = [prototype_matrix[0, 1:] == 1,
                       prototype_matrix[1:, 0] == 1,
                       prototype_matrix[0, 0] == self.m_estimated,
                       prototype_matrix[1:, 1:] == X,
                       prototype_matrix >> 0,
                       X >= 0,
                       X <= 1,
                       cp.diag(X) == [1] * dim,
                       ] # >>0 means positive semidefinite
        prob = cp.Problem(objective, constraints)

        result = prob.solve()
        #print(result)
        #print(X.value)
        print('Found solution via MatchLift is ' + prob.status)

        relation_matrix_after_matchLift = X.value

        return relation_matrix_after_matchLift, self.m_estimated