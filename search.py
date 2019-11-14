import numpy as np
from sklearn.neighbors import NearestNeighbors


class Search:
    """
    This class is used to apply neighbourhood search algorithms on point clouds.
    K-nearest neighbours and radius search is implemented.
    """
    def __init__(self, inputCloud):
        """
        Inputs: inputCloud: np.array of row vectors shape (<numberPoints>, 3)
        """
        self.inputCloud = inputCloud

        self.initializer = 0

    def nearestKSearch(self, point, k):
        """
        Search for k-nearest neighbors for the given query point.

        Inputs: point   query point as vector with dimension (1,3) or (3,1)
                k       number of nearest neighbors that shall be included in
                            output set

        Outputs: outputPointSet     k points that are closest to query point.
                        Shape: (len,3) with 0<=len<=k meaning (x,y,z) for each
                        point in the set. Query point is also included at last
                        position.

        Based on method in:
        docs.pointclouds.org/1.9.1/classpcl_1_1search_1_1_search.html
        and
        https://scikit-learn.org/stable/modules/neighbors.html
        """

        'Column to row vector as required by method kneighbors'
        if point.shape == (3, 1) or point.shape == (3,):
            point = np.reshape(point, (1, 3))

        'avoid error due to less points given than k'
        if self.inputCloud.shape[0] < k:
            k = self.inputCloud.shape[0]

        if self.initializer == 0:
            'Initialisation of nearest neighbour class'
            self.neigh = NearestNeighbors(n_neighbors=k)

            self.neigh.fit(self.inputCloud)

            self.initializer = 1

        'Calculate k nearest neighbours for given point'
        result = self.neigh.kneighbors(point)

        outputPointSet = result[1][0]
        'Note: May use result[0][0] to get distance for each pair'

        return outputPointSet

    def radiusSearch(self, point, radius, max_nn=float("inf")):
        '''
        Search for all the nearest neighbors of the query point in
        a given radius.

        Inputs: point   query point as vector with dimension (1,3) or (3,1)
                radius  radius of sphere to determine included points
                max_nn  maximum number of included points. Default infinit

        Outputs:    outputPointSet  points that are included in the sphere with
                        given radius around query point. Points on sphere
                        are included. Shape: (len,3) meaning (x,y,z) for each
                        point in the set. Query point is also included at last
                        position.

        Based on method in
        http://docs.pointclouds.org/1.9.1/classpcl_1_1search_1_1_search.html
        and
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html
        '''

        'Column to row vector as required by method radius_neighbors'
        if point.shape == (3, 1) or point.shape == (3,):
            point = np.reshape(point, (1, 3))

        if self.initializer == 0:
            'Initialisation of nearest neighbour class'
            self.neigh = NearestNeighbors(radius=radius)
            self.neigh.fit(self.inputCloud)

            self.initializer = 1

        'Calculate nearest neighbours \
        Uses Minkowski distance with p=2 as default which is equal to \
        Euclidean distance'

        result = self.neigh.radius_neighbors(point)

        outputPointSet = result[1][0]
        'Note: May use result[0][0] to get distance for each point in \
        the sphere'

        return outputPointSet