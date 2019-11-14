import sys
sys.path.append('../')

import numpy as np
from scipy.spatial import ConvexHull
from constraint import *
import random
import copy
import plotly.offline as py
import plotly.graph_objs as go
import math
import os
from utils.mapRGB import mapRGB
from search import Search
from utils.mode import Mode

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


class polyhedronGenerator:
    def __init__(self, input_points, nr_points_polyhedron, path=None, name="Polyhedron", mode='clean', variance=0.1,
                 scaling=1, roll_pitch_yaw=[0, 0, 0], k_round_edges=1, squash_xy = [1, 1]):
        """
        Inputs:
            input_points(list): List of vertices of polyhedron
            nr_points_polyhedron(int): Nr of polyhedron points that shall be sampled
            path (str): Path where to store polyhedra. If none, store to default temporary location
            name (str): Name of polyhedron. Will be assigned during saving
            mode: clean or noisy
            variance (float): variance level. Only applicable when mode == noisy
            scaling (float): each point will be scaled with this factor
            roll_pitch_yaw (list of floats): rotation angles in roll, pitch and yaw direction in degree
            k_round_edges (int): Parameter k for k-nearest neighbours used to round the edges of a point cloud
            squash_xy (list of floats): factor to squash x and y axis
        """
        'input are vertices'
        self.input_points = input_points

        'user-wished number of points that are sampled on the convex hull of the polyhedron'
        self.nr_points_polyhedron = nr_points_polyhedron

        self.name = name

        if mode == 'clean' or mode == 'noisy':
            self.mode = mode
        else:
            raise ValueError('mode must be either clean or noisy')

        self.path = path

        'Variance of Gaussian distribution that is applied to make data noisy'
        self.variance = variance

        'scale all points with same factor. Is applied before possible addition of noise.'
        self.scaling = scaling

        self.roll_pitch_yaw = roll_pitch_yaw

        self.k_round_edges = k_round_edges

        self.squash_xy = squash_xy

        self.hull = 0
        self.points_polyhedron = 0


    def save(self):
        """
        Store sampled polyhedron as Point objects with shape (N,)
        """

        'Combine coordinates and segment number'
        point_list = []
        for segment_idx in range(self.points_polyhedron.shape[0]):
            for point_idx in range(self.points_polyhedron[segment_idx].shape[0]):
                point_list.append(Point(coordinates=self.points_polyhedron[segment_idx][point_idx],
                                        segment=segment_idx))

        point_list = np.array(point_list)

        'store to file'
        if self.path == None:
            'store to default temporary location. Overwrite files'

            path = os.path.join('data/temp_data/', self.name + '.npy')
            print('File: polyhedronGenerator. Stored to default temporary path.')
        else:
            counter = 0
            path = os.path.join(self.path, self.name + '_' + str(counter) + '.npy')
            while os.path.exists(path):
                counter = counter + 1
                path = os.path.join(self.path, self.name + '_' + str(counter) + '.npy')

        np.save(path, point_list)

    def load(self, path):
        """
        Load previously saved polyhedron under path

        Inputs:
            path(str)
        """

        point_list = np.load(path, allow_pickle=True)

        self.points_polyhedron = []

        j = -1
        for i in range(point_list.shape[0]):
            if j == point_list[i].getSegment():
                self.points_polyhedron[j].append(point_list[i].getCoordinates())
            else:
                j += 1
                self.points_polyhedron.append([point_list[i].getCoordinates()])

        self.points_polyhedron = np.array(self.points_polyhedron)

    def rotate_roll(self, theta_rad):
        'Rotate with angle theta given in radians around x-axis'

        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(theta_rad), -np.sin(theta_rad)],
                                    [0, np.sin(theta_rad), np.cos(theta_rad)]])

        return rotation_matrix

    def rotate_pitch(self, theta_rad):
        'Rotate with angle theta given in radians around y-axis'

        rotation_matrix = np.array([[np.cos(theta_rad), 0, np.sin(theta_rad)],
                                    [0, 1, 0],
                                    [-np.sin(theta_rad), 0, np.cos(theta_rad)]])

        return rotation_matrix

    def rotate_yaw(self, theta_rad):
        'Rotate with angle theta given in radians around z-axis'

        rotation_matrix = np.array([[np.cos(theta_rad), -np.sin(theta_rad), 0],
                                    [np.sin(theta_rad), np.cos(theta_rad), 0],
                                    [0, 0, 1]])

        return rotation_matrix

    def rotate3D(self, yaw_rad, pitch_rad, roll_rad):
        """
        Return 3D rotation matrix in accordance with angles given in radians.

        Referring to
            https://en.wikipedia.org/wiki/Rotation_matrix
        a yaw, pitch, roll rotation can be obtained by the matrix product of
        the three individual rotation matrices

        Inputs:
            yaw_rad(float): yaw angle
            pitch_rad(float): pitch angle
            roll_rad(float): roll angle
        """

        rotation_matrix = np.matmul(np.matmul(self.rotate_yaw(yaw_rad), self.rotate_pitch(pitch_rad)), \
                          self.rotate_roll(roll_rad))

        return rotation_matrix

    def visualize(self, jupyter=False, auto_open=False):
        """
        Visualizes the points given in self.points_polyhedron using plotly.
        Shape of self.points_polyhedron must be (<segments>, <points in segment>, 3)

        Inputs:
                jupyter(bool): True if plot shall be shown in Jupyter notebook. False if plot shall be shown in browser.
                auto_open(bool): to define whether .html file shall be opened or just stored
                
        """
        
        if jupyter == True and auto_open == True:
            print('Function visualize: Parameter auto_open has no effect in jupyter notebooks.')
        
        data = []
        for facet_id in range(self.points_polyhedron.shape[0]):
            x = y = z = []
            for idx in range(self.points_polyhedron[facet_id].shape[0]):
                x = np.append(x, self.points_polyhedron[facet_id][idx][0])
                y = np.append(y, self.points_polyhedron[facet_id][idx][1])
                z = np.append(z, self.points_polyhedron[facet_id][idx][2])

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

        if self.path == None:
            'store to default temporary location. Overwrite files'

            path = os.path.join('data/temp_data/', self.name + '.html')
            print('File: polyhedronGenerator. Stored to default temporary path.')
        else:
            counter = 0
            path = os.path.join(self.path, self.name + '_' + str(counter) + '.html')
            while os.path.exists(path):
                counter = counter + 1
                path = os.path.join(self.path, self.name + '_' + str(counter) + '.html')

        if jupyter == True:
            py.iplot(fig, filename=path)
            'store plot'
            py.plot(fig, filename=path, auto_open=False)
        else:
            py.plot(fig, filename=path, auto_open=auto_open)


    def calculateConvexHull(self):
        self.hull = ConvexHull(self.input_points)

    def convexHullGenerate(self):

        'calculate convex hull element'
        self.calculateConvexHull()

        'calculate number of samples per facet such that they roughly add up to the user chosen number of samples'
        nr_facet_samples = int(np.floor(self.nr_points_polyhedron / self.hull.nsimplex))
        self.nr_points_polyhedron = nr_facet_samples

        'formulate constraint satisfaction problem'
        problem = Problem()

        alpha1 = np.linspace(0.0, 1.0, num=100) #100 elements per alpha results in ~5000 combinations ' \
                                                #that satisfy the CSP. This is considered sufficiently dense.'
        alpha2 = np.linspace(0.0, 1.0, num=100)

        problem.addVariable("alpha1", alpha1)
        problem.addVariable("alpha2", alpha2)

        problem.addConstraint(lambda a, b: a + b <= 1, ("alpha1", "alpha2"))
        solutions = problem.getSolutions()

        points_polyhedron = [] #This list contains the chosen samples of all facets

        'For each triangle facet, generate random samples that are inside the triangle'
        for simplex in self.hull.simplices:

            v0 = np.array(self.hull.points[simplex[0]])   #any vector of the current simplex fits as starting' \
                                                          #  point. Here simply first element of the' \
                                                          #  current simplex is chosen.'
            v1 = np.array(self.hull.points[simplex[1]])
            v2 = np.array(self.hull.points[simplex[2]])
            points_facet = []
            for alpha_combination in solutions:
                points_facet.append(v0 + alpha_combination["alpha1"] * (-v0 + v1) + \
                              alpha_combination["alpha2"] * (-v0 + v2))

            'also append vertex points'
            points_facet.append(v0)
            points_facet.append(v1)
            points_facet.append(v2)

            'choose portion of user defined number of random samples from all available samples'
            points_facet = random.choices(points_facet, k=nr_facet_samples)

            points_polyhedron.append(points_facet)

        'merge all triangle facets that lie in the same plane (i.e. belong to the same facet)'
        planes = np.array(self.hull.equations)
        merged_list = [] #this list keeps track of which facets have already been merged
        for idx in range(planes.shape[0]):
            same_plane_value_elementwise = planes == planes[idx]
            for i in range(idx, same_plane_value_elementwise.shape[0]):
                if i != idx and not(i in merged_list) and all(x == True for x in same_plane_value_elementwise[i, :]):
                    points_polyhedron[idx] = points_polyhedron[idx] + points_polyhedron[i]
                    merged_list.append(i)

        'remove all triangle facets that have been merged previously.' \
        'Reversed order to avoid indexing problems.'
        for i in reversed(merged_list):
            points_polyhedron.pop(i)

        points_polyhedron = np.array([np.array(i) for i in points_polyhedron])

        'rotation'
        if np.any(self.roll_pitch_yaw):
            roll = np.pi * self.roll_pitch_yaw[0] / 180
            pitch = np.pi * self.roll_pitch_yaw[1] / 180
            yaw = np.pi * self.roll_pitch_yaw[2] / 180
            for i in range(points_polyhedron.shape[0]):
                points_polyhedron[i] = np.transpose(np.matmul(self.rotate3D(yaw, pitch, roll), np.transpose(points_polyhedron[i])))

        'round the edges'
        if self.k_round_edges > 1:
            points_polyhedron = self.round_edges(points_polyhedron, self.k_round_edges)

        'squashing x,y axis'
        if np.any([i != 1 for i in self.squash_xy]):
            points_polyhedron = self.squash_axis(points_polyhedron, self.squash_xy[0], self.squash_xy[1])

        'normalize to [0,1]^3'
        points_polyhedron = self.normalize_point_cloud(points_polyhedron)

        'apply normal distribution to each point if mode is noisy'
        if self.mode == 'noisy':
            points_polyhedron = self.applyNoise(points_polyhedron)

        'scaling'
        points_polyhedron = points_polyhedron * self.scaling

        self.points_polyhedron = points_polyhedron

        return copy.deepcopy(self.points_polyhedron)

    def squash_axis(self, point_cloud, squash_x, squash_y):
        """
        This function squashes the x and y axis of a point cloud point wise.

        Inputs:
            point_cloud (np.array shape (<nrClusters>, <NrPoints>, 3))
            squash_x, squash_y (float): Factor how much to squash
        """

        for i in range(point_cloud.shape[0]):       # for each cluster
            for j in range(point_cloud[i].shape[0]):     # for each point in cluster
                point_cloud[i][j, 0] = point_cloud[i][j, 0] * squash_x
                point_cloud[i][j, 1] = point_cloud[i][j, 1] * squash_y
        return point_cloud

    def round_edges(self, point_cloud, k):
        """
        Rounden the edges of a point cloud by averaging over the local neighbourhood of each point in
        the point cloud.

        Inputs:
            point_cloud (np.array shape (<nrClusters>, <NrPoints>, 3))
            k (int): Parameter for k-nearest neighbours
        """

        'flatten the point cloud as input to KNN'
        flat_point_cloud = []
        for i in range(point_cloud.shape[0]):
            flat_point_cloud.extend(point_cloud[i])

        search_obj = Search(np.array(flat_point_cloud))

        point_cloud_rounded = copy.deepcopy(point_cloud) * 0
        'iterate over all points in the point cloud.'
        for i in range(point_cloud.shape[0]):
            for j in range(point_cloud[i].shape[0]):
                'find all k nearest neighbours to the current point. Returned indices are allocated to points.'
                point_neighbours_indices = search_obj.nearestKSearch(point_cloud[i][j], k)
                point_neighbours = np.array([flat_point_cloud[k] for k in point_neighbours_indices])

                'update point location by average over local neighbourhood'
                point_cloud_rounded[i][j] = np.sum(point_neighbours, axis=0)/k

        return point_cloud_rounded

    def applyNoise(self, pointSet):
        """
        Apply noise on given points

        Input: pointSet: must be shape (<Nr_segments>, <Nr_points_in_segment>, 3)

        Output: pointSet: noisy pointSet
        """
        for idx in range(pointSet.shape[0]):
            'Gaussian distribution with 0 mean and user-defined variance'
            noise_matrix = np.random.normal(0, math.sqrt(self.variance), pointSet[idx].shape)

            'add noise matrix to clean data points'
            pointSet[idx] = pointSet[idx] + noise_matrix

        return pointSet

    def normalize_point_cloud(self, point_cloud):
        """
        Normalizes a point cloud in the range [0, 1] for each dimension

        Inputs:
            point_cloud(numpy array(<nrClusters>,<nrPoints>, 3)): list of points in the given point cloud

        Outputs:
            point_cloud_coordinates(numpy array(<nrClusters>,<nrPoints>, 3)): list of coordinates in range [0, 1]
        """


        'flatten the point cloud to calculate minimum'
        flat_point_cloud = []
        for i in range(point_cloud.shape[0]):
            flat_point_cloud.extend(point_cloud[i])
        flat_point_cloud = np.array(flat_point_cloud)

        'calculate minimum of each coordinate'
        min_coordinates = np.array([np.min(flat_point_cloud[:, 0]), np.min(flat_point_cloud[:, 1]),
                                                                          np.min(flat_point_cloud[:, 2])])

        max_coordinates = np.array([np.max(flat_point_cloud[:, 0]), np.max(flat_point_cloud[:, 1]),
                           np.max(flat_point_cloud[:, 2])]) - min_coordinates

        'align point cloud with coordinate system by shifting the minimum in each dimension to the coordinate axes'
        for i in range(point_cloud.shape[0]):  # for each cluster
            point_cloud[i] = point_cloud[i] - min_coordinates

        'normalization via maximum of the 3 dimensions'
        normalization_factor = np.max(max_coordinates)
        point_cloud = point_cloud / normalization_factor

        return point_cloud

    def sphereGenerate(self, radius, phi_limits=[0.0, 2*np.pi], cos_theta_limits=[-1.0, 1.0]):
        """
        Creates a uniformly sampled sphere
        """

        self.name = 'Sphere'

        pointCloud = []

        for i in range(self.nr_points_polyhedron):
            phi = np.random.uniform(phi_limits[0], phi_limits[1])
            cos_theta = np.random.uniform(cos_theta_limits[0], cos_theta_limits[1])
            theta = np.arccos(cos_theta)

            x = radius * np.cos(phi) * np.sin(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(theta)

            pointCloud.append(np.array([x, y, z]))

        pointCloud = np.array([[np.array(i) for i in pointCloud]])

        'scaling'
        pointCloud = pointCloud * self.scaling

        'apply normal distribution to each point if mode is noisy'
        if self.mode == 'noisy':
            pointCloud = self.applyNoise(pointCloud)

        self.points_polyhedron = pointCloud

        return copy.deepcopy(self.points_polyhedron)

def return_polyhedra(scaleC0=1, scaleC1=1, scaleC2=1, scaleC3=1, scaleC4=1, scaleC5=1, mode=Mode.TRAIN):
    """
    This function returns a dictionary of polyhedra vertex points.
    If one of the scaleC_i inputs is unequal to 1, the result may no more be the polyhedron given via its name.
    This is intended to allow variation for dataset creation.

    Inputs:
        scaleC_i(float): Vertex scaling. Not used currently, as may produce very difficult shapes.
        mode: Mode.TRAIN, Mode.TEST, Mode.PREDICT
    """

    'Platonic Solids'
    # Tetrahedron
    C0 = scaleC0 * math.sqrt(2) / 4

    V0 = [C0, -C0, C0]
    V1 = [C0, C0, -C0]
    V2 = [-C0, C0, C0]
    V3 = [-C0, -C0, -C0]
    tetrahedron = np.array([V0, V1, V2, V3])

    # Octahedron
    C0 = scaleC0 * math.sqrt(2) / 2

    V0 = [0.0, 0.0, C0]
    V1 = [0.0, 0.0, -C0]
    V2 = [C0, 0.0, 0.0]
    V3 = [-C0, 0.0, 0.0]
    V4 = [0.0, C0, 0.0]
    V5 = [0.0, -C0, 0.0]

    octahedron = np.array([V0, V1, V2, V3, V4, V5])

    # Cube
    C0 = scaleC0 * 0.5

    V0 = [C0, C0, C0]
    V1 = [C0, C0, -C0]
    V2 = [C0, -C0, C0]
    V3 = [C0, -C0, -C0]
    V4 = [-C0, C0, C0]
    V5 = [-C0, C0, -C0]
    V6 = [-C0, -C0, C0]
    V7 = [-C0, -C0, -C0]

    cube = np.array([V0, V1, V2, V3, V4, V5, V6, V7])

    #Cuboid

    C0 = scaleC0 * 0.5
    C1 = scaleC1 * 0.5
    C2 = scaleC2 * 0.5

    V0 = [C0, C1, C2]
    V1 = [C0, C1, -C2]
    V2 = [C0, -C1, C2]
    V3 = [C0, -C1, -C2]
    V4 = [-C0, C1, C2]
    V5 = [-C0, C1, -C2]
    V6 = [-C0, -C1, C2]
    V7 = [-C0, -C1, -C2]

    cuboid = np.array([V0, V1, V2, V3, V4, V5, V6, V7])

    # Icosahedron
    C0 = scaleC0 * (1 + math.sqrt(5)) / 4

    V0 = [0.5, 0.0, C0]
    V1 = [0.5, 0.0, -C0]
    V2 = [-0.5, 0.0, C0]
    V3 = [-0.5, 0.0, -C0]
    V4 = [C0, 0.5, 0.0]
    V5 = [C0, -0.5, 0.0]
    V6 = [-C0, 0.5, 0.0]
    V7 = [-C0, -0.5, 0.0]
    V8 = [0.0, C0, 0.5]
    V9 = [0.0, C0, -0.5]
    V10 = [0.0, -C0, 0.5]
    V11 = [0.0, -C0, -0.5]

    icosahedron = np.array([V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11])


    # Dodecahedron
    C0 = scaleC0 * (1 + math.sqrt(5)) / 4
    C1 = scaleC1 * (3 + math.sqrt(5)) / 4

    V0 = [0.0, 0.5, C1]
    V1 = [0.0, 0.5, -C1]
    V2 = [0.0, -0.5, C1]
    V3 = [0.0, -0.5, -C1]
    V4 = [C1, 0.0, 0.5]
    V5 = [C1, 0.0, -0.5]
    V6 = [-C1, 0.0, 0.5]
    V7 = [-C1, 0.0, -0.5]
    V8 = [0.5, C1, 0.0]
    V9 = [0.5, -C1, 0.0]
    V10 = [-0.5, C1, 0.0]
    V11 = [-0.5, -C1, 0.0]
    V12 = [C0, C0, C0]
    V13 = [C0, C0, -C0]
    V14 = [C0, -C0, C0]
    V15 = [C0, -C0, -C0]
    V16 = [-C0, C0, C0]
    V17 = [-C0, C0, -C0]
    V18 = [-C0, -C0, C0]
    V19 = [-C0, -C0, -C0]

    dodecahedron = np.array([V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10,
                             V11, V12, V13, V14, V15, V16, V17, V18, V19])

    'Archimedean Solids'
    # Truncated Tetrahedron
    C0 = scaleC0 * math.sqrt(2) / 4
    C1 = scaleC1 * 3 * math.sqrt(2) / 4

    V0 = [C0, -C0, C1]
    V1 = [C0, C0, -C1]
    V2 = [-C0, C0, C1]
    V3 = [-C0, -C0, -C1]
    V4 = [C1, -C0, C0]
    V5 = [C1, C0, -C0]
    V6 = [-C1, C0, C0]
    V7 = [-C1, -C0, -C0]
    V8 = [C0, -C1, C0]
    V9 = [C0, C1, -C0]
    V10 = [-C0, C1, C0]
    V11 = [-C0, -C1, -C0]

    truncated_tetrahedron = np.array([V0, V1, V2, V3, V4, V5, V6, V7, V8, V9,
                                      V10, V11])

    # Cuboctahedron
    C0 = scaleC0 * math.sqrt(2) / 2

    V0 = [C0, 0.0, C0]
    V1 = [C0, 0.0, -C0]
    V2 = [-C0, 0.0, C0]
    V3 = [-C0, 0.0, -C0]
    V4 = [C0, C0, 0.0]
    V5 = [C0, -C0, 0.0]
    V6 = [-C0, C0, 0.0]
    V7 = [-C0, -C0, 0.0]
    V8 = [0.0, C0, C0]
    V9 = [0.0, C0, -C0]
    V10 = [0.0, -C0, C0]
    V11 = [0.0, -C0, -C0]

    cuboctahedron = np.array([V0, V1, V2, V3, V4, V5, V6, V7, V8, V9,
                              V10, V11])

    # Truncated Octahedron
    C0 = scaleC0 * math.sqrt(2) / 2
    C1 = scaleC1 * math.sqrt(2)

    V0 = [C0, 0.0, C1]
    V1 = [C0, 0.0, -C1]
    V2 = [-C0, 0.0, C1]
    V3 = [-C0, 0.0, -C1]
    V4 = [C1, C0, 0.0]
    V5 = [C1, -C0, 0.0]
    V6 = [-C1, C0, 0.0]
    V7 = [-C1, -C0, 0.0]
    V8 = [0.0, C1, C0]
    V9 = [0.0, C1, -C0]
    V10 = [0.0, -C1, C0]
    V11 = [0.0, -C1, -C0]
    V12 = [0.0, C0, C1]
    V13 = [0.0, C0, -C1]
    V14 = [0.0, -C0, C1]
    V15 = [0.0, -C0, -C1]
    V16 = [C1, 0.0, C0]
    V17 = [C1, 0.0, -C0]
    V18 = [-C1, 0.0, C0]
    V19 = [-C1, 0.0, -C0]
    V20 = [C0, C1, 0.0]
    V21 = [C0, -C1, 0.0]
    V22 = [-C0, C1, 0.0]
    V23 = [-C0, -C1, 0.0]

    truncated_octahedron = np.array([V0, V1, V2, V3, V4, V5, V6, V7, V8, V9,
                                     V10, V11, V12, V13, V14, V15, V16, V17, V18, V19,
                                     V20, V21, V22, V23])

    # Truncated Cube
    C0 = scaleC0 * (1 + math.sqrt(2)) / 2

    V0 = [C0, 0.5, C0]
    V1 = [C0, 0.5, -C0]
    V2 = [C0, -0.5, C0]
    V3 = [C0, -0.5, -C0]
    V4 = [-C0, 0.5, C0]
    V5 = [-C0, 0.5, -C0]
    V6 = [-C0, -0.5, C0]
    V7 = [-C0, -0.5, -C0]
    V8 = [C0, C0, 0.5]
    V9 = [C0, C0, -0.5]
    V10 = [C0, -C0, 0.5]
    V11 = [C0, -C0, -0.5]
    V12 = [-C0, C0, 0.5]
    V13 = [-C0, C0, -0.5]
    V14 = [-C0, -C0, 0.5]
    V15 = [-C0, -C0, -0.5]
    V16 = [0.5, C0, C0]
    V17 = [0.5, C0, -C0]
    V18 = [0.5, -C0, C0]
    V19 = [0.5, -C0, -C0]
    V20 = [-0.5, C0, C0]
    V21 = [-0.5, C0, -C0]
    V22 = [-0.5, -C0, C0]
    V23 = [-0.5, -C0, -C0]

    truncated_cube = np.array([V0, V1, V2, V3, V4, V5, V6, V7, V8, V9,
                               V10, V11, V12, V13, V14, V15, V16, V17, V18, V19,
                               V20, V21, V22, V23])


    # Rhombicuboctahedron

    C0 = scaleC0 * (1 + math.sqrt(2)) / 2

    V0 = [0.5, 0.5, C0]
    V1 = [0.5, 0.5, -C0]
    V2 = [0.5, -0.5, C0]
    V3 = [0.5, -0.5, -C0]
    V4 = [-0.5, 0.5, C0]
    V5 = [-0.5, 0.5, -C0]
    V6 = [-0.5, -0.5, C0]
    V7 = [-0.5, -0.5, -C0]
    V8 = [C0, 0.5, 0.5]
    V9 = [C0, 0.5, -0.5]
    V10 = [C0, -0.5, 0.5]
    V11 = [C0, -0.5, -0.5]
    V12 = [-C0, 0.5, 0.5]
    V13 = [-C0, 0.5, -0.5]
    V14 = [-C0, -0.5, 0.5]
    V15 = [-C0, -0.5, -0.5]
    V16 = [0.5, C0, 0.5]
    V17 = [0.5, C0, -0.5]
    V18 = [0.5, -C0, 0.5]
    V19 = [0.5, -C0, -0.5]
    V20 = [-0.5, C0, 0.5]
    V21 = [-0.5, C0, -0.5]
    V22 = [-0.5, -C0, 0.5]
    V23 = [-0.5, -C0, -0.5]

    rhombicuboctahedron = np.array([V0, V1, V2, V3, V4, V5, V6, V7, V8, V9,
                                    V10, V11, V12, V13, V14, V15, V16, V17, V18, V19,
                                    V20, V21, V22, V23])


    # Icosidodecahedron
    C0 = scaleC0 * (1 + math.sqrt(5)) / 4
    C1 = scaleC1 * (3 + math.sqrt(5)) / 4
    C2 = scaleC2 * (1 + math.sqrt(5)) / 2

    V0 = [0.0, 0.0, C2]
    V1 = [0.0, 0.0, -C2]
    V2 = [C2, 0.0, 0.0]
    V3 = [-C2, 0.0, 0.0]
    V4 = [0.0, C2, 0.0]
    V5 = [0.0, -C2, 0.0]
    V6 = [0.5, C0, C1]
    V7 = [0.5, C0, -C1]
    V8 = [0.5, -C0, C1]
    V9 = [0.5, -C0, -C1]
    V10 = [-0.5, C0, C1]
    V11 = [-0.5, C0, -C1]
    V12 = [-0.5, -C0, C1]
    V13 = [-0.5, -C0, -C1]
    V14 = [C1, 0.5, C0]
    V15 = [C1, 0.5, -C0]
    V16 = [C1, -0.5, C0]
    V17 = [C1, -0.5, -C0]
    V18 = [-C1, 0.5, C0]
    V19 = [-C1, 0.5, -C0]
    V20 = [-C1, -0.5, C0]
    V21 = [-C1, -0.5, -C0]
    V22 = [C0, C1, 0.5]
    V23 = [C0, C1, -0.5]
    V24 = [C0, -C1, 0.5]
    V25 = [C0, -C1, -0.5]
    V26 = [-C0, C1, 0.5]
    V27 = [-C0, C1, -0.5]
    V28 = [-C0, -C1, 0.5]
    V29 = [-C0, -C1, -0.5]

    icosidodecahedron = np.array([V0, V1, V2, V3, V4, V5, V6, V7, V8, V9,
                                  V10, V11, V12, V13, V14, V15, V16, V17, V18, V19,
                                  V20, V21, V22, V23, V24, V25, V26, V27, V28, V29])

    # Truncated Cuboctahedron
    C0 = scaleC0 * (1 + math.sqrt(2)) / 2
    C1 = scaleC1 * (1 + 2 * math.sqrt(2)) / 2

    V0 = [C0, 0.5, C1]
    V1 = [C0, 0.5, -C1]
    V2 = [C0, -0.5, C1]
    V3 = [C0, -0.5, -C1]
    V4 = [-C0, 0.5, C1]
    V5 = [-C0, 0.5, -C1]
    V6 = [-C0, -0.5, C1]
    V7 = [-C0, -0.5, -C1]
    V8 = [C1, C0, 0.5]
    V9 = [C1, C0, -0.5]
    V10 = [C1, -C0, 0.5]
    V11 = [C1, -C0, -0.5]
    V12 = [-C1, C0, 0.5]
    V13 = [-C1, C0, -0.5]
    V14 = [-C1, -C0, 0.5]
    V15 = [-C1, -C0, -0.5]
    V16 = [0.5, C1, C0]
    V17 = [0.5, C1, -C0]
    V18 = [0.5, -C1, C0]
    V19 = [0.5, -C1, -C0]
    V20 = [-0.5, C1, C0]
    V21 = [-0.5, C1, -C0]
    V22 = [-0.5, -C1, C0]
    V23 = [-0.5, -C1, -C0]
    V24 = [0.5, C0, C1]
    V25 = [0.5, C0, -C1]
    V26 = [0.5, -C0, C1]
    V27 = [0.5, -C0, -C1]
    V28 = [-0.5, C0, C1]
    V29 = [-0.5, C0, -C1]
    V30 = [-0.5, -C0, C1]
    V31 = [-0.5, -C0, -C1]
    V32 = [C1, 0.5, C0]
    V33 = [C1, 0.5, -C0]
    V34 = [C1, -0.5, C0]
    V35 = [C1, -0.5, -C0]
    V36 = [-C1, 0.5, C0]
    V37 = [-C1, 0.5, -C0]
    V38 = [-C1, -0.5, C0]
    V39 = [-C1, -0.5, -C0]
    V40 = [C0, C1, 0.5]
    V41 = [C0, C1, -0.5]
    V42 = [C0, -C1, 0.5]
    V43 = [C0, -C1, -0.5]
    V44 = [-C0, C1, 0.5]
    V45 = [-C0, C1, -0.5]
    V46 = [-C0, -C1, 0.5]
    V47 = [-C0, -C1, -0.5]

    truncated_cuboctahedron = np.array([V0, V1, V2, V3, V4, V5, V6, V7, V8, V9,
                                        V10, V11, V12, V13, V14, V15, V16, V17, V18, V19,
                                        V20, V21, V22, V23, V24, V25, V26, V27, V28, V29,
                                        V30, V31, V32, V33, V34, V35, V36, V37, V38, V39,
                                        V40, V41, V42, V43, V44, V45, V46, V47])

    # Truncated Icosahedron
    C0 = scaleC0 * (1 + math.sqrt(5)) / 4
    C1 = scaleC1 * (1 + math.sqrt(5)) / 2
    C2 = scaleC2 * (5 + math.sqrt(5)) / 4
    C3 = scaleC3 * (2 + math.sqrt(5)) / 2
    C4 = scaleC4 * 3 * (1 + math.sqrt(5)) / 4

    V0 = [0.5, 0.0, C4]
    V1 = [0.5, 0.0, -C4]
    V2 = [-0.5, 0.0, C4]
    V3 = [-0.5, 0.0, -C4]
    V4 = [C4, 0.5, 0.0]
    V5 = [C4, -0.5, 0.0]
    V6 = [-C4, 0.5, 0.0]
    V7 = [-C4, -0.5, 0.0]
    V8 = [0.0, C4, 0.5]
    V9 = [0.0, C4, -0.5]
    V10 = [0.0, -C4, 0.5]
    V11 = [0.0, -C4, -0.5]
    V12 = [1.0, C0, C3]
    V13 = [1.0, C0, -C3]
    V14 = [1.0, -C0, C3]
    V15 = [1.0, -C0, -C3]
    V16 = [-1.0, C0, C3]
    V17 = [-1.0, C0, -C3]
    V18 = [-1.0, -C0, C3]
    V19 = [-1.0, -C0, -C3]
    V20 = [C3, 1.0, C0]
    V21 = [C3, 1.0, -C0]
    V22 = [C3, -1.0, C0]
    V23 = [C3, -1.0, -C0]
    V24 = [-C3, 1.0, C0]
    V25 = [-C3, 1.0, -C0]
    V26 = [-C3, -1.0, C0]
    V27 = [-C3, -1.0, -C0]
    V28 = [C0, C3, 1.0]
    V29 = [C0, C3, -1.0]
    V30 = [C0, -C3, 1.0]
    V31 = [C0, -C3, -1.0]
    V32 = [-C0, C3, 1.0]
    V33 = [-C0, C3, -1.0]
    V34 = [-C0, -C3, 1.0]
    V35 = [-C0, -C3, -1.0]
    V36 = [0.5, C1, C2]
    V37 = [0.5, C1, -C2]
    V38 = [0.5, -C1, C2]
    V39 = [0.5, -C1, -C2]
    V40 = [-0.5, C1, C2]
    V41 = [-0.5, C1, -C2]
    V42 = [-0.5, -C1, C2]
    V43 = [-0.5, -C1, -C2]
    V44 = [C2, 0.5, C1]
    V45 = [C2, 0.5, -C1]
    V46 = [C2, -0.5, C1]
    V47 = [C2, -0.5, -C1]
    V48 = [-C2, 0.5, C1]
    V49 = [-C2, 0.5, -C1]
    V50 = [-C2, -0.5, C1]
    V51 = [-C2, -0.5, -C1]
    V52 = [C1, C2, 0.5]
    V53 = [C1, C2, -0.5]
    V54 = [C1, -C2, 0.5]
    V55 = [C1, -C2, -0.5]
    V56 = [-C1, C2, 0.5]
    V57 = [-C1, C2, -0.5]
    V58 = [-C1, -C2, 0.5]
    V59 = [-C1, -C2, -0.5]

    truncated_icosahedron = np.array([V0, V1, V2, V3, V4, V5, V6, V7, V8, V9,
                                      V10, V11, V12, V13, V14, V15, V16, V17, V18, V19,
                                      V20, V21, V22, V23, V24, V25, V26, V27, V28, V29,
                                      V30, V31, V32, V33, V34, V35, V36, V37, V38, V39,
                                      V40, V41, V42, V43, V44, V45, V46, V47, V48, V49,
                                      V50, V51, V52, V53, V54, V55, V56, V57, V58, V59])

    # Truncated Dodecahedron
    C0 = scaleC0 * (3 + math.sqrt(5)) / 4
    C1 = scaleC1 * (1 + math.sqrt(5)) / 2
    C2 = scaleC2 * (2 + math.sqrt(5)) / 2
    C3 = scaleC3 * (3 + math.sqrt(5)) / 2
    C4 = scaleC4 * (5 + 3 * math.sqrt(5)) / 4

    V0 = [0.0, 0.5, C4]
    V1 = [0.0, 0.5, -C4]
    V2 = [0.0, -0.5, C4]
    V3 = [0.0, -0.5, -C4]
    V4 = [C4, 0.0, 0.5]
    V5 = [C4, 0.0, -0.5]
    V6 = [-C4, 0.0, 0.5]
    V7 = [-C4, 0.0, -0.5]
    V8 = [0.5, C4, 0.0]
    V9 = [0.5, -C4, 0.0]
    V10 = [-0.5, C4, 0.0]
    V11 = [-0.5, -C4, 0.0]
    V12 = [0.5, C0, C3]
    V13 = [0.5, C0, -C3]
    V14 = [0.5, -C0, C3]
    V15 = [0.5, -C0, -C3]
    V16 = [-0.5, C0, C3]
    V17 = [-0.5, C0, -C3]
    V18 = [-0.5, -C0, C3]
    V19 = [-0.5, -C0, -C3]
    V20 = [C3, 0.5, C0]
    V21 = [C3, 0.5, -C0]
    V22 = [C3, -0.5, C0]
    V23 = [C3, -0.5, -C0]
    V24 = [-C3, 0.5, C0]
    V25 = [-C3, 0.5, -C0]
    V26 = [-C3, -0.5, C0]
    V27 = [-C3, -0.5, -C0]
    V28 = [C0, C3, 0.5]
    V29 = [C0, C3, -0.5]
    V30 = [C0, -C3, 0.5]
    V31 = [C0, -C3, -0.5]
    V32 = [-C0, C3, 0.5]
    V33 = [-C0, C3, -0.5]
    V34 = [-C0, -C3, 0.5]
    V35 = [-C0, -C3, -0.5]
    V36 = [C0, C1, C2]
    V37 = [C0, C1, -C2]
    V38 = [C0, -C1, C2]
    V39 = [C0, -C1, -C2]
    V40 = [-C0, C1, C2]
    V41 = [-C0, C1, -C2]
    V42 = [-C0, -C1, C2]
    V43 = [-C0, -C1, -C2]
    V44 = [C2, C0, C1]
    V45 = [C2, C0, -C1]
    V46 = [C2, -C0, C1]
    V47 = [C2, -C0, -C1]
    V48 = [-C2, C0, C1]
    V49 = [-C2, C0, -C1]
    V50 = [-C2, -C0, C1]
    V51 = [-C2, -C0, -C1]
    V52 = [C1, C2, C0]
    V53 = [C1, C2, -C0]
    V54 = [C1, -C2, C0]
    V55 = [C1, -C2, -C0]
    V56 = [-C1, C2, C0]
    V57 = [-C1, C2, -C0]
    V58 = [-C1, -C2, C0]
    V59 = [-C1, -C2, -C0]
    
    
    truncated_dodecahedron = np.array([V0, V1, V2, V3, V4, V5, V6, V7, V8, V9,
                                       V10, V11, V12, V13, V14, V15, V16, V17, V18, V19,
                                       V20, V21, V22, V23, V24, V25, V26, V27, V28, V29,
                                       V30, V31, V32, V33, V34, V35, V36, V37, V38, V39,
                                       V40, V41, V42, V43, V44, V45, V46, V47, V48, V49,
                                       V50, V51, V52, V53, V54, V55, V56, V57, V58, V59])

    # Rhombicosidodecahedron
    C0 = scaleC0 * (1 + math.sqrt(5)) / 4
    C1 = scaleC1 * (3 + math.sqrt(5)) / 4
    C2 = scaleC2 * (1 + math.sqrt(5)) / 2
    C3 = scaleC3 * (5 + math.sqrt(5)) / 4
    C4 = scaleC4 * (2 + math.sqrt(5)) / 2

    V0 = [0.5, 0.5, C4]
    V1 = [0.5, 0.5, -C4]
    V2 = [0.5, -0.5, C4]
    V3 = [0.5, -0.5, -C4]
    V4 = [-0.5, 0.5, C4]
    V5 = [-0.5, 0.5, -C4]
    V6 = [-0.5, -0.5, C4]
    V7 = [-0.5, -0.5, -C4]
    V8 = [C4, 0.5, 0.5]
    V9 = [C4, 0.5, -0.5]
    V10 = [C4, -0.5, 0.5]
    V11 = [C4, -0.5, -0.5]
    V12 = [-C4, 0.5, 0.5]
    V13 = [-C4, 0.5, -0.5]
    V14 = [-C4, -0.5, 0.5]
    V15 = [-C4, -0.5, -0.5]
    V16 = [0.5, C4, 0.5]
    V17 = [0.5, C4, -0.5]
    V18 = [0.5, -C4, 0.5]
    V19 = [0.5, -C4, -0.5]
    V20 = [-0.5, C4, 0.5]
    V21 = [-0.5, C4, -0.5]
    V22 = [-0.5, -C4, 0.5]
    V23 = [-0.5, -C4, -0.5]
    V24 = [0.0, C1, C3]
    V25 = [0.0, C1, -C3]
    V26 = [0.0, -C1, C3]
    V27 = [0.0, -C1, -C3]
    V28 = [C3, 0.0, C1]
    V29 = [C3, 0.0, -C1]
    V30 = [-C3, 0.0, C1]
    V31 = [-C3, 0.0, -C1]
    V32 = [C1, C3, 0.0]
    V33 = [C1, -C3, 0.0]
    V34 = [-C1, C3, 0.0]
    V35 = [-C1, -C3, 0.0]
    V36 = [C1, C0, C2]
    V37 = [C1, C0, -C2]
    V38 = [C1, -C0, C2]
    V39 = [C1, -C0, -C2]
    V40 = [-C1, C0, C2]
    V41 = [-C1, C0, -C2]
    V42 = [-C1, -C0, C2]
    V43 = [-C1, -C0, -C2]
    V44 = [C2, C1, C0]
    V45 = [C2, C1, -C0]
    V46 = [C2, -C1, C0]
    V47 = [C2, -C1, -C0]
    V48 = [-C2, C1, C0]
    V49 = [-C2, C1, -C0]
    V50 = [-C2, -C1, C0]
    V51 = [-C2, -C1, -C0]
    V52 = [C0, C2, C1]
    V53 = [C0, C2, -C1]
    V54 = [C0, -C2, C1]
    V55 = [C0, -C2, -C1]
    V56 = [-C0, C2, C1]
    V57 = [-C0, C2, -C1]
    V58 = [-C0, -C2, C1]
    V59 = [-C0, -C2, -C1]
    
    rhombicosidodecahedron = np.array([V0, V1, V2, V3, V4, V5, V6, V7, V8, V9,
                                       V10, V11, V12, V13, V14, V15, V16, V17, V18, V19,
                                       V20, V21, V22, V23, V24, V25, V26, V27, V28, V29,
                                       V30, V31, V32, V33, V34, V35, V36, V37, V38, V39,
                                       V40, V41, V42, V43, V44, V45, V46, V47, V48, V49,
                                       V50, V51, V52, V53, V54, V55, V56, V57, V58, V59])

    'Prisms & Antiprisms'
    # Triangular Prism
    C0 = scaleC0 * math.sqrt(3) / 6
    C1 = scaleC1 * math.sqrt(3) / 3

    V0 = [0.5, -C0, 0.5]
    V1 = [0.5, -C0, -0.5]
    V2 = [-0.5, -C0, 0.5]
    V3 = [-0.5, -C0, -0.5]
    V4 = [0.0, C1, 0.5]
    V5 = [0.0, C1, -0.5]

    triangular_prism = np.array([V0, V1, V2, V3, V4, V5])

    # Pentagonal Prism
    C0 = scaleC0 * math.sqrt(10 * (5 - math.sqrt(5))) / 20
    C1 = scaleC1 * math.sqrt(5 * (5 + 2 * math.sqrt(5))) / 10
    C2 = scaleC2 * (1 + math.sqrt(5)) / 4
    C3 = scaleC3 * math.sqrt(10 * (5 + math.sqrt(5))) / 10

    V0 = [C2, C0, 0.5]
    V1 = [C2, C0, -0.5]
    V2 = [-C2, C0, 0.5]
    V3 = [-C2, C0, -0.5]
    V4 = [0.5, -C1, 0.5]
    V5 = [0.5, -C1, -0.5]
    V6 = [-0.5, -C1, 0.5]
    V7 = [-0.5, -C1, -0.5]
    V8 = [0.0, C3, 0.5]
    V9 = [0.0, C3, -0.5]

    pentagonal_prism = np.array([V0, V1, V2, V3, V4, V5, V6, V7, V8, V9])

    # Hexagonal Prism
    C0 = scaleC0 * math.sqrt(3) / 2

    V0 = [1.0, 0.0, 0.5]
    V1 = [1.0, 0.0, -0.5]
    V2 = [-1.0, 0.0, 0.5]
    V3 = [-1.0, 0.0, -0.5]
    V4 = [0.5, C0, 0.5]
    V5 = [0.5, C0, -0.5]
    V6 = [0.5, -C0, 0.5]
    V7 = [0.5, -C0, -0.5]
    V8 = [-0.5, C0, 0.5]
    V9 = [-0.5, C0, -0.5]
    V10 = [-0.5, -C0, 0.5]
    V11 = [-0.5, -C0, -0.5]

    hexagonal_prism = np.array([V0, V1, V2, V3, V4, V5, V6, V7, V8, V9,
                                V10, V11])

    # Heptagonal Prism
    C0 = scaleC0 * 1 / (4 * math.cos(math.pi / 14))
    C1 = scaleC1 * (1 + math.cos(math.pi / 7)) * math.sqrt(7) / 7
    C2 = scaleC2 * math.cos(math.pi / 7)
    C3 = scaleC3 * 1 / (math.tan(math.pi / 7) * 2)
    C4 = scaleC4 * 1 / (4 * math.sin(math.pi / 14))
    C5 = scaleC5 * 1 / (2 * math.sin(math.pi / 7))

    V0 = [C4, -C0, 0.5]
    V1 = [C4, -C0, -0.5]
    V2 = [-C4, -C0, 0.5]
    V3 = [-C4, -C0, -0.5]
    V4 = [C2, C1, 0.5]
    V5 = [C2, C1, -0.5]
    V6 = [-C2, C1, 0.5]
    V7 = [-C2, C1, -0.5]
    V8 = [0.5, -C3, 0.5]
    V9 = [0.5, -C3, -0.5]
    V10 = [-0.5, -C3, 0.5]
    V11 = [-0.5, -C3, -0.5]
    V12 = [0.0, C5, 0.5]
    V13 = [0.0, C5, -0.5]

    heptagonal_prism = np.array([V0, V1, V2, V3, V4, V5, V6, V7, V8, V9,
                                 V10, V11, V12, V13])

    # Octagonal Prism
    C0 = scaleC0 * (1 + math.sqrt(2)) / 2

    V0 = [C0, 0.5, 0.5]
    V1 = [C0, 0.5, -0.5]
    V2 = [C0, -0.5, 0.5]
    V3 = [C0, -0.5, -0.5]
    V4 = [-C0, 0.5, 0.5]
    V5 = [-C0, 0.5, -0.5]
    V6 = [-C0, -0.5, 0.5]
    V7 = [-C0, -0.5, -0.5]
    V8 = [0.5, C0, 0.5]
    V9 = [0.5, C0, -0.5]
    V10 = [0.5, -C0, 0.5]
    V11 = [0.5, -C0, -0.5]
    V12 = [-0.5, C0, 0.5]
    V13 = [-0.5, C0, -0.5]
    V14 = [-0.5, -C0, 0.5]
    V15 = [-0.5, -C0, -0.5]

    octagonal_prism = np.array([V0, V1, V2, V3, V4, V5, V6, V7, V8, V9,
                                V10, V11, V12, V13, V14, V15])

    'Dipyramids & Trapezohedra'
    # Triangular Dipyramid
    C0 = scaleC0 * math.sqrt(3) / 3
    C1 = scaleC1 * 2 / 3
    C2 = scaleC2 * 2 * math.sqrt(3) / 3

    V0 = [0.0, 0.0, C1]
    V1 = [0.0, 0.0, -C1]
    V2 = [1.0, C0, 0.0]
    V3 = [-1.0, C0, 0.0]
    V4 = [0.0, -C2, 0.0]

    triangular_dipyramid = np.array([V0, V1, V2, V3, V4])

    # Pentagonal Dipyramid
    C0 = scaleC0 * math.sqrt(5 * (5 - 2 * math.sqrt(5))) / 5
    C1 = scaleC1 * (math.sqrt(5) - 1) / 2
    C2 = scaleC2 * math.sqrt(10 * (5 + math.sqrt(5))) / 10
    C3 = scaleC3 * math.sqrt(10 * (5 - math.sqrt(5))) / 5
    C4 = scaleC4 * (5 + math.sqrt(5)) / 5

    V0 = [0.0, 0.0, C4]
    V1 = [0.0, 0.0, -C4]
    V2 = [1.0, -C0, 0.0]
    V3 = [-1.0, -C0, 0.0]
    V4 = [C1, C2, 0.0]
    V5 = [-C1, C2, 0.0]
    V6 = [0.0, -C3, 0.0]

    pentagonal_dipyramid = np.array([V0, V1, V2, V3, V4, V5, V6])

    'Johnson solids'
    # Square Pyramid
    C0 = scaleC0 * math.sqrt(2) / 2

    V0 = [C0, 0.0, 0.0]
    V1 = [-C0, 0.0, 0.0]
    V2 = [0.0, C0, 0.0]
    V3 = [0.0, -C0, 0.0]
    V4 = [0.0, 0.0, C0]

    square_pyramid = np.array([V0, V1, V2, V3, V4])

    # Pentagonal Pyramid
    C0 = scaleC0 * math.sqrt(10 * (5 - math.sqrt(5))) / 20
    C1 = scaleC1 * (10 + math.sqrt(10 * (5 - math.sqrt(5)))) / 10
    C2 = scaleC2 * math.sqrt(5 * (5 + 2 * math.sqrt(5))) / 10
    C3 = scaleC3 * (1 + math.sqrt(5)) / 4
    C4 = scaleC4 * math.sqrt(10 * (5 + math.sqrt(5))) / 10

    V0 = [C3, C0, 0.0]
    V1 = [-C3, C0, 0.0]
    V2 = [0.5, -C2, 0.0]
    V3 = [-0.5, -C2, 0.0]
    V4 = [0.0, C4, 0.0]
    V5 = [0.0, 0.0, C1]

    pentagonal_pyramid = np.array([V0, V1, V2, V3, V4, V5])

    # Triangular Cupola
    C0 = scaleC0 * math.sqrt(2) / 2

    V0 = [C0, 0.0, C0]
    V1 = [C0, 0.0, -C0]
    V2 = [-C0, 0.0, C0]
    V3 = [C0, C0, 0.0]
    V4 = [C0, -C0, 0.0]
    V5 = [-C0, C0, 0.0]
    V6 = [0.0, C0, C0]
    V7 = [0.0, C0, -C0]
    V8 = [0.0, -C0, C0]

    triangular_cupola = np.array([V0, V1, V2, V3, V4, V5, V6, V7, V8])

    # Square Cupola
    C0 = scaleC0 * math.sqrt(2) / 2
    C1 = scaleC1 * (1 + math.sqrt(2)) / 2

    V0 = [C1, 0.5, 0.0]
    V1 = [C1, -0.5, 0.0]
    V2 = [-C1, 0.5, 0.0]
    V3 = [-C1, -0.5, 0.0]
    V4 = [0.5, C1, 0.0]
    V5 = [0.5, -C1, 0.0]
    V6 = [-0.5, C1, 0.0]
    V7 = [-0.5, -C1, 0.0]
    V8 = [0.5, 0.5, C0]
    V9 = [0.5, -0.5, C0]
    V10 = [-0.5, 0.5, C0]
    V11 = [-0.5, -0.5, C0]

    square_cupola = np.array([V0, V1, V2, V3, V4, V5, V6, V7, V8, V9,
                              V10, V11])

    # Elongated Square Pyramid
    C0 = scaleC0 * (1 + math.sqrt(2)) / 2

    V0 = [0.5, 0.5, 0.5]
    V1 = [0.5, 0.5, -0.5]
    V2 = [0.5, -0.5, 0.5]
    V3 = [0.5, -0.5, -0.5]
    V4 = [-0.5, 0.5, 0.5]
    V5 = [-0.5, 0.5, -0.5]
    V6 = [-0.5, -0.5, 0.5]
    V7 = [-0.5, -0.5, -0.5]
    V8 = [0.0, 0.0, C0]

    elongated_square_pyramid = np.array([V0, V1, V2, V3, V4, V5, V6, V7, V8])

    # Elongated Pentagonal Pyramid
    C0 = scaleC0 * math.sqrt(10 * (5 - math.sqrt(5))) / 20
    C1 = scaleC1 * math.sqrt(5 * (5 + 2 * math.sqrt(5))) / 10
    C2 = scaleC2 * (1 + math.sqrt(5)) / 4
    C3 = scaleC3 * math.sqrt(10 * (5 + math.sqrt(5))) / 10
    C4 = scaleC4 * (5 + math.sqrt(10 * (5 - math.sqrt(5)))) / 10

    V0 = [C2, C0, 0.5]
    V1 = [C2, C0, -0.5]
    V2 = [-C2, C0, 0.5]
    V3 = [-C2, C0, -0.5]
    V4 = [0.5, -C1, 0.5]
    V5 = [0.5, -C1, -0.5]
    V6 = [-0.5, -C1, 0.5]
    V7 = [-0.5, -C1, -0.5]
    V8 = [0.0, C3, 0.5]
    V9 = [0.0, C3, -0.5]
    V10 = [0.0, 0.0, C4]

    elongated_pentagonal_pyramid = np.array([V0, V1, V2, V3, V4, V5, V6, V7, V8, V9,
                                             V10])
    # Elongated Triangular Dipyramid
    C0 = scaleC0 * math.sqrt(3) / 6
    C1 = scaleC1 * math.sqrt(3) / 3
    C2 = scaleC2 * (3 + 2 * math.sqrt(6)) / 6

    V0 = [0.5, -C0, 0.5]
    V1 = [0.5, -C0, -0.5]
    V2 = [-0.5, -C0, 0.5]
    V3 = [-0.5, -C0, -0.5]
    V4 = [0.0, C1, 0.5]
    V5 = [0.0, C1, -0.5]
    V6 = [0.0, 0.0, C2]
    V7 = [0.0, 0.0, -C2]

    elongated_triangular_dipyramid = np.array([V0, V1, V2, V3, V4, V5, V6, V7])

    # Elongated Square Dipyramid
    C0 = scaleC0 * (1 + math.sqrt(2)) / 2

    V0 = [0.5, 0.5, 0.5]
    V1 = [0.5, 0.5, -0.5]
    V2 = [0.5, -0.5, 0.5]
    V3 = [0.5, -0.5, -0.5]
    V4 = [-0.5, 0.5, 0.5]
    V5 = [-0.5, 0.5, -0.5]
    V6 = [-0.5, -0.5, 0.5]
    V7 = [-0.5, -0.5, -0.5]
    V8 = [0.0, 0.0, C0]
    V9 = [0.0, 0.0, -C0]

    elongated_square_dipyramid = np.array([V0, V1, V2, V3, V4, V5, V6, V7, V8, V9])

    # Elongated Pentagonal Dipyramid
    C0 = scaleC0 * math.sqrt(10 * (5 - math.sqrt(5))) / 20
    C1 = scaleC1 * math.sqrt(5 * (5 + 2 * math.sqrt(5))) / 10
    C2 = scaleC2 * (1 + math.sqrt(5)) / 4
    C3 = scaleC3 * math.sqrt(10 * (5 + math.sqrt(5))) / 10
    C4 = scaleC4 * (5 + math.sqrt(10 * (5 - math.sqrt(5)))) / 10

    V0 = [C2, C0, 0.5]
    V1 = [C2, C0, -0.5]
    V2 = [-C2, C0, 0.5]
    V3 = [-C2, C0, -0.5]
    V4 = [0.5, -C1, 0.5]
    V5 = [0.5, -C1, -0.5]
    V6 = [-0.5, -C1, 0.5]
    V7 = [-0.5, -C1, -0.5]
    V8 = [0.0, C3, 0.5]
    V9 = [0.0, C3, -0.5]
    V10 = [0.0, 0.0, C4]
    V11 = [0.0, 0.0, -C4]

    elongated_pentagonal_dipyramid = np.array([V0, V1, V2, V3, V4, V5, V6, V7, V8, V9,
                                               V10, V11])
    'Note: Johnson solids have more shapes that can be used based on convex hull'

    # Cube Cutoff
    C0 = 0.5
    C1 = -0.1

    V0 = [C1, C0, C0]
    V1 = [C1, C0, -C0]
    V2 = [C0, -C0, C0]
    V3 = [C0, -C0, -C0]
    V4 = [-C0, C0, C0]
    V5 = [-C0, C0, -C0]
    V6 = [-C0, -C0, C0]
    V7 = [-C0, -C0, -C0]

    cube_cutoff = np.array([V0, V1, V2, V3, V4, V5, V6, V7])

    # Cut-off Parallelogram with offset
    C0 = np.sqrt(2) / 2

    V0 = [C0, 0, 0.5]
    V1 = [-C0, 0, 0.5]
    V2 = [C0, 0, -0.5]
    V3 = [-C0, 0, -0.5]
    V4 = [0.2, 1, 0.5]
    V5 = [0.2, -1, 0.5]
    V6 = [0.2, 1, -0.5]
    V7 = [0.2, -1, -0.5]
    V8 = [-0.2, 1, 0.5]
    V9 = [-0.2, -1, 0.5]
    V10 = [-0.2, 1, -0.5]
    V11 = [-0.2, -1, -0.5]

    cutoff_parallelogram_offset = np.array([V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11])

    if mode == Mode.TRAIN:

        dict_polyhedra = {
            "Tetrahedron": tetrahedron,
            "Octahedron": octahedron,
            #"Cuboid": cuboid,  covered by cube if squashing is used
            #"Icosahedron": icosahedron, too coarse cube capping causes problems with triangles
            "Dodecahedron": dodecahedron,
            "Truncated Tetrahedron": truncated_tetrahedron,
            "Cuboctahedron": cuboctahedron,
            "Truncated Octahedron": truncated_octahedron,
            #"Truncated Cube": truncated_cube,
            #"Rhombicuboctahedron": rhombicuboctahedron, too coarse cube capping causes problems with triangles
            #"Icosidodecahedron": icosidodecahedron,
            #"Truncated Cuboctahedron": truncated_cuboctahedron,
            #"Truncated Icosahedron": truncated_icosahedron, looks like sphere
            #"Truncated Dodecahedron": truncated_dodecahedron, triangles too unstructured
            #"Rhombicosidodecahedron": rhombicosidodecahedron, looks like sphere
            "Triangular Prism": triangular_prism,
            "Pentagonal Prism": pentagonal_prism,
            "Hexagonal Prism": hexagonal_prism,
            "Heptagonal Prism": heptagonal_prism,
            #"Triangular Dipyramid": triangular_dipyramid, too coarse cube capping causes problems with triangles
            #"Pentagonal Dipyramid": pentagonal_pyramid, too coarse cube capping causes problems with triangles
            "Square Pyramid": square_pyramid,
            "Triangular Cupola": triangular_cupola,
            "Square Cupola": square_cupola,
            #"Elongated Pentagonal Pyramid": elongated_pentagonal_pyramid,
            #"Elongated Triangular Dipyramid": elongated_triangular_dipyramid,  too coarse cube capping causes problems with triangles
            #"Elongated Square Dipyramid": elongated_square_dipyramid, too coarse cube capping causes problems with triangles
            #"Elongated Pentagonal Dipyramid": elongated_pentagonal_dipyramid   too coarse cube capping causes problems woth triangles
            "Cube Cutoff": cube_cutoff,
            "Cutoff Parallelogram Offset": cutoff_parallelogram_offset,
        }
    elif mode == Mode.TEST or mode == Mode.PREDICT:
        dict_polyhedra = {
            "Cube": cube,
            "Octagonal Prism": octagonal_prism,
            "Pentagonal Pyramid": pentagonal_pyramid,
        }
    return dict_polyhedra
