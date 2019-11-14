import sys
sys.path.append('../')
sys.setrecursionlimit(50000)

import numpy as np
import copy
from numpy import linalg as LA
import random

import plotly.offline as py
import plotly.graph_objs as go
import math
from Classic_RGS.point import Point
from search import Search
from utils.mapRGB import mapRGB
import os


class NormalEstimation:
    '''
    Compute the normals for the set of points given via inputCloud
    '''

    def __init__(self, inputCloud, viewPoint=[0, 0, 0]):
        '''
        Inputs:
            inputCloud: set of points for which normal estimation is to be done
            viewPoint: point towards which normals are flipped in case of ambiguity
        '''

        self.inputCloud = inputCloud
        'store viewpoint as column vector'
        self.viewPoint = np.reshape(viewPoint, (3, 1))

    def compute3DCentroid(self, pointSet):
        '''
        Computes the 3D (X,Y,Z) Centroid of a set of points given as
        a 2D-tensor

        Inputs:     pointSet as np.array
                        with size (length,3) where each row is [x,y,z]
        Outputs:    centroid [x,y,z] as vector with shape (3,1)
                        This is the geometric center of the pointSet
        '''

        'Initialisation as 0'
        centroid = np.zeros((3,1))

        'sum the x/y/z-values for all points'
        for idx in range(pointSet.shape[0]):
            centroid[0] = centroid[0]+pointSet[idx][0]
            centroid[1] = centroid[1]+pointSet[idx][1]
            centroid[2] = centroid[2]+pointSet[idx][2]

        'division by number of points to get an average'
        centroid = centroid/pointSet.shape[0]

        return centroid

    def computeCovarianceMatrix(self, pointSet, centroid):
        '''
        This method computes the normalized covariance matrix acc.
        to formula (4.7) given in
        [1] Radu Bogdan Rusu (2009): Semantic 3D Object Maps for Everyday
            Manipulation in Human Living Environments

        Inputs:     pointSet as np.array
                        with size (length,3) where each row is [x,y,z]
                    centroid [x,y,z] as np.array.
                        This is the geometric center of the pointSet
        '''

        if pointSet.shape[0] == 0:
            print('received empty set of points')
            return 0

        'Covariance Matrix initialisation'
        COV_3x3 = np.zeros((3, 3), dtype='double')
        'p_i from formula'
        p_idx = np.zeros(3)

        '''sum up all the contributions from each point to the
        covariance matrix acc. formula, ref. above'''
        for idx in range(pointSet.shape[0]):
            p_idx[0] = pointSet[idx][0] - centroid[0]
            p_idx[1] = pointSet[idx][1] - centroid[1]
            p_idx[2] = pointSet[idx][2] - centroid[2]

            'list to column vector'
            p_idx_vect = p_idx.reshape(3, 1)

            'outer product'
            COV_3x3_idx = np.matmul(p_idx_vect, p_idx_vect.transpose())

            'summation'
            COV_3x3 = COV_3x3 + COV_3x3_idx

        'normalization acc. formula'
        COV_3x3 = 1/pointSet.shape[0]*COV_3x3

        return COV_3x3

    def flipNormalTowardsViewpoint(self, point, normal):
        '''
        Orientation of normal corresponding to a point as computed here
        is ambiguous. Solve via n_{i} \dot (v_{p}-p_{i}) > 0

        Reference:
            http://pointclouds.org/documentation/tutorials/normal_estimation.php#normal-estimation

        Inputs: point: column vector shape (3,1) meaning [x,y,z]^T
                normal: column vector shape (4,1) meaning Hessian form

        Outputs: normal: flipped normal vector shape (4,1) meaning Hessian form
        '''

        'Centering'
        centeredPoint = point-self.viewPoint
        centeredPoint = np.append(centeredPoint, 0)
        centeredPoint = np.reshape(centeredPoint, (4, 1))

        'Dot product and flip if not oriented towards viewpoint'
        if (np.dot(np.transpose(normal), centeredPoint) < 0):
            normal[0] = -1*normal[0]
            normal[1] = -1*normal[1]
            normal[2] = -1*normal[2]

        return normal

    def computeFeature(self):
        '''
        Computes the normal and surfaceCurvature belonging to each point and returns as list

        Inputs: pointSet: set of points in local neighborhood
                    shape (len,3,1), meaning column vector [x,y,z]^T

        Outputs: output:    normals [a,b,c] acc. ax+by+cz
                                and surfaceCurvatures ordered in accordance
                                with order of pointSet
        Based on method in:
            http://docs.pointclouds.org/1.9.1/classpcl_1_1_normal_estimation.html
        '''

        'Initialization'
        normals = []
        surfaceCurvatures = []

        'loop through all points in pointSet'
        for idx in range(self.inputCloud.shape[0]):

            planeParameters, surfaceCurvature = \
                self.computePointNormal(self.inputCloud)
            normal = self.flipNormalTowardsViewpoint(np.reshape(self.inputCloud[idx], (3, 1)),
                                            planeParameters)

            'appending lists ordered by indices of pointSet'
            if normals == [] and surfaceCurvatures == []:
                normals = np.array([[normal[0], normal[1], [normal[2]]]])
                surfaceCurvatures = [surfaceCurvature]
            else:
                normals = np.append(normals,
                                         [[normal[0], normal[1], [normal[2]]]], axis=0)
                surfaceCurvatures = np.append(surfaceCurvatures, surfaceCurvature)

        return normals, surfaceCurvatures

    def computePointNormal(self, pointSet):
        '''
        Compute the Least-Squares plane fit for a given set of points,
        using their indices, and return the estimated plane parameters
        together with the surface curvature.

        Inputs:     pointSet as np.array
                        with size (length,3) where each row is [x,y,z]

        Outputs:    planeParameters as vector with size (4,1) representing
                        coefficients a,b,c,-d of plane in
                        Hessian form a*x+b*y+c*z=-d
                    surfaceCurvature 1D scalar with equation:
                        lambda_0/(lambda_0+lambda_1+lambda_2)

        Based on:
            http://docs.pointclouds.org/1.9.1/features_2include_2pcl_2features_2normal__3d_8h_source.html#l00060
        '''

        'Estimate the XYZ centroid'
        centroid = self.compute3DCentroid(pointSet)

        'Compute 3x3 covariance matrix'
        COV_3x3 = self.computeCovarianceMatrix(pointSet, centroid)

        'Get normal plane and surface curvature'
        [planeParameters, surfaceCurvature] = \
            self.solvePlaneParameters(COV_3x3, centroid)

        '''Deepcopy required to avoid changes to objects during
        computation of further points'''
        return [copy.deepcopy(planeParameters),
                copy.deepcopy(surfaceCurvature)]

    def solvePlaneParameters(self, covarianceMatrix_3x3, centroid):
        '''
        Solve the eigenvalues and eigenvectors of a given
        3x3 covariance matrix, and estimate the least-squares plane normal
        and surface curvature.

        Inputs:     covarianceMatrix 3x3
                    centroid vector shape (3,1) meaning [x,y,z]^T

        Outputs:    planeParameters as vector with size (4,1) representing
                        coefficients a,b,c,-d of plane in
                        Hessian form a*x+b*y+c*z=-d
                    surfaceCurvature 1D scalar with equation:
                        lambda_0/(lambda_0+lambda_1+lambda_2)

        Based on solvePlaneParameters method from module features, ref.
        http://docs.pointclouds.org/1.9.1/group__features.html
        '''

        'add 4th row to centroid to make it (4,1)-shape as required for \
        transpose below'
        centroid = np.append(centroid, 0)
        centroid = np.reshape(centroid, (4, 1))

        'Compute smallest eigenvalue and corresponding eigenvector'
        eigenValue, eigenVector = self.eigen33(covarianceMatrix_3x3)

        'Compute curvature surface change'
        'Note: trace(COV)=sum of eigenvalues for semidefinite matrices'
        eigSum = covarianceMatrix_3x3[0][0] + covarianceMatrix_3x3[1][1] + \
            covarianceMatrix_3x3[2][2]

        if eigSum != 0:
            surfaceCurvature = eigenValue / eigSum
        else:
            surfaceCurvature = 0

        planeParameters = np.array((eigenVector[0], eigenVector[1],
                                   eigenVector[2], 0))
        planeParameters = np.reshape(planeParameters, (4, 1))
        'Computation of distance d of plane from origin (here: centroid) \
        for Hessian normal form -d = normal_vector \dot centroid_vector'
        planeParameters[3] = -1*np.dot(np.transpose(centroid), planeParameters)

        return planeParameters, surfaceCurvature

    def eigen33(self, mat):
        '''
        Compute smallest eigenvalue and its eigenvectors of the symmetric
        semidefinite input matrix

        Inputs: mat      symmetric, semidefinite input matrix size (...,M,M).
                            Here typically 3x3 covariance matrix

        Outputs:    eigenvalue smallest eigenvalue of input matrix (scalar)
                    eigenvector corresponding to smallest eigenvalue
                    size (M,1) (corresponds to plane normal vector)

        Based on method in
        http://docs.pointclouds.org/trunk/group__common.html
        '''

        eigenValues, eigenVectorsNormalized = LA.eigh(mat)

        'rounding due to numerical problems (negative eigenvalues in order e-31)'
        eigenValues = np.around(eigenValues, decimals=15)

        '''Note: eigenvalues are returned in ascending order.
        Thus index 0 is smallest eigenvalue'''
        if(eigenValues[0] < 0):
            print('File: rgs.py; Attention: Eigenvalues are ' +
                  'expected to be positive. Value is: '+str(eigenValues[0]))

        return eigenValues[0], eigenVectorsNormalized[:, 0]

class RegionGrowing:
    '''
    Python implementation based on
    docs.pointclouds.org/1.9.1/classpcl_1_1_region_growing.html
    '''
    def __init__(self, inputCloud, smoothnessThreshold=(3.0/180*math.pi), curvatureThreshold=1.0,
                 minClusterSize=10):
        '''
        Inputs:
            inputCloud
            smoothnessThreshold
            curvatureThreshold
            minClusterSize
            
            all default values acc. 
                docs.pointclouds.org/1.9.1/classpcl_1_1_region_growing.html
        '''

        'conversion to shape (<nrPoints>, 3, 1)'
        if len(inputCloud.shape) == 3 and inputCloud.shape[1] == 3 and inputCloud.shape[2] == 1: #as required
            self.inputCloud = inputCloud

        elif len(inputCloud.shape) == 3 and inputCloud.shape[1] == 1 and inputCloud.shape[2] == 3: #row vectors, size(<nrPoints>, 1, 3)
            'reshape input cloud to (len, 3,1) shape'
            for i in range(inputCloud.shape[0]):
                if i == 0:
                    self.inputCloud = np.expand_dims(np.reshape(inputCloud[i], (3, 1)), axis=0)
                else:
                    point = np.expand_dims(np.reshape(inputCloud[i], (3, 1)), axis=0)
                    self.inputCloud = np.concatenate((self.inputCloud, point), axis=0)

        elif len(inputCloud.shape) == 2 and inputCloud.shape[1] == 3: #size(<nrPoints>,3)
            'add another dimension to have column vectors'
            self.inputCloud = np.expand_dims(inputCloud, axis = 2)

        else:
            raise TypeError('Shape of input must be (<nrPoints, 1, 3) or (<nrPoints>, 3)')


        self.nr_points = self.inputCloud.shape[0]
        self.clusters = []
        self.numberSegments = 0
        self.smoothnessThreshold = smoothnessThreshold
        self.curvatureThreshold = curvatureThreshold
        self.minClusterSize = minClusterSize

        'Provision to store point objects'
        self.points = 0

    def getClusters(self):
        'return all found clusters'
        return self.clusters

    def getPoints(self):
        'return all processed point objects'
        return self.points

    def extract(self, method='KNN', k=30, radius=0.5, viewPoint=[0,0,0]):
        '''
        This method launches the segmentation algorithm and returns 
        the clusters that were obtained during the segmentation

        Inputs: method: 'KNN' or 'radius'
                k: number of k nearest neighbours if KNN method chosen
                radius: radius of search sphere if radius method chosen
        Outputs:
        
        Based on:
            http://docs.pointclouds.org/1.9.1/region__growing_8hpp_source.html#l00261
        '''

        'get neighbours for each point'
        neighbours = self.findPointNeighbours(method=method, k=k, radius=radius)

        'len(neighbours)==len(inputCloud) -> iterate over all points'
        for i in range(len(neighbours)):
            'get normals and curvature associated to each point in the input point cloud.' \
            'pointParameters is size tuple((len,3,1),(len,3,1),(1,len)) meaning tuple(points, normals, surfaceCurvatures)'

            'list of neighbours for each point separately'
            pointSet = np.array([j for j in self.inputCloud[neighbours[i]]])

            ne = NormalEstimation(pointSet, viewPoint=viewPoint)

            #ne = NormalEstimation(self.inputCloud, viewPoint=viewPoint)
            normals, surfaceCurvatures = ne.computeFeature()

            if i == 0:
                'first element in neighbour set is the self point -> extract normal and curvature of 0th element only,' \
                'because we compute normal for each point separately via its local neighbourhood'
                self.points = [
                    Point(np.array([[self.inputCloud[i][0]], [self.inputCloud[i][1]], [self.inputCloud[i][2]]]),
                          np.array([normals[0][0], normals[0][1], [normals[0][2]]]), surfaceCurvatures[0])]
            else:
                self.points.append(Point(np.array([[self.inputCloud[i][0]], [self.inputCloud[i][1]], [self.inputCloud[i][2]]]),
                          np.array([normals[0][0], normals[0][1], [normals[0][2]]]), surfaceCurvatures[0]))

        for i in range(len(self.points)):
            self.points[i].setNeighbours(neighbours[i])

        segments = self.applySmoothRegionGrowingAlgorithm(copy.deepcopy(self.points))

        'Do the clustering'
        outlierList = []
        for idx in range(len(segments)):
            if len(segments[idx]) >= self.minClusterSize:
                self.clusters.append(segments[idx])
            else:
                outlierList.append(segments[idx])

        'append outlierlist at the end'
        self.clusters.append(outlierList)

    def findPointNeighbours(self, method, k, radius):
        '''
        This method finds KNN for each point and saves them to the array 
        because the algorithm needs to find KNN a few times. 
        
        Inputs:
            method: Either 'KNN' or 'radius'
            radius: radius of search sphere if method is 'radius'
            k:      number of neighbours im method is 'KNN'
        Output: List of indices of neighbours for each point ordered in order of self.inputCloud
        '''
        if k > self.inputCloud.shape[0]:
            k = self.inputCloud.shape[0]
        
        search = Search(np.squeeze(self.inputCloud, axis=2))
        
        pointNeighbours = [np.array([]) for _ \
                               in range(self.inputCloud.shape[0])]
        
        for idx in range(self.inputCloud.shape[0]):
            if method == 'radius':
                pointNeighbours[idx] = \
                    search.radiusSearch(self.inputCloud[idx], radius)
            else:
                'apply k nearest neighbour search by default'
                pointNeighbours[idx] = \
                    search.nearestKSearch(self.inputCloud[idx], k)
        
        return pointNeighbours

    def applySmoothRegionGrowingAlgorithm(self, pointParameters):
        """
        This function implements the algorithm described in the article
        "Segmentation of point clouds using smoothness constraint" by Rabbani et al.

        Input:  pointParameters: list of point objects

        Output: regionList: list of sublists(segments). Each segment contains the points that belong in this segment.

        Based on:
            http://docs.pointclouds.org/1.9.1/classpcl_1_1_region_growing.html#ad9fdb4c1edc24da8b149866dcfc819d5
        """

        regionList = []

        while np.count_nonzero(pointParameters) != 0:

            curvaturesList = []

            for i in range(len(pointParameters)):
                try:
                    curvaturesList.append(pointParameters[i].getCurvature())
                except:
                    'if pointParameters[i] is 0, i.e. has been removed'
                    curvaturesList.append(float("inf"))

            seedList = []
            currRegionList = []

            'find point with minimum curvature. If multiple points have minimum curvature, choose first one.'
            #idx_min = np.argmin(pointParameters[2])
            idx_min = np.argmin(curvaturesList)

            'use point with minimum curvature as current seed. shape(1,3)'
            #seed = np.array([[pointParameters[0][idx_min], pointParameters[1][idx_min], pointParameters[2][idx_min]]])
            seed = pointParameters[idx_min]
            seedList.append(seed)
            'seed also starts a new region'
            currRegionList.append(seed)

            'indicate removed point from original list by setting to 0 in order to keep correct indexing'
            pointParameters[idx_min] = 0

            'while loop to allow update of seedList length during execution'
            i = 0
            while i <= len(seedList)-1:
                neighbours = seedList[i].getNeighbours()
                for j in range(len(neighbours)):
                    neighbourPoint = pointParameters[neighbours[j]]
                    'neighbourPoint is 0 if already removed from original point set or if the same point.' \
                    'Dot product verifies if normal angle difference is higher than than threshold'

                    if neighbourPoint != 0:
                        'calculate dot product and round in order to improve numerical stability'
                        dotProduct = np.around(np.abs(
                            np.dot(np.transpose(neighbourPoint.getNormal()), seedList[i].getNormal())), decimals=10)
                        if dotProduct > 1:
                            print('rgs.py: Attention: dotProduct > 1 -> arccos(dotProduct) undefined')
                        if np.arccos(dotProduct) < self.smoothnessThreshold:
                            currRegionList.append(neighbourPoint)
                            pointParameters[neighbours[j]] = 0
                            'add current neighbour point to list of seed points'
                            if neighbourPoint.getCurvature() < self.curvatureThreshold:
                                seedList.append(neighbourPoint)
                i = i+1

            'end of region -> add found region to regionList'
            regionList.append(currRegionList)

        print('Region Finding Finished')
        return regionList

class Visualization:
    '''

    Note:
        - Arrow visualization does not work in visdom
        - 3D Arrow and points visualization in the same plot worked in no
          python tool that I found (plotly only support 2D arrows as
          annotations which adapt their position depending on viewer view)
        - Arrow visualization works in plotly only separately via cone plot
            https://plot.ly/python/cone-plot/
            https://plot.ly/python/reference/#cone
        - Other 3D point visualization libraries for python:
                https://github.com/daavoo/pyntcloud
                https://github.com/heremaps/pptk
    '''

    def __init__(self, output_path, name, temp=False):
        """
        Inputs:
            output_path(str): Directory where html files shall be stored
            name(str): Name of the outputs
            temp(bool): True, if html files are only temporary and shall overwrite existing files. Else False.
        """
        self.output_path = output_path
        self.name = name

        if not os.path.exists(output_path):
            os.makedirs(output_path)

    def displayNormals(self, pointSet, sizeref=1, jupyter=False, auto_open=False, max_cone_nr=3000):
        '''
        Visualize the normal corresponding to each point as a cone

        Inputs:
            pointSet: list of points
            sizeref: Is used for modifying the cone size. Attention, a plotly internal factor that is not accessible
                        is multiplied with this factor to define the cone size. If any two points are very close, this
                        internal factor is very small and it may happen that cones become so small that they are no more
                        visible. If no cones are visible, try e.g. sizeref=100.

                        Tipp: Set sizeref=1, because internal factor assures that cones are visible and sufficiently
                            small that they are separated.
            jupyter(bool): True if graph shall be shown in jupyter. False if graph shall be shown in browser.
            auto_open(bool): True if graph shall be opened automatically. Has no effect if jupyter is True.
            max_cone_nr(int): Helpful if cones are not displayed due to plotly internal scaling.
        '''

        if jupyter == True and auto_open == True:
            print('Function visualize: Parameter auto_open has no effect in jupyter notebooks.')

        if len(pointSet) > max_cone_nr:
            print('Function displayNormals: There are more than '+str(max_cone_nr) +' points.\nExperiments showed that due to '
                  'plotly internal cone scaling a big number of cones \nmay result in invisible scaling. Chose '
                  +str(max_cone_nr) + ' points randomly.')
            pointSet = random.sample(pointSet, max_cone_nr)

        'Conversion to shape (number,) required by cone plot'
        x = y = z = u = v = w = []
        for idx in range(len(pointSet)):
            x = np.append(x, pointSet[idx].getCoordinates()[0])
            y = np.append(y, pointSet[idx].getCoordinates()[1])
            z = np.append(z, pointSet[idx].getCoordinates()[2])
            u = np.append(u, pointSet[idx].getNormal()[0])
            v = np.append(v, pointSet[idx].getNormal()[1])
            w = np.append(w, pointSet[idx].getNormal()[2])

        data = [{
                "type": "cone",
                "x": x,
                "y": y,
                "z": z,
                "u": u,
                "v": v,
                "w": w,
                "sizemode": "absolute",
                "sizeref": sizeref,
                "anchor": "tail",
                "showscale": False,
                'colorscale': [
                    # Let all cones have blue color rgb(0, 0, 255)
                    [0, 'rgb(0, 0 , 255)'],
                    [1, 'rgb(0, 0 , 255)']]
                #"colorbar": {
                #        "x": 0,
                #        "xanchor": "right",
                #        "side": "left"
                #        }
                }]

        layout = {
                "scene": {
                        "aspectmode": 'data'#,
                        #"domain": {"x": [0, 1]},
                        #"camera": {
                        #        "eye": {"x": 1.25, "y": 1.25, "z": 1.25}
                        #        }
                        }
                }

        fig = {"data": data, "layout": layout}

        path = os.path.join(self.output_path, self.name + '_Normals.html')

        if jupyter == True:
            py.iplot(fig, filename=path)
            'store plot'
            py.plot(fig, filename=path, auto_open=False)
        else:
            py.plot(fig, filename=path, auto_open=auto_open)

    def displayPointCloud(self, pointSet, jupyter=False, auto_open=False):
        """
        This function is used to display a point cloud in one colour.

        Inputs: pointSet: list of points
                jupyter(bool): True if graph shall be shown in jupyter. False if graph shall be shown in browser.
                auto_open(bool): True if graph shall be opened automatically. Has no effect if jupyter is True.
        """

        if jupyter == True and auto_open == True:
            print('Function visualize: Parameter auto_open has no effect in jupyter notebooks.')

        x = y = z = []

        for idx in range(len(pointSet)):
            x = np.append(x, pointSet[idx].getCoordinates()[0])
            y = np.append(y, pointSet[idx].getCoordinates()[1])
            z = np.append(z, pointSet[idx].getCoordinates()[2])


        trace = go.Scatter3d(
            x = x,
            y = y,
            z = z,
            mode = 'markers',
            marker = dict(
                symbol='circle',
                size=2,
                line = dict(
                    width = 0,
                    color = '#404040')
            )
        )
        data = [trace]

        layout = go.Layout(
                    title='Point Cloud Scatter Plot',
                    scene = dict(aspectmode='data'),
                    width=1000,
                    margin=dict(
                    r=20, l=10,
                    b=10, t=30))

        fig = go.Figure(data=data, layout=layout)

        path = os.path.join(self.output_path, self.name + '_Scatter.html')

        if jupyter == True:
            py.iplot(fig, filename=path)
            'store plot'
            py.plot(fig, filename=path, auto_open=False)
        else:
            py.plot(fig, filename=path, auto_open=auto_open)

    def displayPointCloudSegmented(self, clusters, include_outlier_list=False, jupyter=False, auto_open=False):
        """
                This method displays the given clusters in different colours.

                Inputs: clusters: list of clusters containing points that were found during segmentation
                        include_outlier_list(bool): Parameter that can be set to True in order to display all segments which are
                            smaller than the minimum cluster size in the same plot as the other clusters (black color is used)
                        jupyter(bool): True if graph shall be shown in jupyter. False if graph shall be shown in browser.
                        auto_open(bool): True if graph shall be opened automatically. Has no effect if jupyter is True.
        """

        if jupyter == True and auto_open == True:
            print('Function visualize: Parameter auto_open has no effect in jupyter notebooks.')

        data = []

        for idx in range(len(clusters)-1):
            x = y = z = []

            for i in range(len(clusters[idx])):

                x = np.append(x, clusters[idx][i].getCoordinates()[0])
                y = np.append(y, clusters[idx][i].getCoordinates()[1])
                z = np.append(z, clusters[idx][i].getCoordinates()[2])

            RGB_colour = mapRGB(idx)


            data.append(go.Scatter3d(
                name='cluster #' + str(idx),
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    symbol='circle',
                    color=RGB_colour,
                    size=2,
                    line=dict(
                        width=1,
                        color=RGB_colour)
                )
            ))

        if include_outlier_list == True:
            x = y = z = []
            outlier_list = self.flatten([clusters[-1]])

            for i in range(len(outlier_list)):
                x = np.append(x, outlier_list[i].getCoordinates()[0])
                y = np.append(y, outlier_list[i].getCoordinates()[1])
                z = np.append(z, outlier_list[i].getCoordinates()[2])

            RGB_colour = 'rgb(0, 0, 0)'

            data.append(go.Scatter3d(
                name='Outlier cluster',
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    symbol='circle',
                    color=RGB_colour,
                    size=2,
                    line=dict(
                        width=1,
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

        path = os.path.join(self.output_path, self.name + '_Clustered.html')

        if jupyter == True:
            py.iplot(fig, filename=path)
            'store plot'
            py.plot(fig, filename=path, auto_open=False)
        else:
            py.plot(fig, filename=path, auto_open=auto_open)

    def displayNormalsSegmented(self, clusters, include_outlier_list=False):
        """
        This method displays the normals as well as the cluster information related to each point of the point cloud.

        Inputs: clusters: list of clusters containing points that were found during segmentation
                include_outlier_list: Parameter that can be set to True in order to display all segments which are
                    smaller than the minimum cluster size in the same plot as the other clusters (black color is used)

        Note: This method does not work well for general point clouds. Especially for point clouds with irregular
        grid (e.g. round shapes), the whole set of cones of some clusters may be shown with invisible size.
        This is due to a plotly internal scaling factor that depends on the smallest distance between any two cones of
        the same cluster. This internal scaling factor is not accessible and can also not be switched off.
        For a normalization one would require access to the plotly internal grid representation which is also not
        accessible.

        Official information on the internal scaling factor:
        https://plot.ly/python/reference/#cone-sizeref
            "This factor (computed internally) corresponds to the minimum "time" to travel across two successive
            x/y/z positions at the average velocity of those two successive positions.
            All cones in a given trace use the same factor."
        """

        print('Function displayNormalsSegmented: Attention: This method is not stable due to '
              'plotly implementation, see comment in function header.')

        data = []

        'get the data formatting for each cluster individually and attach to overall data format list'
        for idx in range(len(clusters)-1):
            x = y = z = u = v = w = []

            for i in range(len(clusters[idx])):

                x = np.append(x, clusters[idx][i].getCoordinates()[0])
                y = np.append(y, clusters[idx][i].getCoordinates()[1])
                z = np.append(z, clusters[idx][i].getCoordinates()[2])
                u = np.append(u, clusters[idx][i].getNormal()[0])
                v = np.append(v, clusters[idx][i].getNormal()[1])
                w = np.append(w, clusters[idx][i].getNormal()[2])

            RGB_colour = mapRGB(idx)

            '''
            Notes:
                - x,y,z is coordinate of cone.
                - u,v,w is direction of cone.
                - anchor: tail puts cone end at x,y,z coordinate
                - sizeref: changes cone diameter
                - showscale: True: enable colorbar
            '''

            data.append({
                "type": "cone",
                "x": x,
                "y": y,
                "z": z,
                "u": u,
                "v": v,
                "w": w,
                "sizemode": "absolute",
                "sizeref": 0.8,
                "anchor": "tail",
                "showscale": False,
                'colorscale': [
                    # Let all values of the values have color rgb(0, 0, 0)
                    [0, RGB_colour],
                    [1, RGB_colour]],
                "hovertext": "Belongs to cluster #"+str(idx)
                #'colorbar': {
                #    'tick0': 0,
                #    'dtick': 1}
            })

        'if outlier list shall be displayed, all segments are displayed in black (colour not used for clusters)'
        if include_outlier_list == True:
            x = y = z = u = v = w = []
            outlier_list = self.flatten([clusters[-1]])

            for i in range(len(outlier_list)):
                x = np.append(x, outlier_list[i].getCoordinates()[0])
                y = np.append(y, outlier_list[i].getCoordinates()[1])
                z = np.append(z, outlier_list[i].getCoordinates()[2])
                u = np.append(u, outlier_list[i].getNormal()[0])
                v = np.append(v, outlier_list[i].getNormal()[1])
                w = np.append(w, outlier_list[i].getNormal()[2])

            RGB_colour = 'rgb(0, 0, 0)'

            '''
            Notes:
                - x,y,z is coordinate of cone.
                - u,v,w is direction of cone.
                - anchor: tail puts cone end at x,y,z coordinate
                - sizeref: changes cone diameter
                - showscale: True: enable colorbar
            '''

            data.append({
                "type": "cone",
                "x": x,
                "y": y,
                "z": z,
                "u": u,
                "v": v,
                "w": w,
                "sizemode": "absolute",
                "sizeref": 0.8,
                "anchor": "tail",
                "showscale": False,
                'colorscale': [
                    # Let all values of the values have color rgb(0, 0, 0)
                    [0, RGB_colour],
                    [1, RGB_colour]],
                "hovertext": "Belongs to outlier group"
                # 'colorbar': {
                #    'tick0': 0,
                #    'dtick': 1}
            })



        layout = {
            "scene": {
                #"aspectratio": {"x": 1, "y": 1, "z": 1},
                "aspectmode": 'data',
                #"domain": {"x": [0, 1]},
                "camera": {
                    "eye": {"x": 1.25, "y": 1.25, "z": 1.25}
                }
            }
        }

        fig = {"data": data, "layout": layout}

        path = os.path.join(self.output_path, self.name + '_NormalsClustered.html')

        py.plot(fig, filename=path, validate=False)

    def flatten(self, list_of_lists):
        """
        Helper method to allow recursive flattening of outlierList

        Inputs: list_of_lists: list of lists element with possibly varying length of its subelements
        """

        if list_of_lists == []:
            return list_of_lists
        if isinstance(list_of_lists[0], list):
            return self.flatten(list_of_lists[0]) + self.flatten(list_of_lists[1:])
        return list_of_lists[:1] + self.flatten(list_of_lists[1:])