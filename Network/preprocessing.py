import sys
sys.path.append('../')
import plotly.offline as py
import plotly.graph_objs as go

import numpy as np
from search import Search
from utils.mapRGB import mapRGB
from numba import jit

from data import patch_dataset
import torch
from utils.mode import Mode
import os

class Voxel:

    '''Class to store properties of a voxel. Required for plotting.'''

    def __init__(self, centroid_normalized=0, centroid_original=0, edge_length=0, value=0):
        'normalized centroid is calculated with bounding box for voxelization at origin size (3,)'
        self.centroid_normalized = centroid_normalized
        'original centroid is shifted to the centroid of the patch that was voxelized size(3,)'
        self.centroid_original = centroid_original
        self.edge_length = edge_length
        'value is 0 if inactive or 1 if active'
        self.value = value

    def setNormalizedCentroid(self, centroid_normalized):
        self.centroid_normalized = centroid_normalized

    def setOriginalCentroid(self, centroid_original):
        self.centroid_original = centroid_original

    def setEdgeLength(self, edge_length):
        self.edge_length = edge_length

    def setValue(self, value):
        self.value = value

    def getNormalizedCentroid(self):
        return self.centroid_normalized

    def getOriginalCentroid(self):
        return self.centroid_original

    def getEdgeLength(self):
        return self.edge_length

    def getValue(self):
        return self.value


class Point_Cloud_Processing:
    def __init__(self, mode):
        '''
        Inputs:
            mode: Mode.TRAIN, Mode.TEST or Mode.PREDICT
        '''

        'Provisions'
        self.cap_shift = None
        self.voxelization_shift = None

        self.mode = mode

    def normalize_point_cloud(self, point_cloud_coordinates):
        """
        Normalizes a point cloud in the range [0, 1] for each dimension

        Inputs:
            point_cloud_coordinates(numpy array(<nrPoints>, 3)): list of coordinates in the given point cloud

        Outputs:
            point_cloud_coordinates(numpy array(<nrPoints>, 3)): list of coordinates in range [0, 1]
            normalization_factor(float): scaling used to normalize the point cloud
        """

        'align point cloud with coordinate system by shifting the minimum in each dimension to the coordinate axes'
        for i in range(3):
            point_cloud_coordinates[:, i] = point_cloud_coordinates[:, i] - np.min(point_cloud_coordinates[:, i])

        'normalization via maximum of the 3 dimensions'
        normalization_factor = np.max([np.max(point_cloud_coordinates[:, 0]),
                                    np.max(point_cloud_coordinates[:, 1]),
                                    np.max(point_cloud_coordinates[:, 2])])
        point_cloud_coordinates = point_cloud_coordinates / normalization_factor

        return point_cloud_coordinates, normalization_factor

    def build_patches(self, point_cloud, k, cap_shift=None):
        """
        This function builds local patches from a point cloud.
        Local points are chosen randomly. Then their nearest neighbours are found in a range that is limited by a cube.
        Patches are returned in normalized range [0, 1].

        Inputs:     point_cloud(tuple): Coordinates in the given point cloud and corresponding segment ID
                    k(int): Parameter for k nearest neighbours search
                    cap_shift(float): parameter for the left/right boundaries of a cube in each dimension

        Outputs:   patch_list(tuple): List of patches that were found, corresponding centroid and a
                        segment that was obtained by majority vote. Each patch is a list of points.
                        Indexing by patch_list[0,0] returns all coordinates of the first patch.
                    normalization_factor(float): factor used to normalize the point cloud
        """

        'Seed can be used to make building patches deterministic.'
        #np.random.seed(5)

        self.cap_shift = np.array([[cap_shift, cap_shift, cap_shift]])

        point_cloud_coordinates = [i for i in point_cloud[0]][0].tolist()

        if self.mode != Mode.PREDICT:
            point_cloud_segments = [i for i in point_cloud[1]][0].tolist()

        point_cloud_coordinates, normalization_factor = self.normalize_point_cloud(np.array(point_cloud_coordinates))
        point_cloud_coordinates = point_cloud_coordinates.tolist()

        patch_list = []
        search_obj = Search(np.array(point_cloud_coordinates))
        available_indices = list(range(0, len(point_cloud_coordinates)))
        'loop until each point is assigned to a patch'
        while not all(i is None for i in available_indices):
            'choose random point from current point cloud'
            seed = np.random.choice(np.where(np.array(available_indices) != None)[0])
            seed_point = np.array([point_cloud_coordinates[seed]])

            'find all k nearest neighbours to the current seed point. Returned indices are allocated to points.'
            point_neighbours_indices = search_obj.nearestKSearch(seed_point, k)
            point_neighbours = np.array([point_cloud_coordinates[i] for i in point_neighbours_indices])

            'cap all points that are outside cube boundary'
            patch, indices = self.cube_capping(seed_point, point_neighbours, point_neighbours_indices)
            indices = np.sort(indices)

            'compute centroid of patch'
            centroid = self.compute_3D_centroid(patch)

            if self.mode != Mode.PREDICT:
                'Segment ID of patch is the segment that is most frequent among points'
                segment_nr = np.bincount(np.array([point_cloud_segments[i] for i in indices])).argmax()

            'remove all patch points from original point cloud'
            for i in range(len(indices)-1, -1, -1):
                available_indices[indices[i]] = None

            if self.mode != Mode.PREDICT:
                patch_list.append(tuple((patch, centroid, segment_nr)))
            else:
                patch_list.append(tuple((patch, centroid)))

        return np.array(patch_list), normalization_factor

    def cube_capping(self, seed_point, patch, indices):
        """
        Remove all points from the list patch which are outside a boundary defined by a cube with origin at the
        coordinates of the seed point.

        Inputs:     seed_point(numpy array(3,)): Coordinates of seed point
                    patch(numpy array (<nrPoints>, 3)): Point coordinates in the patch that were found by KNN
                    indices(numpy list (<nrPoints>,)): List of corresponding indices to the points in the patch
        Outputs:    patch(numpy array (<nrPoints>, 3)): Point coordinates where the points
                        outside of the cube were removed
                    indices(numpy list (<nrPoints>,)): List of indices where the indices of points outside of the cube were removed
        """

        'make point a row vector'
        seed_point = np.reshape(seed_point, (1, 3))

        'convert patch and indices to python list, because deleting from numpy array was buggy'
        patch = patch.tolist()
        indices = indices.tolist()

        'iterate through patch list in reversed order to avoid indexing issue while removing elements'
        for point_idx in range(len(patch)-1, -1, -1):
            pointCoordinates = np.array([patch[point_idx]])
            shift_from_seed = np.abs(seed_point - pointCoordinates)
            'if any of the coordinates exceeds the cube boundary remove the point from the list'
            if np.any(np.greater_equal(shift_from_seed, self.cap_shift)):
                del patch[point_idx]
                del indices[point_idx]

        if np.any(np.abs(seed_point - np.array(patch)) > 0.1):
            print('preprocessing: Attention. Some point is outside neighbourhood boundaries.')

        return np.array(patch), np.array(indices)

    def compute_3D_centroid(self, pointSet):
        '''
        Computes the 3D (X,Y,Z) Centroid of a set of points given as
        a 2D-tensor

        Inputs:     pointSet as np.array
                        with size (length,3) where each row is [x,y,z]
        Outputs:    centroid [x,y,z] as vector with shape (3,1)
                        This is the geometric center of the pointSet
        '''

        'Initialisation as 0'
        centroid = np.zeros((3, 1))

        'sum the x/y/z-values for all points'
        for idx in range(pointSet.shape[0]):
            centroid[0] = centroid[0]+pointSet[idx][0]
            centroid[1] = centroid[1]+pointSet[idx][1]
            centroid[2] = centroid[2]+pointSet[idx][2]

        'division by number of points to get an average'
        centroid = centroid/pointSet.shape[0]

        return centroid

    #@jit
    def voxelize_patches(self, patches, nr_voxels_per_dim, visualize=False):
        """
        Voxelize a list of patches.

        Input:
            patches(tuple): List of tuples of patch and corresponding segment ID
            nr_voxels_per_dim(int): Number of voxels inside the voxelization box
                        in each dimension -> space will be voxeled in nr_voxels_per_dim^3
            visualize(bool): Set to true, if patches shall be displayed. Normally not the case, so
                        this variable shall save time during network execution.
        Output:
            voxelized_patch_objects_list(list of tuples): List of tuples of voxelized patches containing voxel objects,
                centroid of original patch and corresponding segment ID. This allows to draw the voxels.
            voxelized_patch_values_list(list of tuples): List of tuples of voxelized patches containing only activation
                values, centroid of original patch and corresponding segment ID. This avoids another loop to extract
                activation values from voxel objects.
        """

        'if only one patch is handed over for voxelization'
        if len(patches.shape) == 1:
            patches = np.expand_dims(patches, axis=0)

        'use odd number of voxels per dim, so left and right from the origin there is an even number of voxels'
        if nr_voxels_per_dim % 2 == 0:
            nr_voxels_per_dim += 1
            print('Use odd number of voxels, so there is even number of voxels left/right from origin. Increased ' +
                  'number of voxels per dimension to ' + str(nr_voxels_per_dim))

        'voxelization boundaries (1,3)'
        self.voxelization_shift = self.cap_shift  # for simplicity use same bounding box

        'determine voxel size'
        volume_voxelization_box = 2 * self.voxelization_shift[0][0] * \
                                  2 * self.voxelization_shift[0][1] * 2 * self.voxelization_shift[0][2]
        voxel_volume = volume_voxelization_box / nr_voxels_per_dim ** 3

        'voxel shift is half the cubic root of its volume in 3 dimensions. Bounding box may be rectangular, but all' \
        'voxels are cubes, so they may not fill out the whole bounding box'
        voxel_shift = np.array([[np.cbrt(voxel_volume) / 2, np.cbrt(voxel_volume) / 2, np.cbrt(voxel_volume) / 2]])

        'calculate voxel centroids'
        voxel_block_centroids = []
        if visualize == True: #not required, if voxels shall not be displayed
            rangeIterator = int(np.floor(nr_voxels_per_dim / 2))
            for x in range(-rangeIterator, rangeIterator + 1):
                for y in range(-rangeIterator, rangeIterator + 1):
                    for z in range(-rangeIterator, rangeIterator + 1):
                        voxel_block_centroids.append(np.array([x, y, z]) * 2 * voxel_shift)

            'shape(<NrPoints>,3)'
            voxel_block_centroids = np.squeeze(np.array(voxel_block_centroids), axis=1)


        voxelized_patch_objects_list = []
        voxelized_patch_values_list = []
        for patch_ID in range(patches.shape[0]):
            voxelized_patch_objects, voxelized_patch_values = \
                self.voxelize_patch(patches[patch_ID][0], nr_voxels_per_dim, voxel_shift,
                                    voxel_block_centroids, visualize)
            if self.mode != Mode.PREDICT:
                voxelized_patch_objects_list.append(tuple((voxelized_patch_objects,
                                                          patches[patch_ID][1], patches[patch_ID][2])))
                voxelized_patch_values_list.append(tuple((voxelized_patch_values,
                                                           patches[patch_ID][1], patches[patch_ID][2])))
            else:
                'segments not available'
                voxelized_patch_objects_list.append(tuple((voxelized_patch_objects,
                                                           patches[patch_ID][1])))
                voxelized_patch_values_list.append(tuple((voxelized_patch_values,
                                                          patches[patch_ID][1])))
        #print("Building voxels finished!")
        return np.array(voxelized_patch_objects_list), np.array(voxelized_patch_values_list)

    #@jit
    def voxelize_patch(self, patch, nr_voxels_per_dim, voxel_shift, voxel_block_centroids, visualize=False):
        """
        This function is used to voxelize a patch of points. Patches are voxelized with centroid of patch at
        the centroid of the box of voxels. Box of voxels is a boundary box that contains all voxels.

        Inputs:     patch(numpy array (<nrPoints>,3): Patch of points
                    nr_voxels_per_dim(int): Number of voxels inside the voxelization box
                        in each dimension -> space will be voxeled in nr_voxels_per_dim^3
                    voxel_shift(float): half the length of the whole voxelization box
                    voxel_block_centroids(list): List of voxel centroids in standard ordering
                    visualize(bool): Set to true, if patches shall be displayed. Normally not the case, so
                                        this variable shall save time during network execution.

        Outputs:    voxels(np.array (<nr_voxels_per_dim>, <nr_voxels_per_dim>, <nr_voxels_per_dim>)):
                        numpy array with voxel objects in the standard x,y,z ordering.
                        Example: voxels[0,0,0] is the voxel at the most bottom, most left, most front voxel
                    voxels_block(np.array): voxel block with only activation value in x, y, z ordering

        """

        'Shift points by centroid such that they have their centroid at (x=0,y=0,z=0)'
        patch_centroid = self.compute_3D_centroid(patch)
        shifted_patch = patch - np.tile(np.reshape(patch_centroid, (1, 3)), (patch.shape[0], 1))

        #voxel_block_values = np.zeros(voxel_block_centroids.shape[0])

        voxels_block = np.zeros((nr_voxels_per_dim, nr_voxels_per_dim, nr_voxels_per_dim))
        shift = int(np.floor(nr_voxels_per_dim/2))
        for point_idx in range(shifted_patch.shape[0]):
            x = self.get_voxel_idx(shifted_patch[point_idx][0], voxel_shift[0][0]*2, shift)
            y = self.get_voxel_idx(shifted_patch[point_idx][1], voxel_shift[0][0]*2, shift)
            z = self.get_voxel_idx(shifted_patch[point_idx][2], voxel_shift[0][0]*2, shift)

            voxels_block[x, y, z] = 1
        'Iterate over points in patch. Set voxel to 1 if at least one point is inside voxel. ' \

        """
        KNN method: worked also, but is much slower than method with rounding. Kept for reference.
        'Determine voxel where point is inside via KNN'

        search = Search(voxel_block_centroids)
        for point_idx in range(shifted_patch.shape[0]):
            closest_voxel = search.nearestKSearch(shifted_patch[point_idx], 1)
            voxel_block_values[closest_voxel] = 1
        """

        'assign information to voxel objects'
        voxels_array = np.empty((nr_voxels_per_dim, nr_voxels_per_dim, nr_voxels_per_dim), dtype=Voxel)
        #voxels_block = np.zeros((nr_voxels_per_dim, nr_voxels_per_dim, nr_voxels_per_dim))

        'use visualize variable to save time during network execution. Use visualize, if patches shall be displayed'
        if visualize == True:
            i = 0
            for x in range(0, nr_voxels_per_dim):
                for y in range(0, nr_voxels_per_dim):
                    for z in range(0, nr_voxels_per_dim):
                        voxels_array[x, y, z] = Voxel(centroid_normalized=voxel_block_centroids[i],
                                                centroid_original=voxel_block_centroids[i]+np.reshape(patch_centroid, 3),
                                                edge_length=voxel_shift[0][0]*2,
                                                #value=int(voxel_block_values[i]))
                                                value=voxels_block[x, y, z])
                        #voxels_block[x, y, z] = voxel_block_values[i]
                        i += 1

        return voxels_array, voxels_block

    @jit
    def build_relation_matrix(self, patches):
        """
        Build a reference matrix that indicates which patches belong together.

        Inputs:
            patches(tuple): Tuple(patches, centroid, segmentID). May also be voxels with same shape
                0 means no relation. 1 means patches belong to same segment.
        """

        patches_range = patches.shape[0]
        'Initialize to all zeros'
        reference_matrix = np.zeros((patches_range, patches_range))

        for i in range(patches_range):
            for j in range(patches_range):
                if patches[i, 2] == patches[j, 2]:
                    reference_matrix[i, j] = 1

        return reference_matrix

    def get_voxel_parameters(self, point, edge_length):
        """
        Convert a point to arrays of vertices and triangular faces that represent the cube. This is required for
        visualization with Mesh3D.

        Inputs:     point(1,3): Coordinates of the point that shall be converted into a voxel
                    edge_length(float): length of the edges of a voxel

        Outputs:    x,y,z: lists with 8 elements: a cube has 8 vertices. Is used as x,y,z in Mesh3D
                    i,j,k: lists with 12 elements: Vertex indices of the triangles. Is used as i,j,k in Mesh3D
        """

        if np.isscalar(edge_length):
            'half edge length for shift left/right from origin and size of voxel'
            half_el = edge_length / 2
        else:
            raise TypeError('Function: get_voxel_parameters: edge_length must be a scalar')


        'cube vertex coordinates'
        x = []
        y = []
        z = []
        for vertex_idx in range(8):
            x.append(point[0] + (vertex_idx // 4)*2*half_el-half_el)
            y.append(point[1] + (vertex_idx // 2 % 2)*2*half_el-half_el)
            z.append(point[2] + (vertex_idx % 2)*2*half_el-half_el)

        'indices of coordinates for triangle vertices'
        i = [0, 1, 0, 1, 0, 6, 2, 7, 4, 7, 1, 7]
        j = [1, 2, 1, 4, 2, 2, 3, 3, 5, 5, 3, 3]
        k = [2, 3, 4, 5, 4, 4, 6, 6, 6, 6, 5, 5]

        return x, y, z, i, j, k

    def visualize_voxels(self, voxels, jupyter=False, auto_open=False):
        """
        This function is used to visualize voxels.

        Inputs:     voxels(np.array(<voxels_per_dim>,<voxels_per_dim>, <voxels_per_dim>)):
                        Array with voxels in standard x,y,z ordering
                    jupyter(bool): Input whether function shall show patch in jupyter or in browser
                    auto_open(bool): Whether to open or not the plotly plot. Set to False if on remote machine to
                        prevent error.
        """
        
        if jupyter == True and auto_open == True:
            print('Function visualize_voxels: Parameter auto_open has no effect in jupyter notebooks.')
        
        'get the parameters to draw the voxel cubes and append to voxel meshes'
        i=0
        voxel_meshes = []
        for x in range(voxels.shape[0]):
            for y in range(voxels.shape[1]):
                for z in range(voxels.shape[2]):
                    curr_voxel = voxels[x, y, z]
                    if curr_voxel.getValue() == 1:
                        voxel_params = self.get_voxel_parameters(curr_voxel.getNormalizedCentroid(),
                                                                curr_voxel.getEdgeLength())
                        voxel_meshes.append(go.Mesh3d(
                            x=voxel_params[0],
                            y=voxel_params[1],
                            z=voxel_params[2],
                            i=voxel_params[3],
                            j=voxel_params[4],
                            k=voxel_params[5],
                            color='black'
                        ))
                    # else:
                    #     'Uncomment to see the whole voxel box'
                    #     voxel_params = self.get_voxel_parameters(curr_voxel.getNormalizedCentroid(),
                    #                                              curr_voxel.getEdgeLength())
                    #     voxel_meshes.append(go.Mesh3d(
                    #         x=voxel_params[0],
                    #         y=voxel_params[1],
                    #         z=voxel_params[2],
                    #         i=voxel_params[3],
                    #         j=voxel_params[4],
                    #         k=voxel_params[5],
                    #         color='gray'
                    #     ))

        data = voxel_meshes

        layout = go.Layout(
            scene=dict(
                aspectmode='cube',
                xaxis=dict(
                    type='linear',
                    range=[-self.voxelization_shift[0][0], self.voxelization_shift[0][0]],
                    showgrid=True,
                    gridcolor='black',
                    showticklabels=False,
                    title=''),
                yaxis=dict(
                    type='linear',
                    range=[-self.voxelization_shift[0][1], self.voxelization_shift[0][0]],
                    showgrid=True,
                    gridcolor='black',
                    showticklabels=False,
                    title=''),
                zaxis=dict(
                    type='linear',
                    range=[-self.voxelization_shift[0][2], self.voxelization_shift[0][2]],
                    showgrid=True,
                    gridcolor='black',
                    showticklabels=False,
                    title=''),
            ),
        )
        fig = go.Figure(data=data, layout=layout)

        path = os.path.join('data/temp_data/', 'voxels.html')
        print('File: preprocessing. Stored to default temporary path.')

        if jupyter == True:
            py.iplot(fig, filename=path)
            'store plot'
            py.plot(fig, filename=path, auto_open=False)
        else:
            py.plot(fig, filename=path, auto_open=auto_open)

    def visualize_patches(self, patch_list, jupyter=False, auto_open=False):
        """
        Visualizes given patches using plotly. The points inside a patch are coloured in the same colour,
        while points in other patches are coloured in a different colour.
        Shape of patch_list must be (<patch>, <points in patch>, 3)

        Inputs:     patch_list(np.array (<nrPatches>,<nrPointsPerPatch>,3): Array containing the patches
                    jupyter(bool): Input whether function shall show patch in jupyter or in browser
                    auto_open(bool): Whether to open or not the plotly plot. Set to False if on remote machine to
                        prevent error.
        """
        
        if jupyter == True and auto_open == True:
            print('Function visualize_patches: Parameter auto_open has no effect in jupyter notebooks.')
        
        data = []
        for patch_id in range(patch_list.shape[0]):
            x = y = z = []
            for idx in range(patch_list[patch_id].shape[0]):
                x = np.append(x, patch_list[patch_id][idx][0])
                y = np.append(y, patch_list[patch_id][idx][1])
                z = np.append(z, patch_list[patch_id][idx][2])

            RGB_colour = mapRGB(patch_id)

            data.append(go.Scatter3d(
                name='patch #' + str(patch_id),
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

        path = os.path.join('data/temp_data/', 'patch_visualization.html')
        print('File: preprocessing. Stored to default temporary path.')

        if jupyter == True:
            py.iplot(fig, filename=path)
            'store plot'
            py.plot(fig, filename=path, auto_open=False)
        else:
            py.plot(fig, filename=path, auto_open=auto_open)

    def apply_preprocessing(self, point_cloud, nr_voxels_per_dim, batch_size=10, build_ref_matrix=False):
        """
        Compute preprocessing steps before network application

        Inputs:
            point_cloud()
            nr_voxels_per_dim (int): nr voxels per dimension
            batch_size(int): batch size that shall be used when getting voxelized patches from the patch dataset
            build_ref_matrix(bool): True, if a reference matrix shall be returned.
        """

        'extract patches. Returns tuple (patch, centroid, segment_ID).'
        k = int(point_cloud[0].shape[1] / 50)  # k shall be such that all points would fit into 50 patches
        cap_shift = 0.1  # (1/cap_shift)^3 << 1000 to avoid too many patches
        patches, normalization_factor = self.build_patches(point_cloud, k=k, cap_shift=cap_shift)

        'voxelize patches. Returns tuple (voxelized_patch_values, centroid, segment_ID)'
        _, voxelized_patches = self.voxelize_patches(patches, nr_voxels_per_dim)

        if build_ref_matrix == True and self.mode != Mode.PREDICT:
            reference_matrix = self.build_relation_matrix(voxelized_patches)
        else:
            reference_matrix = None

        if self.mode == Mode.TRAIN:
            shuffle_mode = True
        else:
            shuffle_mode = False

        patch_data = patch_dataset.Patch_dataset(patch_list=voxelized_patches, mode=self.mode)
        patch_loader = torch.utils.data.DataLoader(dataset=patch_data,
                                               batch_size=batch_size,
                                               shuffle=shuffle_mode, num_workers=4)
        return patch_loader, reference_matrix, patches, voxelized_patches, normalization_factor

    def get_voxel_idx(self, coordinate, voxel_length, shift):
        """
        Get the index of one coordinate in the voxelization box.

        Inputs:
            coordinate(float): one of three coordiantes of a point in a patch
            voxel_length(float): the total length of one side of a voxel
            shift(int): Half the number of voxels per dimension. To shift the index into the range [0, nr_voxels_per_dim]
        """
        index = np.round(coordinate / voxel_length) + shift
        'value exactly at voxelization box boundaries would be assigned outside the boundary'
        if index < 0:
            index = 0
        if index > 2*shift:
            index = 2*shift
        return int(index)

