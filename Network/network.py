import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import numpy as np


class Network(nn.Module):
    def __init__(self, model, n_channels, n_classes, nr_voxels_per_dim, device):
        """
        Upper network class. Allows to combine different convolutional systems with a fully connected system

        Inputs:
            model(str): name of convolutional model [alexnet, resnet]
            n_channels (int): Number of input channels
            n_classes (int): Number of output classes
            nr_voxels_per_dim (int): Number of voxels in each dimension
            device: Device to run the network on
        """

        super(Network, self).__init__()

        self.model = model
        self.shift = None #shift between two patches
        self.device = device

        if self.model == 'alexnet':
            print('network.py: Attention, this AlexNet variant is not maintained.')
            n_feature_maps = 32         #nr of feature maps before entry into dense layer
            self.convNet = _3D_AlexNet(n_channels, nr_voxels_per_dim, feature_maps_out=n_feature_maps)
        elif self.model == 'resnet':
            n_feature_maps = 128
            self.convNet = ResNet_3D(n_channels, nr_voxels_per_dim,
                                     feature_maps_out=n_feature_maps, dense_out=n_feature_maps)
        else:
            raise Exception('network.py: Unknown model name')

        'Number of input neurons to fully connected layer results from' \
        'two patches with nr_voxels_per_dim in each dimension, plus 3 coordinates from the centroid'
        fcl_inputs = int(2*np.prod(self.convNet.c_d_h_w))
        
        self.fcl = Fully_Connected(fcl_inputs, n_classes)

    def forward(self, input1, input2, shift):
        """
        Inputs:
            input1, input2(tensor): Two voxel blocks
            shift(tensor, float): Shift between two patches
        Outputs:
            output_fcl(tensor, size 2): First element is probability that patches do not belong together.
                Second element is probability that patches belong together
        """
        self.shift = shift
        output_conv1 = self.convNet(input1)
        output_conv2 = self.convNet(input2)
        
        input_fcl = torch.cat((torch.flatten(output_conv1, start_dim=1),
                               torch.flatten(output_conv2, start_dim=1)), dim=1)
        
        
        output_fcl = self.fcl(input_fcl, self.shift)

        return output_fcl


class _3D_AlexNet(nn.Module):
    def __init__(self, n_channels, nr_voxels_per_dim, feature_maps_out):
        """
        Variant of convolutional part of AlexNet

        Inputs:
            n_channels (int): Number of input channels
            nr_voxels_per_dim (int): Number of voxels per dimension
            feature_maps_out (int): Number of output feature maps
        """

        super(_3D_AlexNet, self).__init__()
        self.c_d_h_w = 0
        d_h_w = [nr_voxels_per_dim, nr_voxels_per_dim, nr_voxels_per_dim]

        'Parameters internal to network'
        feature_maps = [16, 32, 64, 32, 32]

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=n_channels, out_channels=feature_maps[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(feature_maps[0]),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(kernel_size=2, stride=1),
            nn.Conv3d(in_channels=feature_maps[0], out_channels=feature_maps[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(feature_maps[1]),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(kernel_size=2, stride=1),
            nn.Conv3d(in_channels=feature_maps[1], out_channels=feature_maps[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(feature_maps[2]),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(kernel_size=2, stride=1),
            nn.Conv3d(in_channels=feature_maps[2], out_channels=feature_maps[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(feature_maps[3]),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(kernel_size=2, stride=1),
            nn.Conv3d(in_channels=feature_maps[3], out_channels=feature_maps[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(feature_maps[4]),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(kernel_size=2, stride=1),
            nn.Conv3d(in_channels=feature_maps[4], out_channels=feature_maps_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(feature_maps_out),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        'calculate depth, height, width after each channel'
        d_h_w = conv_output_shape(d_h_w, kernel_size=3, stride=1, pad=1)
        d_h_w = maxpool3D_output_shape(d_h_w, 2, stride=1)
        d_h_w = conv_output_shape(d_h_w, kernel_size=3, stride=1, pad=1)
        d_h_w = maxpool3D_output_shape(d_h_w, kernel_size=2, stride=1)
        d_h_w = conv_output_shape(d_h_w, kernel_size=3, stride=1, pad=1)
        d_h_w = maxpool3D_output_shape(d_h_w, kernel_size=2, stride=1)
        d_h_w = conv_output_shape(d_h_w, kernel_size=3, stride=1, pad=1)
        d_h_w = maxpool3D_output_shape(d_h_w, kernel_size=2, stride=1)
        d_h_w = conv_output_shape(d_h_w, kernel_size=3, stride=1, pad=1)
        d_h_w = maxpool3D_output_shape(d_h_w, kernel_size=2, stride=1)
        d_h_w = conv_output_shape(d_h_w, kernel_size=3, stride=1, pad=1)
        d_h_w = maxpool3D_output_shape(d_h_w, kernel_size=2, stride=2)
        # d_h_w = conv_output_shape(d_h_w, kernel_size=1, stride=1)
        self.c_d_h_w = np.append([feature_maps_out], np.array(d_h_w))

    def forward(self, input):
        output = self.conv(input)
        return output


class Fully_Connected(nn.Module):
    def __init__(self, in_features, num_outputs):
        """
        Inputs:
            in_features (int): number of input neurons
            num_outputs (int): number of output neurons
        """
        super(Fully_Connected, self).__init__()
        'Initialize the fully connected layer'

        out_features = [30, 30, 30, 20]

        self.fcl = nn.Sequential(
            #nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=in_features, out_features=out_features[0], bias=True),
            nn.ReLU(inplace=False),
            #nn.Dropout(p=0.5, inplace=False),
        )

        self.fcl.apply(weights_init)

        self.fcl2 = nn.Sequential(
            nn.Linear(in_features=out_features[0] + 3, out_features=out_features[1], bias=True),
            nn.ReLU(inplace=False),
            #nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=out_features[1], out_features=out_features[2], bias=True),
            nn.ReLU(inplace=False),
            #nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=out_features[2], out_features=out_features[3], bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=out_features[3], out_features=num_outputs, bias=True)
        )
        self.fcl2.apply(weights_init)

    def forward(self, input, shift):
        output = self.fcl(input)

        output = self.fcl2(torch.cat((output, torch.flatten(shift, start_dim=1)), dim=1))
        return output


class ResNet_3D(nn.Module):

    def __init__(self, n_channels, nr_voxels_per_dim, feature_maps_out, dense_out):
        """
        ResNet implementation.

        Inputs:
            n_channels (int): Number of input channels
            nr_voxels_per_dim (int): Number of voxels per dimension
            feature_maps_out (int): Number of output feature maps
            dense_out (int): Number of output neurons in dense output layer
        """

        super(ResNet_3D, self).__init__()

        'intermediate feature maps'
        feature_maps = [8, 8, 16, 32, 64, feature_maps_out]
        conv_kernel_size = [3, 3, 3, 3, 3, 3]
        conv_stride = [1, 1, 2, 2, 2, 2]
        conv_pad = []
        for size_idx in range(len(conv_kernel_size)):
            conv_pad.append(int(np.floor(conv_kernel_size[size_idx]/2 * conv_stride[size_idx])))


        pool_size = 3
        pool_stride = 1
        pool_pad = int(np.floor(pool_size/2 * pool_stride))

        'calculation of sizes. only relevant, if global average pooling is not used.'
        d_h_w = [nr_voxels_per_dim, nr_voxels_per_dim, nr_voxels_per_dim]
        d_h_w = conv_output_shape(d_h_w, kernel_size=conv_kernel_size[0], stride=conv_stride[0], pad=conv_pad[0])
        d_h_w = maxpool3D_output_shape(d_h_w, kernel_size=pool_size, stride=pool_stride, pad=pool_pad)

        for i in range(1, len(conv_kernel_size)):
            pad_1st, pad_2nd = compute_resnet_padding(kernel_size=conv_kernel_size[i], stride=conv_stride[i])
            d_h_w = resblock_output_shape(d_h_w, kernel_size=conv_kernel_size[i], stride=conv_stride[i],
                                          pad_1st_conv=pad_1st, pad_2nd_conv=pad_2nd)

        'cdhw after last Residual layer'
        c_d_h_w = np.append([feature_maps_out], np.array(d_h_w))
        'convert cdhw to nr of input neurons of dense layer. For avg pooling, nr of feature maps corresponds to FCL input.'
        #dense_in = np.prod(c_d_h_w)
        dense_in = feature_maps[5] #as global average pooling returns one output per feature map
        
        self.resnet = nn.Sequential(
            nn.Conv3d(in_channels=n_channels, out_channels=feature_maps[0],
                      kernel_size=conv_kernel_size[0], stride=conv_stride[0], padding=conv_pad[0]),
            nn.BatchNorm3d(num_features=feature_maps[0]),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(kernel_size=pool_size, stride=pool_stride, padding=pool_pad),

            ResBlock(n_channels_in=feature_maps[0], n_channels_out=feature_maps[1],
                     kernel_size=conv_kernel_size[1], stride=conv_stride[1]),
            ResBlock(n_channels_in=feature_maps[1], n_channels_out=feature_maps[2],
                     kernel_size=conv_kernel_size[2], stride=conv_stride[2]),
            ResBlock(n_channels_in=feature_maps[2], n_channels_out=feature_maps[3],
                     kernel_size=conv_kernel_size[3], stride=conv_stride[3]),
            ResBlock(n_channels_in=feature_maps[3], n_channels_out=feature_maps[4],
                     kernel_size=conv_kernel_size[4], stride=conv_stride[4]),
            ResBlock(n_channels_in=feature_maps[4], n_channels_out=feature_maps[5],
                     kernel_size=conv_kernel_size[5], stride=conv_stride[5]),
            
            nn.AdaptiveAvgPool3d((1, 1, 1))

        )
        self.fcl = nn.Sequential(
            nn.Linear(in_features=dense_in, out_features=dense_out, bias=True),
            nn.ReLU(inplace=False),
        )
        
        self.c_d_h_w = dense_out
        
        self.resnet.apply(weights_init)
        self.fcl.apply(weights_init)
        

    def forward(self, input):
        output = self.resnet(input)
        output = self.fcl(torch.flatten(output, start_dim=1))
        
        return output


class ResBlock(nn.Module):

    def __init__(self, n_channels_in, n_channels_out, kernel_size, stride):
        """
        Basic residual Block of a Residual Network. Returns addition between skip connection and
        two sequential convolutional systems. Needed to reduce height/width/depth to allow bigger channel size.

        Inputs:
            n_channels (int): Number of input channels
            n_channels_out (int): Number of output channels
            kernel_size (int): size of kernel in 3 dimensions
            stride (int): stride length of kernel
        """

        super(ResBlock, self).__init__()

        'Parameters'
        stride_skip = 2

        if kernel_size % 2 != 1:
            kernel_size = kernel_size - 1
            print('ResBlock: Only uneven kernel size allowed. Changed ' + \
            'kernel_size to ' + kernel_size + '.')

        pad_skip = 2 #valid for stride_skip = 2
        pad_first_conv, pad_sec_conv = compute_resnet_padding(kernel_size, stride)

        self.skip = 0
        if n_channels_in != n_channels_out:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels=n_channels_in, out_channels=n_channels_out, kernel_size=1,
                          stride=stride_skip, padding=pad_skip),
                nn.BatchNorm3d(n_channels_out)
                )

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=n_channels_in, out_channels=n_channels_out, kernel_size=kernel_size,
                      stride=stride, padding=pad_first_conv),
            nn.BatchNorm3d(n_channels_out),
            nn.ReLU(inplace=False),
            nn.Conv3d(in_channels=n_channels_out, out_channels=n_channels_out, kernel_size=kernel_size,
                      stride=1, padding=pad_sec_conv),
            nn.BatchNorm3d(n_channels_out),
        )

    def forward(self, input):
        conv_output = self.conv(input)
        if self.skip == 0:
            #return conv_output + input
            return nn.ReLU(inplace=False)(conv_output + input)
        else:
            #return conv_output + self.skip(input)
            return nn.ReLU(inplace=False)(conv_output + self.skip(input))

def resblock_output_shape(d_h_w, kernel_size=1, stride=1, pad_1st_conv=0, pad_2nd_conv=0, dilation=1):
    """
    Calculate the depth, height and width after resblock application
    """

    d_h_w = conv_output_shape(d_h_w, kernel_size=kernel_size, stride=stride, pad=pad_1st_conv)
    d_h_w = conv_output_shape(d_h_w, kernel_size=kernel_size, stride=1, pad=pad_2nd_conv)

    return d_h_w

def conv_output_shape(d_h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Calculate the depth, height and width after conv3D application
    """
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size, kernel_size)
    d = floor(((d_h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1)/stride) + 1)
    h = floor(((d_h_w[1] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1)/stride) + 1)
    w = floor(((d_h_w[2] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1)/stride) + 1)
    return [d, h, w]

def maxpool3D_output_shape(d_h_w, kernel_size=1, stride=None, pad=0, dilation=1):
    """
    Calculate the depth, height and width after maxpool3D application
    """
    from math import floor
    if stride is None:
        stride = kernel_size
    d = floor((d_h_w[0] + 2*pad - dilation * (kernel_size-1) - 1)/stride + 1)
    h = floor((d_h_w[1] + 2*pad - dilation * (kernel_size-1) - 1)/stride + 1)
    w = floor((d_h_w[2] + 2*pad - dilation * (kernel_size-1) - 1)/stride + 1)

    return [d, h, w]

def weights_init(m):
    'initializes the weights of a convolutional or dense layer with He initialization (= Kaiming initialization)'
    if type(m) in [nn.Conv3d, nn.Linear]:
        nn.init.kaiming_uniform_(m.weight)

def compute_resnet_padding(kernel_size, stride):
    'computes padding size of first and second convolutional layer in this resnet variant'

    pad_1st_conv = int(np.floor(kernel_size/2 * stride))
    pad_2nd_conv = int(np.floor(kernel_size/2))

    return pad_1st_conv, pad_2nd_conv