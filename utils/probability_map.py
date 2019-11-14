import torch

def get_probability_map(estimated_relation_matrix):
    'compute softmax function on the output of the network'

    softmax = torch.nn.Softmax(dim=2)
    outputSoftMax = softmax(estimated_relation_matrix)

    return outputSoftMax[:, :, 1]