import torch

def calculate_loss_stats(loss_list, is_train=False, writer=None, steps=0):
    """
    Used to calculate and print statistics on the loss for visualizing training progress.

    Inputs: loss_list(list of floats): List of losses
            is_train(bool): if true: training mode, else: testing
            writer: tensorboard writer object
            steps (int): Number of passed steps
    Outputs:
            average, minimum and maximum loss in loss list
    """

    'conversion to tensor'
    loss_list = torch.stack(loss_list)

    'calculate statistics'
    avg = torch.mean(loss_list)
    maxL = torch.max(loss_list)
    minL = torch.min(loss_list)

    'Print result to tensor board and std. output'
    if is_train:
        mode = 'Train'
    else:
        mode = 'Test'

    'Add average loss value to tensorboard'
    writer.add_scalar(mode + '_Loss', avg, steps)

    return avg.item(), maxL.item(), minL.item()