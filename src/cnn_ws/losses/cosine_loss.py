import torch
import torch.nn as nn


class CosineLoss(nn.Module):

    def __init__(self, size_average=True, use_sigmoid=False):
        super(CosineLoss, self).__init__()
        self.averaging = size_average
        self.use_sigmoid = use_sigmoid

    def forward(self, input_var, target_var):
        '''
            Cosine loss:
            1.0 - (y.x / |y|*|x|)
        '''
        if self.use_sigmoid:
            loss_val = sum(1.0 - nn.functional.cosine_similarity(torch.sigmoid(input_var), target_var))
        else:
            loss_val = sum(1.0 - nn.functional.cosine_similarity(input_var, target_var))
        if self.averaging:
            loss_val = loss_val/input_var.data.shape[0]
        return loss_val
