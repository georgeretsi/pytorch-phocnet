'''avoid saving and loading directly to gpu'''


import torch
from torch.nn.parameter import Parameter
#import torch.nn as nn


def my_torch_save(model, filename):

    # cpu save
    if next(model.parameters()).is_cuda:
        model = model.cpu()

    model_parameters = {name : param.data for name, param in model.named_parameters()}
    torch.save(model_parameters, filename)


def my_torch_load(model, filename, use_list=None):

    model_parameters = torch.load(filename)
    own_state = model.state_dict()

    for name in model_parameters.keys():
        if use_list is not None:
            if name not in use_list:
                continue
        if name in own_state:
            #print name
            param = model_parameters[name]
            if isinstance(param, Parameter):
                param = param.data

            if own_state[name].shape[:] != param.shape[:]:
                continue
            own_state[name].copy_(param)





