# standard library improrts
import copy

# third party imports
from torch.nn import ModuleList
import torch.nn.functional as F

def _get_activation_function(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu
    else:
        raise RuntimeError(f'Activation should be relu or gelu, not {activation}')

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])