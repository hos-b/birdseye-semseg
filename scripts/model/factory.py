from model.large_mcnn import TransposedMCNN, TransposedMCNNXL, ExtendedMCNNT
from model.noisy_mcnn import NoisyMCNN
from model.pyrocc.pyrocc import PyramidOccupancyNetwork
from model.graph_bevnet import GraphBEVNet
from model.dual_mcnn import DualTransposedMCNN


def get_model(model_name: str, *args):
    """
    Get a model by name.
    :param model_name: Name of the model.
    :param args: Additional arguments for the model.
    :return: The model.
    """
    if model_name == 'mcnnT':
        return TransposedMCNN(*args)
    elif model_name == 'mcnnTXL':
        return TransposedMCNNXL(*args)
    elif model_name == 'mcnnTE':
        return ExtendedMCNNT(*args)
    elif model_name == 'mcnnNoisy':
        return NoisyMCNN(*args)
    elif model_name == 'pyrocc':
        return PyramidOccupancyNetwork(*args)
    elif model_name == 'bevnet':
        return GraphBEVNet(*args)
    elif model_name == 'mcnnT4x':
        return DualTransposedMCNN(*args)
    else:
        raise ValueError(f'unknown model name: {model_name}')