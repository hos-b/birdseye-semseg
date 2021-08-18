from model.large_mcnn import TransposedMCNN, TransposedMCNNXL, ExtendedMCNNT, ExtendedMCNNT2xAggr
from model.noisy_mcnn import NoisyMCNN
from model.pyrocc.pyrocc import PyramidOccupancyNetwork
from model.graph_bevnet import GraphBEVNet
from model.dual_mcnn import DualTransposedMCNN4x, DualTransposedMCNN3x, DualTransposedMCNN2x


def get_model(model_name: str, *args):
    """
    Get a model by name.
    :param model_name: Name of the model.
    :param args: Additional arguments for the model.
    :return: The model.
    """
    if model_name == 'mcnnT':
        print('creating TransposedMCNN')
        return TransposedMCNN(*args)
    elif model_name == 'mcnnTXL':
        print('creating TransposedMCNNXL')
        return TransposedMCNNXL(*args)
    elif model_name == 'mcnnTE':
        print('creating ExtendedMCNNT')
        return ExtendedMCNNT(*args)
    elif model_name == 'mcnnTE2xAggr':
        print('creating ExtendedMCNNT2xAggr')
        return ExtendedMCNNT2xAggr(*args)
    elif model_name == 'mcnnNoisy':
        print('creating NoisyMCNN')
        return NoisyMCNN(*args)
    elif model_name == 'pyrocc':
        print('creating PyramidOccupancyNetwork')
        return PyramidOccupancyNetwork(*args)
    elif model_name == 'bevnet':
        print('creating GraphBEVNet')
        return GraphBEVNet(*args)
    elif model_name == 'mcnnT2x':
        print('creating DualTransposedMCNN2x')
        return DualTransposedMCNN2x(*args)
    elif model_name == 'mcnnT4x':
        print('creating DualTransposedMCNN4x')
        return DualTransposedMCNN4x(*args)
    elif model_name == 'mcnnT3x':
        print('creating DualTransposedMCNN3x')
        return DualTransposedMCNN3x(*args)
    else:
        raise ValueError(f'unknown model name: {model_name}')