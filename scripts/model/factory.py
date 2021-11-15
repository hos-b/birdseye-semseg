from model.large_mcnn import TransposedMCNN, TransposedMCNNXL, ExtendedMCNNT, ExtendedMCNNT2xAggr
from model.noisy_mcnn import NoisyMCNNT3x
from model.pyrocc.pyrocc import PyramidOccupancyNetwork
from model.graph_bevnet import GraphBEVNet
from model.dual_mcnn import DualTransposedMCNN4x, DualTransposedMCNN3x, DualMCNNT3Expansive, DualTransposedMCNN2x
from model.dual_mcnn import DualTransposedMCNN3x_1x, DualTransposedMCNN3x_1xPost, DualTransposedMCNN3xFlatMasking

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
    elif model_name == 'mcnnTE2xAggr':
        return ExtendedMCNNT2xAggr(*args)
    elif model_name == 'mcnnT3xNoisy':
        return NoisyMCNNT3x(*args)
    elif model_name == 'pyrocc':
        return PyramidOccupancyNetwork(*args)
    elif model_name == 'bevnet':
        return GraphBEVNet(*args)
    elif model_name == 'mcnnT2x':
        return DualTransposedMCNN2x(*args)
    elif model_name == 'mcnnT3x':
        return DualTransposedMCNN3x(*args)
    elif model_name == 'mcnnT3x1x':
        return DualTransposedMCNN3x_1x(*args)
    elif model_name == 'mcnnT3xFlat':
        return DualTransposedMCNN3xFlatMasking(*args)
    elif model_name == 'mcnnT3x1xPost':
        return DualTransposedMCNN3x_1xPost(*args)
    elif model_name == 'mcnnT3xE':
        return DualMCNNT3Expansive(*args)
    elif model_name == 'mcnnT4x':
        return DualTransposedMCNN4x(*args)
    else:
        raise ValueError(f'unknown model name: {model_name}')