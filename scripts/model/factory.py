from model.large_mcnn import TransposedMCNN, TransposedMCNNXL, ExtendedMCNNT, ExtendedMCNNTDoubleAggr
from model.noisy_mcnn import NoisyMCNNT3x, NoisyMCNNT3xRT
from model.pyrocc.pyrocc import PyramidOccupancyNetwork
from model.dual_mcnn import DualTransposedMCNN4x, DualTransposedMCNN3x, DualMCNNT3Expansive, DualTransposedMCNN2x
from model.dual_mcnn import DualTransposedMCNN3x_1x, DualTransposedMCNN3x_1xPost, DualTransposedMCNN3xFlatMasking
from model.slim_mcnn import SlimMCNNT3x

def get_model(model_name: str, *args, **kwargs):
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
    elif model_name == 'mcnnTEDoubleAggr':
        return ExtendedMCNNTDoubleAggr(*args)
    elif model_name == 'mcnnT3xNoisy':
        return NoisyMCNNT3x(*args)
    elif model_name == 'mcnnT3xNoisyRT':
        return NoisyMCNNT3xRT(
        *args, kwargs.get('mcnnt3x_path', None),
        kwargs.get('detach_mcnn', True)
    )
    elif model_name == 'pyrocc':
        return PyramidOccupancyNetwork(*args)
    elif model_name == 'mcnnT2x':
        return DualTransposedMCNN2x(*args)
    elif model_name == 'mcnnT3x':
        return DualTransposedMCNN3x(*args)
    elif model_name == 'slimcnnT3x64':
        return SlimMCNNT3x(*args, 64)
    elif model_name == 'slimcnnT3x32':
        return SlimMCNNT3x(*args, 32)
    elif model_name == 'slimcnnT3x16':
        return SlimMCNNT3x(*args, 16)
    elif model_name == 'slimcnnT3x8':
        return SlimMCNNT3x(*args, 8)
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