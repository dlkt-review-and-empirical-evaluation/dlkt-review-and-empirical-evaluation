from argparse import ArgumentTypeError
from distutils.util import strtobool

from numpy import inf
import model_creator

model_choices = model_creator.model_choices


def boolean(x):
    """
    >>> boolean('true')
    True
    >>> boolean('false')
    False
    """
    return bool(strtobool(x))


def comma_separated_list(x):
    """
    >>> comma_separated_list('10')
    [10]
    >>> comma_separated_list('20,30')
    [20,30]
    """
    try:
        return [int(val) for val in x.split(',')]
    except:
        raise ArgumentTypeError(
            f"Could not parse {x} to a list. Expected a comma separated list of integers, e.g. 20,50")


args_map = {
    'layer-sizes': {
        'default': "50,20,40",
        'type': comma_separated_list,
        'help': 'Comma separated values (up to four) for sizes of neural network hidden layers, the values will override model defaults\
                \nLayers in order: recurrent key_embedding value_embedding skill_summary'
    },
    'use-generator': {
        'default': False,
        'action': 'store_true',
        'help': 'Whether to generate padded input data sequences per batch to save memory.\
                 \nOtherwise all attempt sequences will be padded to max attempt count at once, which is faster if the data can fit to memory.'
    },
    'epochs': {
        'default': 100,
        'type': int,
        'help': 'Number of training passes.'
    },
    'min-attempt-count': {
        'default': 2,
        'type': int,
        'help': 'Remove students with less than min attempt count attempts. (default 2 is to use only attempts with a next attempt)'
    },
    'max-attempt-count': {
        'default': inf,
        'type': int,
        'help': 'Apply maximum attempt count to filter or split attempt sequences'
    },
    'max-attempt-filter': {
        'default': 'split',
        'choices': ['split', 'remove', 'cut'],
        'type': str,
        'help': 'Determine how maximum attempt count is applied. \
                  \nSplit creates more data (implemented chiefly to test SAKT). \
                  \nRemove removes students similarly to min-attempt-count). \
                  \nCut removes attempts beyond max attempt count'
    },
    'target-mode': {
        'default': 'next-skills',
        'choices': ['skills', 'next-skills'],  # Todo: implement both as in (DKT+)
        'type': str
    },
    'data-format': {
        'default': 'csv',
        'choices': ['pickle', 'csv', 'tsv', 'hdf'],
        'type': str,
        'help': 'Hdf assumes only one dataframe is saved in the file'
    },
    'batch-size': {
        'default': 32,
        'type': int,
        'help': ' '
    },
    'init-lr': {
        'default': 0.01,
        'type': float,
        'help': 'Initial learning rate'
    },
    'skill-col': {
        'default': 'skill_id',
        'type': str,
        'help': ' '
    },
    'correct-col': {
        'default': 'correct',
        'type': str,
        'help': ' '
    },
    'user-col': {
        'default': 'user_id',
        'type': str,
        'help': ''
    },
    'model': {
        'default': 'lstm-dkt',
        'choices': model_choices,
        'type': str,
        'help': 'Select model architecture'
    },
    'test': {
        'default': 0,
        'type': int,
        'help': 'Reduces number of data points to make testing faster'
    },
    'load-weights': {
        'default': False,
        'type': bool,
        'help': 'whether to load weights before training or not, uses weight-save-path if weight-load-path is not provided',
    },
    'weight-save-path': {
        'default': None,
        'type': str,
        'help': 'model weight filepath to for saving weights'
    },
    'weight-load-path': {
        'default': None,
        'type': str,
        'help': 'model weight filepath to load weights from'
    },
    'model-save-path': {
        'default': None,
        'type': str,
        'help': ' '
    },
    'dropout': {
        'default': 0.4,
        'type': float,
        'help': 'dropout rate'
    },
    'log-path': {
        'default': None,
        'type': str,
        'help': 'Non-kfold option: path to save result logs and possibly weights (if save-weights is provided)'
    },
    'kfold': {
        'default': 0,
        'type': int,
        'help': ' '
    },
    'no-grouping': {
        'action': 'store_true',
        'help': 'Prevents grouping of data, this should be used if input data is already grouped by user ids'
    },
    'no-data-cleaning': {
        'action': 'store_true',
        'help': 'Prevents cleaning data. Cleaning includes categorizing skill ids and binarizing non-binary correctnesses'
    },
    'early-stopping': {
        'default': 5,
        'type': int,
    },
    'seed': {
        'default': 42,
        'type': int,
        'help': 'random seed for reproducibility'
    },
    'n-heads': {
        'default': 5,
        'type': int,
        'help': 'number of attention heads for multi-head attention models. Applies only for the transformer models'
    },
    'n-blocks': {
        'default': 1,
        'type': int,
        'help': 'number of transformer blocks for transformer models. Applies only for the transformer models'
    },
    'onehot-input': {
        'action': 'store_true',
        'help': 'Embeds the integer inputs of the models by onehotting instead of using and embedding layer'
    },
    'output-per-skill': {
        'action': 'store_true',
        'help': 'Output a binary value per skill (as in original DKT) instead of a skill layer + binary output layer'
    },
    'save-dir': {
        'default': None,
        'type': str,
        'help': 'Kfold option: directory where results and model weights will be saved'
    },
    'save-weights': {
        'action': 'store_true',
        'help': 'saves weights in save-dir (kfold) or weight-save-path'
    }
}
