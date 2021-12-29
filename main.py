import argparse
import random
from os import path as osp

import numpy as np
import tensorflow as tf

import conf
from dlkt_model import DLKTModel
from utils import data_utils


class Args(object):
    def __init__(self, args_dict):
        for key, val in args_dict.items():
            self.__dict__[key] = val['default']


def load_data(args, data_file):
    data = data_utils.pandas_format_map[args.data_format](data_file)
    use_cols = [args.user_col, args.correct_col, args.skill_col]
    for col in use_cols:
        data_utils.assert_column_exists(data, col)
    data = data[use_cols]

    if not args.no_data_cleaning:
        print('Categorizing skill_column and ensuring correctness is a binary variable...')
        data = data_utils.clean_data(
            data, user_col=args.user_col, skill_col=args.skill_col, correct_col=args.correct_col)

    print('Data rows after dropping nan rows: {}'.format(len(data)))
    return data


def run(args=None):
    if args is None:
        args = Args(args.args_map)

    # Set random seeds
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    arg_vars = vars(args)
    print('Reading data...')
    data = load_data(args, args.data_file)
    if args.test > 0:
        data = data.iloc[:args.test]

    if args.validation_file is not None:
        print('Reading validation data...')
        validation_data = load_data(args, args.validation_file)
    else:
        validation_data = None

    n_skills = data[args.skill_col].nunique()

    if not args.no_grouping:
        print('grouping data by user ids...')
        data = data_utils.group_data(data, args.user_col)

    data = data_utils.filter_data(data, args.correct_col,
                                  args.min_attempt_count, args.max_attempt_count, args.max_attempt_filter)

    attempt_counts = data[args.correct_col].apply(len)
    print('Data:')
    print('  number of skills:', n_skills)
    print('  number of students in data:', len(data))
    print('  total number of attempts in data: {}'.format(attempt_counts.sum()))
    print('  average attempt sequence length: {:.3f}'.format(
        attempt_counts.mean()))
    print('  minimum attempt sequence length:',
          attempt_counts.min())
    print('  maximum attempt sequence length:', attempt_counts.max())

    print('Model:')
    print('  using model type:', args.model)

    my_dkt_model = DLKTModel(data=data,
                             n_skills=n_skills,
                             dropout=args.dropout,
                             validation_data=validation_data,
                             model_type=args.model,
                             layer_sizes=args.layer_sizes,
                             student_col=args.user_col,
                             use_generator=args.use_generator,
                             skill_col=args.skill_col,
                             correct_col=args.correct_col,
                             onehot_input=args.onehot_input,
                             output_per_skill=args.output_per_skill,
                             batch_size=args.batch_size,
                             init_lr=args.init_lr,
                             early_stopping=args.early_stopping,
                             n_heads=args.n_heads,
                             n_blocks=args.n_blocks)

    if args.kfold > 0:
        my_dkt_model.kfold_eval(dataname=osp.splitext(osp.basename(args.data_file))[0], k=args.kfold,
                                epochs=args.epochs, save_dir=args.save_dir, save_weights=args.save_weights)
    else:
        log_path_hyperparams = '__'.join(
            [f'{x}_{arg_vars[x]}'
             for x in (
                 'init_lr',
                 'dropout',
                 'onehot_input',
                 'output_per_skill',
                 'layer_sizes',
                 'max_attempt_count')
             ])
        log_path = args.log_path if args.log_path is not None \
            else f"{arg_vars['data_file'].split('/')[-1]}__{arg_vars['model']}__{log_path_hyperparams}.log.csv"

        print(f'Using log path: {log_path}')
        print('Train test split rate:', my_dkt_model.train_test_split_rate)

        my_dkt_model.fit(args.epochs, weight_save_path=args.weight_save_path, weight_load_path=args.weight_load_path,
                         log_path=log_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameterized knowledge tracing done simple, maybe',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('data_file')
    parser.add_argument('validation_file', nargs='?', default=None)
    for key, val in conf.args_map.items():
        if val.get('action'):
            parser.add_argument('--' + key,
                                default=val.get('default'),
                                action=val.get('action'),
                                help=str(val.get('help') or '') + ' (default: %(default)s)')
        else:
            parser.add_argument('--' + key,
                                default=val.get('default'),
                                nargs=val.get('nargs'),
                                help=str(val.get('help') or '') +
                                     ' (default: %(default)s)',
                                choices=val.get('choices'),
                                type=val.get('type'))

    args = parser.parse_args()
    run(args)
