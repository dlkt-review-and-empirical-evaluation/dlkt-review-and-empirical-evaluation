import argparse
from numpy import inf
import pandas as pd
from utils import data_utils

if __name__ == "__main__":

    args_map = {
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
        'min-attempt-count': {
            'default': 2,
            'type': int,
            'help': 'Remove students with less than min attempt count'
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
    }

    parser = argparse.ArgumentParser(description='Baselines',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input_file')
    parser.add_argument('output_file')

    for key, val in args_map.items():
        parser.add_argument('--' + key,
                            default=val.get('default'),
                            nargs=val.get('nargs'),
                            help=str(val.get('help') or '') +
                                 '(default: %(default)s)',
                            choices=val.get('choices'),
                            type=val.get('type'))

    args = parser.parse_args()
    data = pd.read_csv(args.input_file, encoding='latin')

    user_col, correct_col, skill_col, min_attempt_count, max_attempt_count, max_attempt_filter = \
        args.user_col, args.correct_col, args.skill_col, args.min_attempt_count, \
        args.max_attempt_count, args.max_attempt_filter

    use_cols = [user_col, correct_col, skill_col]
    for col in use_cols:
        data_utils.assert_column_exists(data, col)
    data = data[use_cols]

    print('Number of attempts in data', len(data))
    data = data_utils.clean_data(data, user_col, skill_col, correct_col)
    data = data.applymap(float)

    grouped = data_utils.group_data(data, user_col)
    grouped = data_utils.filter_data(grouped, correct_col,
                                     min_attempt_count, max_attempt_count, max_attempt_filter)

    ungrouped = grouped.apply(data_utils.ungroup_series)
    print('Number of attempts in data after clean up and filtering', len(data))
    ungrouped.to_csv(args.output_file, index=False)
    print(f'Wrote {args.output_file}')
