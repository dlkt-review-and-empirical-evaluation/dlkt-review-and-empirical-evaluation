"""
Util functions
currently mainly data manipulation TODO: maybe refactor as data utils
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd


def encode_value_inputs(skill_ids: pd.Series, corrects: pd.Series):
    s = skill_ids.values
    c = corrects.values
    return s * 2 + c


# Ref: https://groups.google.com/forum/#!msg/keras-users/7sw0kvhDqCw/QmDMX952tq8J
def pad_sequences(sequences, maxlen=None, dim=1, dtype='int32', padding='pre', truncating='pre', padding_value=0.):
    '''
        Override keras method to allow multiple feature dimensions.

        @dim: input feature dimension (number of features per timestep)
    '''
    lengths = [len(s) for s in sequences]

    n_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((n_samples, maxlen, dim)) * padding_value).astype(dtype)
    for i, s in enumerate(sequences):
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError(f"Truncating type {padding} not understood")

        if padding == 'post':
            x[i, :len(trunc)] = trunc
        elif padding == 'pre':
            x[i, -len(trunc):] = trunc
        else:
            raise ValueError(f"Truncating type {padding} not understood")
    return x


def filter_data(data, correct_column, min_attempt_count=5, max_attempt_count=None, max_filter_mode='split'):
    print("Filtering:")
    print("  min attempt count:", min_attempt_count)
    print("  max attempt count:", max_attempt_count)
    print("  max attempt filtering mode:", max_filter_mode)
    print('  number of students before applying filters: {}'.format(len(data)))
    # Remove student data with insufficient attempts
    attempt_counts = data[correct_column].apply(len)
    data = data[attempt_counts >= min_attempt_count]

    print(
        '  number of students after removing students with less than {} attempts: {}'.format(min_attempt_count,
                                                                                             len(data)))
    if not isinstance(max_attempt_count, int):
        return data.dropna().reset_index(drop=True)

    # Remove student data with high attempt count
    attempt_counts = data[correct_column].apply(len)
    if max_filter_mode == 'remove':
        data = data[attempt_counts <= max_attempt_count]
    elif max_filter_mode == 'cut':
        data = data.applymap(lambda x: x[:max_attempt_count])
    elif max_filter_mode == 'split':
        to_split = data[attempt_counts > max_attempt_count]
        while len(to_split) > 0:
            no_split = data[attempt_counts <= max_attempt_count]
            split = to_split.applymap(lambda x: x[:max_attempt_count])
            rest = to_split.applymap(lambda x: x[max_attempt_count:])
            data = pd.concat([no_split, split, rest]).reset_index(drop=True)
            attempt_counts = data[correct_column].apply(len)
            to_split = data[attempt_counts > max_attempt_count]

    print("  number of students after applying max attempt filter {}: {}".format(
        max_filter_mode, len(data)))
    return data.dropna().reset_index(drop=True)


def assert_vector_data_validity(vectors, vector_size=None):
    if vector_size is None:
        vector_size = len(vectors[0][0])

    print('Asserting vector size is {} for all data points'.format(vector_size))

    for attempts in vectors:
        n = len(attempts)
        assert n > 1, 'Sequence length should be greater than 1, found {}'.format(
            n)

        for ar in attempts:
            assert isinstance(attempts, (np.ndarray, list, tuple)
                              ), 'expected iterable, found: {}'.format(ar)
            assert len(
                ar) == vector_size, 'All vectors should have same dimension count, found {}'.format(ar)

    print('All fine')


def offset(ar, n=1, keep_dim=False):
    if not keep_dim:
        return ar[n:]
    return ar[n:] + [np.nan] * n


def add_start_token(ar, token):
    return [token] + [ar]


def group_data(data, student_column):
    assert_column_exists(data, student_column)
    student_groups = pd.DataFrame(
        [data.groupby(student_column)[x].apply(np.array) for x in data.columns]).T.reset_index(drop=True)
    # Create columns for skill ids and corrects at time t+1
    return student_groups


def add_next_skills_and_corrects(data, skill_column, correct_column, next_prefix='next_'):
    next_skill_column = next_prefix + skill_column
    next_correct_column = next_prefix + correct_column
    data[next_skill_column] = data[skill_column].apply(offset)
    data[next_correct_column] = data[correct_column].apply(offset)
    # Remove final attempts because there are no next correctness targets, (next cols final value is nan)
    for column in data.columns:
        data[column] = data[column].apply(lambda x: x[:len(x) - 1])
    return data


def ungroup_series_fast(s):
    return s.explode().dropna()


def ungroup_series(s):
    """
    >>> ungroup_series(pd.Series([[1, 2], [3, 4], [5, 6]])).tolist()
    [1, 2, 3, 4, 5, 6]
    """
    ungrouped = []
    for group in s:
        ungrouped += group
    return pd.Series(ungrouped)


def assert_column_exists(df, col):
    assert col in df.columns, f"""
column "{col}" not found in given data.
Found columns: {df.columns.values}.
Please specify which of the data columns is used instead.
  --skill-col SKILL_COL (default: skill_id)
  --correct-col CORRECT_COL (default: correct)
  --user-col USER_COL (default: user_id)
Run the command with argument --help for more information.
"""


def clean_data(data, skill_col, correct_col):
    assert_column_exists(data, skill_col)
    # Categorize skill ids
    data[skill_col] = data[skill_col].astype('category')
    data[skill_col].cat.categories = range(len(data[skill_col].cat.categories))
    assert_column_exists(data, correct_col)
    # Convert correctness to binary
    max_percentages_per_exercise = data.groupby(
        [skill_col])[correct_col].max().to_dict()
    max_pass_percentage = data[skill_col].apply(
        lambda x: max_percentages_per_exercise[x])
    data[correct_col] = (data[correct_col] == max_pass_percentage).apply(int)
    return data


def read_csv(filepath):
    return pd.read_csv(filepath, encoding='latin', low_memory=False)


def read_tsv(filepath):
    return pd.read_csv(filepath, encoding='latin', low_memory=False, delimiter='\t')


pandas_format_map = {
    'pickle': pd.read_pickle,
    'hdf': pd.read_hdf,
    'csv': read_csv,
    'tsv': read_tsv
}


def avg_kfold_results(results, model_name='model', sd_sep='/', k=5, save_path=None):
    means = pd.DataFrame(results).mean().apply(
        lambda x: np.round(x, 3)).apply(str)
    sds = pd.DataFrame(results).std().apply(
        lambda x: np.round(x, 3)).apply(str)
    # sep='±') This breaks triton runs ¤$¤%#!!!
    avgs_and_sds = means.str.cat(sds, sep=sd_sep)
    print(model_name)
    print(avgs_and_sds)
    if save_path is not None:
        save_dir = os.path.join(save_path, model_name)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_file = os.path.join(save_dir, f'{k}fold-avgs.csv')
        avgs_and_sds.to_csv(save_file, index=True, header=False)
        print(f'wrote {save_file}')
    return avgs_and_sds
