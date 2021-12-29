"""
Data Util functions
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


def add_start_token(ar, token):
    return np.concatenate([token], ar)


def group_data(data, student_column):
    assert_column_exists(data, student_column)
    student_groups = pd.DataFrame(
        [data.groupby(student_column)[x].apply(np.array) for x in data.columns]).T.reset_index(drop=True)
    return student_groups


def ungroup_series(s):
    """
    >>> ungroup_series(pd.Series([[1, 2], [3, 4], [5, 6]])).tolist()
    [1, 2, 3, 4, 5, 6]
    """
    return s.explode().dropna()


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


def clean_data(data, user_col, skill_col, correct_col):
    # Drop nan rows
    _data = data.dropna()
    print(f'Removed {len(data) - len(_data)} rows with nan values in columns {user_col}, {skill_col} or {correct_col}')
    data = _data
    # Categorize skill ids
    data[skill_col] = data[skill_col].astype('category')
    data[skill_col].cat.categories = range(len(data[skill_col].cat.categories))
    assert_column_exists(data, correct_col)
    # Remove non-binary correct rows
    _data = data.query(f'{correct_col} in (0, 1)')
    print(f"Removed {len(data) - len(_data)} non binary correct rows")

    return _data


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
    # sep='±') This breaks cluster runs ¤$¤%#!!!
    avgs_and_sds = means.str.cat(sds, sep=sd_sep)
    print(model_name)
    print(avgs_and_sds)
    if save_path is not None:
        save_dir = os.path.join(save_path, model_name)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_file = os.path.join(save_dir, f'{k}fold-avg-results.csv')
        avgs_and_sds.to_csv(save_file, index=True, header=False)
        print(f'wrote {save_file}')
    return avgs_and_sds
