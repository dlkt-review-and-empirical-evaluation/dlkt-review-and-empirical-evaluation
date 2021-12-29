import argparse
import os.path as osp

import pandas as pd
import numpy as np
from numpy import inf

from sklearn.metrics import accuracy_score as acc, roc_auc_score as auc, f1_score as f1, matthews_corrcoef as mcc, \
    mean_squared_error as mse, precision_score as prec, recall_score as recall
from sklearn.model_selection import KFold
from utils import data_utils
# from custom_metrics import bic, aic, aicc


def rmse(y_target, y_pred):
    return np.sqrt(mse(y_target, y_pred))


def eval(y_target, y_pred):
    metrics = {
        'acc': acc, 'auc': auc, 'prec': prec, 'recall': recall, 'f1': f1,
        'mcc': mcc, 'rmse': rmse,
        # 'bic': bic, 'aic': aic, 'aicc': aicc
    }
    results = {}
    mses = mse(y_target, y_pred)
    for key, val in metrics.items():
        if key in ['acc', 'mcc', 'f1', 'prec', 'recall']:
            result = val(y_target, np.round(y_pred))
        elif key in ['bic', 'aic', 'aicc']:
            result = val(mses, len(y_target), 0)
        else:
            result = val(y_target, y_pred)
        results[key] = result
    return results


# Simple baselines
def mean(train_data, test_data, correct_col='correct'):
    targets = test_data[correct_col].tolist()
    y_target = np.reshape(targets, [-1])

    y_pred = np.ones(len(y_target)) * np.mean(train_data[correct_col])
    return eval(y_target, y_pred)


def majority_vote(train_data, test_data, correct_col='correct'):
    targets = test_data[correct_col].values
    train_targets = train_data[correct_col]

    y_target = np.reshape(targets, [-1])
    y_pred = np.zeros(len(y_target))
    if train_targets.sum() > len(train_targets) / 2:
        y_pred += 1
    return eval(y_target, y_pred)


def predict_next_as_previous(grouped_data, correct_col='correct'):
    corrects = grouped_data[correct_col]
    first_pred = corrects.apply(np.mean).mean().round()  # predict first as most common label
    predictions = corrects.apply(lambda seq: np.concatenate([[first_pred], seq[:-1]]))
    return eval(corrects.explode().tolist(), predictions.explode().tolist())


def predict_next_as_previous_majority_vote(grouped_data, correct_col, n=3, mean=True):
    mv = (lambda x: np.mean(x)) if mean else lambda x: np.round(np.mean(x))
    targets, preds = [], []
    corrects = grouped_data[correct_col]
    first_pred = corrects.apply(np.mean).mean().round()  # predict first most common label
    for i in range(len(grouped_data)):
        correct_seq = corrects.iloc[i]
        for j in range(len(correct_seq)):
            targets.append(correct_seq[j])
            if j == 0:
                preds.append(first_pred)
            elif j < n:
                preds.append(mv(correct_seq[:j]))
            else:
                preds.append(mv(correct_seq[j - n:j]))
    return eval(targets, preds)


def eval_kfold(grouped_data, dataname='tmp', k=5, correct_col='correct'):
    print(f'running baseline models for {dataname}')

    kfold = KFold(n_splits=k)
    m_results = []
    pnp_results = []

    pnpm_results_dict = {n: [] for n in (3, 5, 9)}

    for train_i, test_i in kfold.split(grouped_data):
        grouped_train = grouped_data.iloc[train_i]
        grouped_test = grouped_data.iloc[test_i]
        data_train = grouped_train.apply(data_utils.ungroup_series)
        data_test = grouped_test.apply(data_utils.ungroup_series)

        m_results.append(mean(data_train, data_test, correct_col))

        pnp_results.append(predict_next_as_previous(
            grouped_test, correct_col))
        for n, results in pnpm_results_dict.items():
            results.append(
                predict_next_as_previous_majority_vote(grouped_test, correct_col, mean=True, n=n))

    save_path = 'baseline-results/' + dataname
    data_utils.avg_kfold_results(
        m_results, 'mean', sd_sep='±', k=5, save_path=save_path)

    data_utils.avg_kfold_results(pnp_results, 'next as previous',
                                 sd_sep='±', k=5, save_path=save_path)
    for n, results in pnpm_results_dict.items():
        data_utils.avg_kfold_results(
            results, f'next as previous {n} mean', sd_sep='±', k=5, save_path=save_path)


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
        'kfold': {
            'default': 5,
            'type': int,
            'help': ' '
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
    parser.add_argument('data_file')
    for key, val in args_map.items():
        parser.add_argument('--' + key,
                            default=val.get('default'),
                            nargs=val.get('nargs'),
                            help=str(val.get('help') or '') +
                                 '(default: %(default)s)',
                            choices=val.get('choices'),
                            type=val.get('type'))

    args = parser.parse_args()
    data = pd.read_csv(args.data_file, encoding='latin')
    datafile = args.data_file.split('/')[-1]

    user_col, correct_col, skill_col, kfold, min_attempt_count, max_attempt_count, max_attempt_filter = \
        args.user_col, args.correct_col, args.skill_col, args.kfold, args.min_attempt_count, \
        args.max_attempt_count, args.max_attempt_filter

    use_cols = [user_col, correct_col, skill_col]
    for col in use_cols:
        data_utils.assert_column_exists(data, col)
    data = data[use_cols]

    data = data_utils.clean_data(data, user_col, skill_col, correct_col)
    data = data.applymap(float)

    grouped = data_utils.group_data(data, user_col)
    grouped = data_utils.filter_data(grouped, correct_col,
                                     min_attempt_count, max_attempt_count, max_attempt_filter)

    eval_kfold(grouped, dataname=osp.splitext(datafile)[0], k=kfold,
               correct_col=correct_col)
