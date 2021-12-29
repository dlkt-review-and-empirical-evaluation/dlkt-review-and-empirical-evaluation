import numpy as np
from sklearn.preprocessing import StandardScaler

from utils import data_utils

"""
Tailored for dkt data generators
Most methods assume batches are of shape (batch_size, sequence_length (may vary within batch), feature_dim)
This definitely not gonna work as general utils as is
"""


def output_per_skill_targets(batch_skill_ids, batch_targets, target_dim):
    y = []
    for skill_ids, targets in zip(batch_skill_ids, batch_targets):
        y_student = np.zeros([len(skill_ids), target_dim])
        for i, (skill_id, target) in enumerate(zip(skill_ids, targets)):
            y_student[i, int(skill_id)] = 1
            y_student[i, -1] = target
        y.append(y_student)
    return y


def pad_batch_sequences(x, value=-1., padding='pre', max_steps=0):
    dim = len(x[0][0])
    max_seq_steps = max([len(seq) for seq in x] + [max_steps])
    return data_utils.pad_sequences(x, padding=padding, maxlen=max_seq_steps, dim=dim, padding_value=value,
                                    dtype='float32')


def scale_batch(x):
    scaler = StandardScaler()
    return [scaler.fit_transform(seq) for seq in x]


def fill_batch(x, batch_size, padding_value=-1.):
    if x.shape[0] == batch_size:
        return x

    pad = np.ones([batch_size - x.shape[0], *x.shape[1:]]) * padding_value
    return np.concatenate([x, pad])


def pad_batch_to_ndarray(x, batch_size=64, pad_value=-1., padding='pre', min_steps=0, squeeze=False):
    if x[0].ndim == 1:
        x = [np.expand_dims(_x, axis=-1) for _x in x]
    padded = pad_batch_sequences(x, pad_value, padding, min_steps)
    filled = fill_batch(padded, batch_size, pad_value)
    if squeeze:
        return np.squeeze(filled)
    return filled
