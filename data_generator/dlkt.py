import numpy as np
import sklearn as sk
from utils import batch_utils as bu, data_utils
from data_generator.base import BaseDataGenerator


class DLKTDataGenerator(BaseDataGenerator):
    def __init__(self, model, skill_ids, corrects, n_skills=None, output_per_skill=True,
                 batch_size=64,
                 padding='pre',
                 min_step_padding=0, epoch_steps=None):
        """
        skill_ids: array of shape (n_students, n_attempts, 1)
        corrects: array of shape (n_students, n_attempts, 1)
        next_corrects: array of shape (n_students, n_attempts, 1)
        skills_in_output: boolean
        batch_size: integer
        padding: str ('pre', 'post')
        scaling: boolean, only applicable for code vectors
        epoch_steps: integer, provide to reduce time to run epoch, by default computed from input length
        """
        super().__init__(batch_size, padding)

        self.keys = model not in ('lstm-dkt', 'vanilla-dkt')
        self.add_start_token = 'dkvmn' not in model
        self.corrects = corrects
        self.skill_ids = skill_ids
        self.onehot_output = output_per_skill
        self.n_skills = n_skills
        if self.n_skills is None:
            self.n_skills = 0 if skill_ids is None else max(
                max(self.skill_ids)) + 1
        self.onehot_dim = self.n_skills + 1
        self.min_step_padding = min_step_padding

        self.batch_size = batch_size
        self.step = 0
        self.done = False
        self.features_len = len(corrects)
        self.total_steps = epoch_steps if epoch_steps else int(
            np.ceil(float(self.features_len) / self.batch_size))
        self.padding = padding

    def generate_batch(self):
        batch_start, batch_end = super().next_batch()

        batch_inputs = []
        batch_targets = []

        batch_corrects = self.corrects[batch_start:batch_end]
        batch_skill_ids = self.skill_ids[batch_start:batch_end]

        # Key inputs
        if self.keys:
            key_inputs = batch_skill_ids.values
            batch_inputs.append(key_inputs)

        # Value inputs
        if self.add_start_token:
            start_token = self.n_skills * 2

            encoded = data_utils.encode_value_inputs(batch_skill_ids, batch_corrects)
            added_start_token = [np.concatenate([[start_token], x]) for x in encoded]
            value_inputs = [x[:-1] for x in added_start_token]
        else:
            value_inputs = data_utils.encode_value_inputs(
                batch_skill_ids, batch_corrects)
        batch_inputs.append(value_inputs)

        # Targets
        if self.onehot_output:
            batch_targets.append(
                bu.output_per_skill_targets(batch_skill_ids, batch_corrects, self.onehot_dim))
            # targets = [np.stack([s, c], axis=-1) for s, c in zip(batch_skill_ids, batch_corrects)]
        else:
            # Use simple correctness as targets
            batch_targets.append(batch_corrects.values)

        # Fill up incomplete batch
        batch_inputs = [bu.pad_batch_to_ndarray(x, self.batch_size, pad_value=-1.,
                                                squeeze=True, min_steps=self.min_step_padding) for x in batch_inputs]
        batch_targets = [bu.pad_batch_to_ndarray(x, self.batch_size, pad_value=-1.,
                                                 min_steps=self.min_step_padding) for x in batch_targets]

        return batch_inputs, batch_targets

    def generate_all(self):
        if self.keys:
            Q, QA, Y = None, None, None
            while not self.done:
                # Get batch
                (batch_Q, batch_QA), (batch_Y,) = self.generate_batch()
                Q = batch_Q if Q is None else np.concatenate(
                    [Q, batch_Q], axis=0)
                QA = batch_QA if QA is None else np.concatenate(
                    [QA, batch_QA], axis=0)
                Y = batch_Y if Y is None else np.concatenate(
                    [Y, batch_Y], axis=0)
            self.reset()
            return [Q, QA], Y
        else:
            X, Y = None, None
            while not self.done:
                # Get batch
                (batch_X,), (batch_Y,) = self.generate_batch()
                X = batch_X if X is None else np.concatenate(
                    [X, batch_X], axis=0)
                Y = batch_Y if Y is None else np.concatenate(
                    [Y, batch_Y], axis=0)
            self.reset()
            return X, Y

    def shuffle(self):
        new_index = sk.utils.shuffle(np.arange(self.features_len))
        self.skill_ids = self.skill_ids.iloc[new_index]
        self.corrects = self.corrects.iloc[new_index]
