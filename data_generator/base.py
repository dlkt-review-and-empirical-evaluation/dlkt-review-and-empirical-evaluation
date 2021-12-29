from abc import ABC, abstractmethod


class BaseDataGenerator(ABC):
    def __init__(self, batch_size=64, padding='pre', features_len=None, target_dim=None):
        """
        code_vectors: array of shape (n_students, n_attempts, vector_dim)
        skill_ids: array of shape (n_students, n_attempts, 1)
        corrects: array of shape (n_students, n_attempts, 1)
        next_corrects: array of shape (n_students, n_attempts, 1)
        skills_in_input: boolean
        skills_in_output: boolean
        batch_size: integer
        padding: str ('pre', 'post')
        scaling: boolean, only applicable for code vectors
        epoch_steps: integer, provide to reduce time to run epoch, by default computed from input length
        """

        self.batch_size = batch_size
        self.padding = padding

        self.features_len = features_len
        self.step = 0
        self.done = False

    def next_batch(self):
        assert (~self.done)

        batch_start = self.step * self.batch_size
        batch_end = (self.step + 1) * self.batch_size

        if batch_end >= self.features_len:
            self.done = True
            batch_end = self.features_len

        self.step += 1

        return batch_start, batch_end

    @abstractmethod
    def generate_batch(self):
        pass

    @abstractmethod
    def shuffle(self):
        pass

    def reset(self, shuffle=True):
        if shuffle:
            self.shuffle()

        self.done = False
        self.step = 0

    def get_generator(self):
        while True:
            self.reset(False)
            while not self.done:
                batch_features, batch_targets = self.generate_batch()
                yield batch_features, batch_targets

    def get_predict_generator(self):
        while True:
            self.reset(False)
            while not self.done:
                batch_features, batch_targets = self.generate_batch()
                yield batch_features
