import os
import pickle as pkl
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

from data_generator.dlkt import DLKTDataGenerator
from model_creator import create_model
from model_evaluation import MetricsCallback, model_evaluate
from utils import data_utils


class DLKTModel:
    def __init__(self, data,
                 n_skills,
                 validation_data=None,
                 layer_sizes=(100,),
                 model_type='dkt',
                 student_col='user_id', skill_col='skill_id', correct_col='correct_binary',
                 max_attempt_count=None,
                 dropout=0,
                 batch_size=32,
                 init_lr=0.01,
                 train_test_split_rate=.1,
                 verbose=2,
                 use_generator=True,
                 early_stopping=5,
                 n_heads=5,
                 n_blocks=1,
                 onehot_input=False,
                 output_per_skill=False
                 ):
        self.verbose = verbose

        self.data = data
        self.student_col = student_col
        self.skill_col = skill_col
        self.correct_col = correct_col

        self.model_type = model_type
        self.use_generator = use_generator
        self.validation_data = validation_data
        self.n_skills = n_skills
        self.early_stopping = early_stopping

        self.max_attempt_count = max_attempt_count

        self.rnn_layer_sizes = layer_sizes
        self.dropout = dropout
        self.batch_size = batch_size
        self.init_lr = init_lr

        self.train_test_split_rate = train_test_split_rate
        self.attempt_counts = self.data[self.correct_col].apply(len)

        self.onehot_input = onehot_input
        self.output_per_skill = output_per_skill

        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.model = None
        self.init_model()

    def init_model(self):
        self.model = create_model(layer_dims=self.rnn_layer_sizes,
                                  n_skills=self.n_skills,
                                  onehot_inputs=self.onehot_input,
                                  output_per_skill=self.output_per_skill,
                                  max_attempt_count=self.attempt_counts.max(),
                                  batch_size=self.batch_size, init_lr=self.init_lr, dropout=self.dropout,
                                  modelname=self.model_type,
                                  n_heads=self.n_heads,
                                  n_blocks=self.n_blocks)

    def create_data_generator(self, data, min_step_padding=0):

        return DLKTDataGenerator(
            skill_ids=data[self.skill_col],
            corrects=data[self.correct_col],
            n_skills=self.n_skills,
            model=self.model_type,
            output_per_skill=self.output_per_skill,
            batch_size=self.batch_size,
            min_step_padding=min_step_padding)

    def predict(self, test_gen):
        def unonehot_targets(labels, preds):
            target_skills = labels[:, :, 0:test_gen.n_skills]
            target_labels = labels[:, :, test_gen.n_skills]
            target_preds = np.sum(preds * target_skills, axis=2)

            return target_labels, target_preds

        test_gen.reset()
        labels, predictions = None, None
        while not test_gen.done:
            batch_features, batch_labels = test_gen.generate_batch()
            batch_predictions = self.model.predict_on_batch(batch_features)

            batch_labels = np.squeeze(batch_labels)
            batch_predictions = np.squeeze(batch_predictions)
            if test_gen.onehot_output:
                batch_labels, batch_predictions = unonehot_targets(batch_labels, batch_predictions)

            labels = batch_labels.ravel() if labels is None else np.concatenate([labels, batch_labels.ravel()], axis=0)
            predictions = batch_predictions.ravel() if predictions is None else np.concatenate(
                [predictions, batch_predictions.ravel()], axis=0)

        return labels, predictions

    def fit(self, epochs=100, load=True, weight_save_path=None, log_path=None, weight_load_path=None, verbose=1):
        if load and weight_save_path is not None:
            if weight_load_path is None:
                weight_load_path = weight_save_path
            if not os.path.exists(weight_load_path):
                print(
                    'Load file {} does not exist. Not loading weights.'.format(weight_load_path))
            else:
                print('loading model weights from {}'.format(weight_load_path))
                self.model.load_weights(weight_load_path)
                print('loaded model weights')

        if self.validation_data is None:
            train_i, test_i = train_test_split(
                range(len(self.data)), test_size=self.train_test_split_rate)
            train_data = self.data.iloc[train_i].reset_index(drop=True)
            test_data = self.data.iloc[test_i].reset_index(drop=True)
        else:
            train_data = self.data
            test_data = self.validation_data
        train_data_generator = self.create_data_generator(train_data)
        test_data_generator = self.create_data_generator(test_data)

        callbacks = []

        if self.early_stopping is not None:
            callbacks.append(EarlyStopping(
                patience=self.early_stopping, restore_best_weights=True))
        if not weight_save_path is None:
            callbacks.append(
                ModelCheckpoint(weight_save_path, monitor='val_loss', verbose=self.verbose, save_best_only=True,
                                save_weights_only=True))
        if not log_path is None:
            callbacks.append(CSVLogger(log_path))

        if self.use_generator:
            callbacks = [MetricsCallback(test_data_generator)] + callbacks
            print('Using generator')
            self.model.fit(
                train_data_generator.get_generator(),
                steps_per_epoch=train_data_generator.total_steps,
                validation_data=test_data_generator.get_generator(),
                validation_steps=test_data_generator.total_steps,
                epochs=epochs,
                verbose=verbose,
                callbacks=callbacks
            )
        else:
            print('Preprocessing inputs...')
            train_data_generator.min_step_padding = self.attempt_counts.max()
            test_data_generator.min_step_padding = self.attempt_counts.max()
            data = train_data_generator.generate_all()
            test_data = test_data_generator.generate_all()
            callbacks = [MetricsCallback(test_data_generator)] + callbacks
            self.model.fit(*data,
                           validation_data=test_data,
                           epochs=epochs, verbose=verbose,
                           callbacks=callbacks)

    def kfold_eval(self, dataname, k=5, weight_file='kfold.weights', save_logs=False, val_rate=.1,
                   save_dir=None, save_weights=False,
                   epochs=50, verbose=1):

        if save_dir is None:
            save_dir = "dlkt-kfold-results/{}-{}-s{}-l{}".format(dataname, self.model_type,
                                                                 self.skill_col,
                                                                 '-'.join([str(x) for x in self.rnn_layer_sizes]))

        save_dirpath = Path(save_dir)
        weight_file = os.path.join(save_dir, weight_file)
        save_dirpath.mkdir(parents=True, exist_ok=True)

        kfold = KFold(n_splits=k)
        results = []
        results_file = os.path.join(save_dir, f'{k}-fold-results.pkl')
        if os.path.isfile(results_file):
            with open(results_file, 'rb') as f:
                results = pkl.load(f)

        # Use same initial model weights for all folds

        for i, (train_val_i, test_i) in enumerate(kfold.split(self.data)):
            if i < len(results):  # Continue from last unfinished fold
                continue
            print("fold", i)
            train_val_data = self.data.iloc[train_val_i].reset_index(drop=True)
            train_i, val_i = train_test_split(
                range(len(train_val_data)), test_size=val_rate)

            train_data = train_val_data.iloc[train_i].reset_index(drop=True)
            val_data = train_val_data.iloc[val_i].reset_index(drop=True)
            test_data = self.data.iloc[test_i].reset_index(drop=True)

            train_data_generator = self.create_data_generator(train_data)
            val_data_generator = self.create_data_generator(val_data)

            callbacks = []
            if verbose > 1:
                callbacks.append(MetricsCallback(val_data_generator))
            if save_logs:
                log_file = os.path.join(save_dir, f'{k}-fold.log.csv')
                callbacks.append(CSVLogger(log_file + str(i)))
            if save_weights:
                callbacks.append(ModelCheckpoint(weight_file + str(i), monitor='val_loss', verbose=self.verbose,
                                                 save_best_only=True, save_weights_only=True))
            if self.early_stopping is not None:
                callbacks.append(EarlyStopping(
                    patience=self.early_stopping, restore_best_weights=True))

            if i > 0:
                print('resetting weights...')
                self.init_model()

            # Train model
            self.model.fit(
                train_data_generator.get_generator(),
                validation_data=val_data_generator.get_generator(),
                steps_per_epoch=train_data_generator.total_steps,
                validation_steps=val_data_generator.total_steps,
                epochs=epochs,
                verbose=verbose,
                callbacks=callbacks
            )

            print("evaluating model...")
            test_data_generator = self.create_data_generator(test_data)
            results.append(model_evaluate(
                test_data_generator, self.model))

            print(f"saving results to {save_dir}...")
            with open(results_file, 'wb') as f:
                pkl.dump(results, f)
            print(f'wrote {results_file}')

        avg_results_file = os.path.join(
            save_dir, '{}fold-avg-results.csv'.format(k))
        data_utils.avg_kfold_results(results, dataname).to_csv(
            avg_results_file, encoding='utf-8', header=False)
        print('wrote:', avg_results_file)
