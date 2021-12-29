from sklearn.metrics import accuracy_score as acc, roc_auc_score as auc, f1_score as f1, matthews_corrcoef as mc, \
    mean_squared_error as mse, precision_score as prec, recall_score as recall, log_loss
from tensorflow.keras.callbacks import Callback
import numpy as np
# from tensorflow.python.keras.utils import layer_utils
# from custom_metrics import bic, aic, aicc

metrics = ('acc', 'auc', 'prec', 'recall', 'f1', 'mcc',
           'rmse', 'log-loss',
           #    'aic', 'aicc', 'bic',
           )


def model_evaluate(test_gen, model, data=None):
    def unonehot_targets(labels, preds):
        target_skills = labels[:, :, 0:test_gen.n_skills]
        target_labels = labels[:, :, test_gen.n_skills]

        target_preds = np.sum(preds * target_skills, axis=2)

        return target_labels, target_preds

    def predict_in_batches():
        test_gen.reset()
        labels, predictions = None, None
        while not test_gen.done:
            batch_features, batch_labels = test_gen.generate_batch()
            batch_predictions = model.predict_on_batch(batch_features)

            batch_labels = np.squeeze(batch_labels)
            batch_predictions = np.squeeze(batch_predictions)
            if test_gen.onehot_output:
                batch_labels, batch_predictions = unonehot_targets(batch_labels, batch_predictions)

            labels = batch_labels.ravel() if labels is None else np.concatenate([labels, batch_labels.ravel()], axis=0)
            predictions = batch_predictions.ravel() if predictions is None else np.concatenate(
                [predictions, batch_predictions.ravel()], axis=0)

        return labels, predictions

    def predict():
        features, labels = data
        predictions = model.predict(features)

        labels = np.squeeze(labels)
        predictions = np.squeeze(predictions)
        if test_gen.onehot_output:
            labels, predictions = unonehot_targets(labels, predictions)
        return labels, predictions

    if data is None:
        labels, predictions = predict_in_batches()
    else:
        labels, predictions = predict()

    y_true, y_pred = labels, predictions

    # padding_i = np.flatnonzero(y_true == -1.0).tolist()
    # y_t = y_true.ravel()[padding_i]
    # assert np.all(y_t == -1.0)

    not_padding_i = np.flatnonzero(y_true != -1.0).tolist()
    y_true = y_true.ravel()[not_padding_i]
    y_pred = y_pred.ravel()[not_padding_i]
    # assert np.all(y_true >= 0.0)

    bin_pred = y_pred.round()
    # mses = mse(y_true, y_pred)
    # n_params = layer_utils.count_params(model.trainable_weights)

    results = {}
    results['acc'] = acc(y_true, bin_pred)
    results['auc'] = auc(y_true, y_pred)
    results['prec'] = prec(y_true, bin_pred)
    results['recall'] = recall(y_true, bin_pred)
    results['f1'] = f1(y_true, bin_pred)
    results['mcc'] = mc(y_true, bin_pred)
    results['rmse'] = np.sqrt(mse(y_true, y_pred))
    # results['aic'] = aic(mses, len(y_true), n_params)
    # results['aicc'] = aicc(mses, len(y_true), n_params)
    # results['bic'] = bic(mses, len(y_true), n_params)
    results['log-loss'] = log_loss(y_true, y_pred)

    return results


class MetricsCallback(Callback):
    def __init__(self, val_data_gen, val_data=None):
        super(MetricsCallback, self).__init__()
        assert (metrics is not None)

        self.val_data = val_data
        self.val_data_gen = val_data_gen

    def on_train_begin(self, logs={}):
        for metric in metrics:
            if not 'metrics' in self.params: return
            self.params['metrics'].append('val_' + metric)

    def on_epoch_end(self, epoch, logs={}):
        results = model_evaluate(self.val_data_gen, self.model, self.val_data)
        for metric in metrics:
            logs['val_' + metric] = results[metric]
