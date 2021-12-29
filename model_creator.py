import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam
from tensorflow import metrics

from models.dkvmn import dkvmn
from models.rnn_dkt import rnn_dkt
from models.rnn_dkt_s_plus import rnn_dkt_s_plus
from models.sakt import sakt
from models.transformer import transformer

tf.compat.v1.disable_eager_execution()
print('Eager execution: ', tf.executing_eagerly())

model_choices = [
    'vanilla-dkt',
    'lstm-dkt',
    'lstm-dkt-s+',
    'dkvmn-paper',
    'dkvmn',
    'sakt',
    'transformer',
]

def create_model(modelname='lstm-dkt',
                 layer_dims=tuple(),
                 n_skills=0,
                 max_attempt_count=None,
                 onehot_inputs=False, output_per_skill=False,
                 batch_size=64, init_lr=0.001, dropout=.2, grad_clipnorm=1.,
                 n_heads=5, n_blocks=1):
    def skill_out_loss(y_true, y_pred):
        target_skills = y_true[:, :, 0:n_skills]
        target_labels = y_true[:, :, n_skills]
        target_preds = K.sum(y_pred * target_skills, axis=2)
        return masked_binary_crossentropy(target_labels, target_preds)

    def masked_binary_crossentropy(true, pred):
        # Masking is done in embedding layers so this is not needed.
        mask = tf.cast(tf.not_equal(true, -1.), tf.float32)
        loss = K.binary_crossentropy(true, pred)
        return tf.multiply(loss, mask)

    def compile(inputs, outputs, name='model'):
        loss = skill_out_loss if output_per_skill else masked_binary_crossentropy
        model = Model(inputs=inputs, outputs=outputs, name=name)
        model.compile(loss=loss, optimizer=Nadam(learning_rate=init_lr, clipnorm=grad_clipnorm))
        model.summary()
        return model

    recurrent_layer_size = layer_dims[0] if len(layer_dims) > 0 else 50
    key_embedding_size = layer_dims[1] if len(layer_dims) > 1 else 50
    val_embedding_size = layer_dims[2] if len(layer_dims) > 2 else 50
    skill_layer_size = None if output_per_skill else (
        layer_dims[3] if len(layer_dims) > 3 else n_skills)

    def get_model(model, residual=True):
        key_inputs = Input(batch_shape=(batch_size, None))
        val_inputs = Input(batch_shape=(batch_size, None))

        common_args = dict(val_inputs=val_inputs,
                           recurrent_layer_size=recurrent_layer_size,
                           n_skills=n_skills,
                           onehot=onehot_inputs,
                           skill_layer_size=skill_layer_size,
                           dropout=dropout)

        if model in ('lstm-dkt', 'vanilla-dkt'):
            version = model.split('-')[0]
            outputs = rnn_dkt(kernel_type=version, **common_args,
                              embedding_size=val_embedding_size)
        elif model == 'lstm-dkt-s+':
            outputs = rnn_dkt_s_plus(**common_args,
                                     rnn_version='lstm',
                                     key_inputs=key_inputs,
                                     key_embedding_size=key_embedding_size,
                                     val_embedding_size=val_embedding_size)
        elif model in ('dkvmn', 'dkvmn-paper'):
            paper_version = model == 'dkvmn-paper'
            outputs = dkvmn(**common_args,
                            paper_version=paper_version,
                            key_inputs=key_inputs,
                            key_embedding_size=key_embedding_size,
                            val_embedding_size=val_embedding_size,
                            batch_size=batch_size)
        elif model == 'sakt':
            outputs = sakt(**common_args,
                           key_inputs=key_inputs,
                           key_embedding_size=key_embedding_size,
                           val_embedding_size=val_embedding_size,
                           max_attempt_count=max_attempt_count,
                           residual=residual,
                           n_heads=n_heads,
                           n_blocks=n_blocks)
        elif model == 'transformer':
            outputs = transformer(**common_args,
                                  key_inputs=key_inputs,
                                  key_embedding_size=key_embedding_size,
                                  val_embedding_size=val_embedding_size,
                                  max_attempt_count=max_attempt_count,
                                  residual=residual,
                                  n_heads=n_heads,
                                  n_blocks=n_blocks)
        else:
            raise NotImplementedError(f'model {model} is not implemented')

        inputs = [val_inputs] if model in ('lstm-dkt', 'vanilla-dkt') else [key_inputs, val_inputs]
        return compile(inputs, outputs, name=model.replace('+', 'plus'))

    if modelname in model_choices:
        print('  model =', modelname)
        print('  batch_size =', batch_size)
        print('  dropout =', dropout)
        print('  gradient clip norm value =', grad_clipnorm)
        print('  init lr =', init_lr)
        return get_model(modelname)
    raise NotImplementedError(f'Model {modelname} not implemented.')
