import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Embedding, TimeDistributed, Dropout
# from tensorflow_core.python.keras.layers import Layer, Dense, Embedding, TimeDistributed, Dropout
from tensorflow.python.keras.layers.dense_attention import BaseDenseAttention


def skill_outputs(inputs, skill_layer_size=None, n_skills=None, dropout=.2):
    if skill_layer_size is None:
        return TimeDistributed(Dense(n_skills,
                                     activation='sigmoid'))(inputs)
    else:
        skill_layer = TimeDistributed(Dense(skill_layer_size,
                                            activation='tanh', name='skill_summary'))
        skills = Dropout(dropout)(skill_layer(inputs))
        return TimeDistributed(Dense(1, activation='sigmoid'))(skills)


def embeddings(inputs, dim, onehot=True, layer_size=None):
    if onehot:
        return OneHotEmbedding(dim)(inputs)
    else:
        assert layer_size is not None
        return CustomMaskEmbedding(dim, layer_size,
                                   mask_value=-1.)(inputs)


class TimeBiasedAttention(BaseDenseAttention):
    def __init__(self, use_scale=False, **kwargs):
        super(TimeBiasedAttention, self).__init__(**kwargs)
        self.use_scale = use_scale

    def build(self, input_shape):
        """Creates scale variable if use_scale==True."""
        if self.use_scale:
            self.scale = self.add_weight(
                name='scale',
                shape=(),
                initializer=tf.ones_initializer(),
                dtype=self.dtype,
                trainable=True)
        else:
            self.scale = None
        super(TimeBiasedAttention, self).build(input_shape)

    def _calculate_scores(self, query, key):
        """Calculates attention scores as a query-key dot product weighted by position distance.

        Args:
          query: Query tensor of shape `[batch_size, Tq, dim]`.
          key: Key tensor of shape `[batch_size, Tv, dim]`.
        Returns:
          Tensor of shape `[batch_size, Tq, Tv]`.
        """
        print(f'key shape {key.shape}, query shape {query.shape}')
        scores = tf.divide(tf.matmul(query, key, transpose_b=True),
                           tf.sqrt(tf.cast(key.shape[-1], tf.float32)))
        if self.scale is not None:
            scores *= self.scale
        return scores

    def get_config(self):
        config = {'use_scale': self.use_scale}
        base_config = super(TimeBiasedAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CustomMaskEmbedding(Embedding):
    def __init__(self, input_dim, output_dim, mask_value, **kwargs):
        super().__init__(input_dim + 1, output_dim, embeddings_initializer='glorot_uniform', **kwargs)
        self.mask_value = mask_value

    def call(self, inputs):
        return super().call(inputs + 1)

    def compute_mask(self, inputs, mask=None):
        return tf.not_equal(inputs, self.mask_value)


class OneHotEmbedding(Layer):
    def __init__(self, output_dim, mask_value=-1., **kwargs):
        super(OneHotEmbedding, self).__init__(trainable=False, **kwargs)
        self.output_dim = output_dim
        self.mask_value = mask_value

    def call(self, inputs, **kwargs):
        return tf.one_hot(tf.cast(inputs, 'int32'), self.output_dim, axis=-1)

    def compute_mask(self, inputs, mask=None):
        return tf.not_equal(inputs, self.mask_value)


def dot_product(t1, t2):
    """
    return: dot product over last dimension
    """
    return tf.reduce_sum(tf.multiply(t1, t2), axis=-1, keepdims=True)


def positional_encoding(
        num_units,
        batch_size,
        max_attempts,
        zero_pad=True,
        scale=True,
        scope="positional_encoding",
        reuse=None):
    '''Sinusoidal Positional_Encoding.
    Args:
      inputs: A 2d Tensor with shape of (N, T).
      num_units: Output dimensionality
      zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
        A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
    '''

    N = batch_size
    T = max_attempts

    position_ind = tf.range(T)
    position_enc = np.array([
        [pos / np.power(10000, 2. * i / num_units) for i in range(num_units)]
        for pos in range(T)], dtype=np.float32)

    # Second part, apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

    # Convert to a tensor
    lookup_table = tf.convert_to_tensor(position_enc)

    if zero_pad:
        lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                  lookup_table[1:, :]), 0)
    outputs = tf.tile(tf.expand_dims(lookup_table, 0), [N, 1, 1])

    if scale:
        outputs = outputs * num_units ** 0.5

    return outputs


class Positions(Layer):
    def __init__(self, **kwargs):
        super(Positions, self).__init__(**kwargs)

    def call(self, inputs, mask=None, **kwargs):
        # Range starting from one so that 0 won't become -1 in next operation
        positions = tf.map_fn(lambda x: tf.range(1, tf.shape(x)[0] + 1), inputs, dtype=tf.int32)
        # Add padded -1s to positions, natural numbers are decreased by one and others will become -1
        positions = positions * tf.cast(tf.not_equal(inputs, -1), dtype=tf.int32) - 1
        return tf.cast(positions, tf.float32)

    def compute_output_shape(self, input_shape):
        return input_shape
