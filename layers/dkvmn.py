import collections

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin
# from tensorflow_core.python.keras.layers.recurrent import DropoutRNNCellMixin

NestedInput = collections.namedtuple('NestedInput',
                                     ['correlation_weights', 'erase_vectors', 'add_vectors'])
NestedState = collections.namedtuple('NestedState', ['erase_signal', 'add_signal'])


class KeyMemory(Layer):
    def __init__(self, memory_size, **kwargs):
        super(KeyMemory, self).__init__(**kwargs)
        self.memory_size = memory_size
        self.supports_masking = True

    def build(self, input_shapes):
        self.memory = self.add_weight('memory',
                                      shape=(self.memory_size, input_shapes[-1]),
                                      trainable=False,
                                      initializer='glorot_uniform',
                                      )

    def call(self, input, mask=None, **kwargs):
        embedding_result = tf.linalg.matvec(self.memory, tf.squeeze(input))
        return tf.reshape(tf.nn.softmax(embedding_result), [-1, self.memory_size])

    def compute_output_shape(self, input_shape):
        return input_shape[:-1], self.memory_size


class ReadWriteMemoryCell(DropoutRNNCellMixin, Layer):
    def __init__(self, memory_size, key_embedding_size, value_embedding_size, dropout, recurrent_dropout, **kwargs):
        self.value_embedding_size = value_embedding_size
        self.key_embedding_size = key_embedding_size
        self.output_size = key_embedding_size + value_embedding_size
        self.memory_size = memory_size
        self.state_size = tf.TensorShape([memory_size, value_embedding_size])
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        # self.state_size = NestedState(erase_signal=tf.TensorShape([memory_size, value_embedding_size]),
        #                               add_signal=tf.TensorShape([memory_size, value_embedding_size]))
        self.supports_masking = True
        super(ReadWriteMemoryCell, self).__init__(**kwargs)

    def build(self, input_shapes):
        self.value_memory = self.add_weight(
            shape=(self.memory_size, self.value_embedding_size), name='value_memory',
            initializer='glorot_uniform',
            # trainable=False
        )  # value memory

    def call(self, inputs, states, **kwargs):
        correlation_weights, erase_signal, add_signal = tf.nest.flatten(inputs)
        # prev_erase, prev_add = tf.nest.flatten(states)
        # self.value_memory = self.value_memory * prev_erase + prev_add  # This doesn't provide good results

        self.value_memory = tf.clip_by_norm(states[0],
                                            50)  # This needs checking whether affects results, 50-100 seems good

        # self.value_memory = states[0]  # Floats explode even with gradient clipping

        def read():
            cws_reshaped = tf.reshape(correlation_weights, [-1, 1, self.memory_size])
            return tf.reshape(K.batch_dot(cws_reshaped, self.value_memory), [-1, self.value_embedding_size])

        def write():
            erase_reshaped = tf.reshape(erase_signal, [-1, 1, self.value_embedding_size])
            add_reshaped = tf.reshape(add_signal, [-1, 1, self.value_embedding_size])
            attention_reshaped = tf.reshape(correlation_weights, [-1, self.memory_size, 1])

            weighted_erase_signal = 1 - K.batch_dot(attention_reshaped, erase_reshaped)
            weighted_add_signal = K.batch_dot(attention_reshaped, add_reshaped)
            # return weighted_erase_signal, weighted_add_signal
            return self.value_memory * weighted_erase_signal + weighted_add_signal

        read_state = read()
        # weighted_erase_signal, weighted_add_signal = write()
        return read_state, [write()]  # NestedState(erase_signal=weighted_erase_signal, add_signal=weighted_add_signal)
        # return read_state, NestedState(erase_signal=weighted_erase_signal, add_signal=weighted_add_signal)
