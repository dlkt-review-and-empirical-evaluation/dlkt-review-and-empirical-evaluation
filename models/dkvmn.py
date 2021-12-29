from tensorflow.keras.layers import Dense, RNN, TimeDistributed, concatenate
# from tensorflow_core.python.keras.layers import Dense, RNN, TimeDistributed, concatenate

from layers.dkvmn import KeyMemory, ReadWriteMemoryCell, NestedInput
from layers.common import skill_outputs, embeddings


def dkvmn(key_inputs,
          val_inputs,
          recurrent_layer_size,
          key_embedding_size,
          val_embedding_size,
          n_skills,
          batch_size=32,
          onehot=True,
          paper_version=False,  # github MXNet version is most of the time better
          skill_layer_size=None,
          dropout=.2):
    key_embeddings = embeddings(key_inputs, onehot=onehot,
                                dim=n_skills, layer_size=key_embedding_size)
    val_embeddings = embeddings(val_inputs, onehot=onehot,
                                dim=2 * n_skills, layer_size=key_embedding_size)

    if paper_version:
        correlation_weights = TimeDistributed(KeyMemory(recurrent_layer_size, name='key_memory'))(key_embeddings)
    else:
        correlation_weights = TimeDistributed(Dense(recurrent_layer_size, activation='softmax',
                                                    name='key_memory'))(key_embeddings)

    erase_vectors = TimeDistributed(Dense(val_embedding_size, activation='sigmoid', name='erase_vectors'))(
        val_embeddings)
    add_vectors = TimeDistributed(Dense(val_embedding_size, activation='tanh', name='add_vectors'))(
        val_embeddings)

    correlation_weights.set_shape([batch_size] + correlation_weights.shape[1:])
    add_vectors.set_shape([batch_size] + add_vectors.shape[1:])
    erase_vectors.set_shape([batch_size] + erase_vectors.shape[1:])

    rnn_cell = ReadWriteMemoryCell(recurrent_layer_size, key_embedding_size, val_embedding_size,
                                   dropout=dropout, recurrent_dropout=dropout)
    rnn = RNN(rnn_cell, return_sequences=True, name='value_memory')

    student_mastery_levels = rnn(NestedInput(correlation_weights=correlation_weights,
                                             erase_vectors=erase_vectors,
                                             add_vectors=add_vectors))

    if paper_version:
        concatted = concatenate([key_embeddings, student_mastery_levels], axis=-1)
    else:
        key_vector_layer = TimeDistributed(Dense(key_embedding_size, activation='tanh', name='key_vector'))
        key_vectors = key_vector_layer(key_embeddings)
        key_vectors.set_shape([batch_size] + key_vectors.shape[1:])
        concatted = concatenate([key_vectors, student_mastery_levels], axis=-1)
    outputs = skill_outputs(concatted,
                            skill_layer_size=skill_layer_size,
                            n_skills=n_skills, dropout=dropout)
    return outputs
