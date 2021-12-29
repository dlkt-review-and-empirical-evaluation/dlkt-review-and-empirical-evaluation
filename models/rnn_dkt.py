from tensorflow.keras.layers import LSTM, SimpleRNNCell, RNN

from layers.common import skill_outputs, embeddings


def get_rnn_layer(version, layer_size, dropout=.2):
    if version == 'lstm':
        return LSTM(layer_size, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)
    elif version == 'vanilla':
        vanilla_rnn_cell = SimpleRNNCell(layer_size, activation='tanh', dropout=dropout,
                                         recurrent_dropout=dropout)
        return RNN(vanilla_rnn_cell, return_sequences=True)
    else:
        raise NotImplementedError(f'{version} rnn is not implemented.')


def rnn_dkt(kernel_type,
            val_inputs,
            recurrent_layer_size,
            n_skills,
            onehot=True,
            embedding_size=None,
            skill_layer_size=None,
            dropout=.2):
    embedded_inputs = embeddings(val_inputs, onehot=onehot,
                                 dim=2 * n_skills + 1, layer_size=embedding_size)

    recurrent_outputs = get_rnn_layer(kernel_type, recurrent_layer_size, dropout)(embedded_inputs)

    outputs = skill_outputs(recurrent_outputs,
                            skill_layer_size=skill_layer_size,
                            n_skills=n_skills, dropout=dropout)
    return outputs
