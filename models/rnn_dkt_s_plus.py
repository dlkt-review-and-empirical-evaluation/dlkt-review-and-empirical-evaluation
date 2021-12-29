from tensorflow.keras.layers import Dense, Dropout, TimeDistributed, concatenate

from layers.common import skill_outputs, embeddings
from models.rnn_dkt import get_rnn_layer


def rnn_dkt_s_plus(key_inputs,
         val_inputs,
         recurrent_layer_size,
         key_embedding_size,
         val_embedding_size,
         n_skills,
         onehot=True,
         rnn_version='lstm',
         skill_layer_size=None,
         dropout=.2):
    key_embeddings = embeddings(key_inputs, onehot=onehot,
                                dim=n_skills, layer_size=key_embedding_size)
    val_embeddings = embeddings(val_inputs, onehot=onehot,
                                dim=2 * n_skills + 1, layer_size=val_embedding_size)

    key_vectors = Dropout(dropout)(TimeDistributed(Dense(key_embedding_size))(key_embeddings))
    combined = concatenate([key_vectors, val_embeddings], axis=-1)
    #combined = add([key_vectors, val_embeddings])

    recurrent_outputs = get_rnn_layer(rnn_version, recurrent_layer_size, dropout)(combined)

    combined = concatenate([key_vectors, recurrent_outputs], axis=-1)
    #combined = add([recurrent_outputs, key_vectors, val_embeddings])

    outputs = skill_outputs(combined,
                            skill_layer_size=skill_layer_size,
                            n_skills=n_skills)

    return outputs
