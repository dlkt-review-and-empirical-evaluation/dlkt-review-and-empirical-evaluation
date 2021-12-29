from tensorflow.keras.layers import LayerNormalization, add, Dense, Dropout

from layers.common import skill_outputs, embeddings, Positions, CustomMaskEmbedding
from layers.transformer import MultiHeadAttention


def sakt(key_inputs,
         val_inputs,
         recurrent_layer_size,
         key_embedding_size,
         val_embedding_size,
         n_skills,
         max_attempt_count,
         onehot=True,
         skill_layer_size=None,
         n_blocks=1,
         n_heads=5,
         dropout=.2,
         residual=True,
         ln_train=True):
    key_embeddings = embeddings(key_inputs, onehot=onehot,
                                dim=n_skills, layer_size=key_embedding_size)
    val_embeddings = embeddings(val_inputs, onehot=onehot,
                                dim=2 * n_skills + 1, layer_size=val_embedding_size)

    if not onehot:
        positions = Positions()(val_inputs)
        pos_embeddings = CustomMaskEmbedding(max_attempt_count, val_embedding_size,
                                             mask_value=-1., name='pos_embed')(positions)
        # pos_embeddings = positional_encoding(recurrent_layer_size, batch_size, max_attempt_count)
        pos_val_embeddings = add([val_embeddings, pos_embeddings])
    else:
        pos_val_embeddings = val_embeddings

    mha = MultiHeadAttention(recurrent_layer_size, n_heads, causal=True, dropout=dropout)(
        key_embeddings, pos_val_embeddings)
    if residual and not onehot and val_embedding_size == recurrent_layer_size:
        mha = add([mha, pos_val_embeddings])
    norm_mha = LayerNormalization(trainable=ln_train)(mha)

    for i in range(n_blocks - 1):
        mha = MultiHeadAttention(recurrent_layer_size, n_heads, causal=True, dropout=dropout)(
            key_embeddings, norm_mha)
        if residual and not onehot:
            mha = add([mha, norm_mha])
        norm_mha = LayerNormalization(trainable=ln_train)(mha)

    relu = Dropout(dropout)(Dense(recurrent_layer_size, activation='relu')(norm_mha))
    linear = Dropout(dropout)(Dense(recurrent_layer_size, activation='linear')(relu))
    if residual:
        linear = add([linear, norm_mha])
    norm_ffn = LayerNormalization(trainable=ln_train)(linear)

    outputs = skill_outputs(norm_ffn,
                            skill_layer_size=skill_layer_size,
                            n_skills=n_skills)

    return outputs
