from tensorflow.python.keras.layers import LayerNormalization, add, Dense, Dropout

from layers.common import skill_outputs, embeddings, Positions, CustomMaskEmbedding
from layers.transformer import MultiHeadAttention


def transformer(key_inputs,
                val_inputs,
                recurrent_layer_size,
                key_embedding_size,
                val_embedding_size,
                n_skills,
                max_attempt_count,
                skill_layer_size=None,
                onehot=False,
                n_heads=5,
                n_blocks=2,
                dropout=.2,
                residual=True,
                ln_train=True):
    assert not onehot  # onehot is not supported

    def transformer_ffn(x):
        relu = Dropout(dropout)(Dense(recurrent_layer_size, activation='relu')(x))
        linear = Dropout(dropout)(Dense(recurrent_layer_size, activation='linear')(relu))
        return linear

    def transformer_block(key_embeddings, val_embeddings, n_heads=1):
        norm_layer = LayerNormalization(trainable=ln_train)
        mha = MultiHeadAttention(recurrent_layer_size, n_heads, causal=True, dropout=dropout)(
            key_embeddings, val_embeddings)
        norm_mha = norm_layer(mha)
        return norm_layer(transformer_ffn(norm_mha))

    positions = Positions()(val_inputs)
    key_embeddings = embeddings(key_inputs, onehot=onehot,
                                dim=n_skills, layer_size=recurrent_layer_size)
    val_embeddings = embeddings(val_inputs, onehot=onehot,
                                dim=2 * n_skills + 1, layer_size=recurrent_layer_size)

    pos_embeddings = CustomMaskEmbedding(max_attempt_count, recurrent_layer_size,
                                         mask_value=-1., name='pos_embed')(positions)

    pos_val_embeddings = add([val_embeddings, pos_embeddings])
    pos_key_embeddings = add([key_embeddings, pos_embeddings])

    encoder_block = transformer_block(pos_val_embeddings, pos_val_embeddings, n_heads=n_heads)
    for _ in range(n_blocks - 1):
        encoder_block = transformer_block(encoder_block, encoder_block, n_heads=n_heads, )

    decoder_attention = MultiHeadAttention(recurrent_layer_size, n_heads, causal=True)(
        pos_key_embeddings, pos_key_embeddings)
    decoder_block = transformer_block(decoder_attention, encoder_block, n_heads=n_heads)
    for _ in range(n_blocks - 1):
        decoder_block = transformer_block(decoder_attention, encoder_block, n_heads=n_heads)
    outputs = skill_outputs(decoder_block,
                            skill_layer_size=skill_layer_size,
                            n_skills=n_skills)

    return outputs
