from tensorflow.keras.initializers import glorot_uniform  # Or your initializer of choice
import tensorflow as tf
import tensorflow.keras.backend as K


def get_init_weights(model):
    initial_weights = model.get_weights()
    if int(tf.__version__[0]) < 2:
        return [glorot_uniform()(w.shape).eval(session=K.get_session()) if w.ndim > 1 else w for w in initial_weights]
    if not tf.executing_eagerly():
        return [glorot_uniform()(w.shape).eval(session=tf.compat.v1.Session()) if w.ndim > 1 else w for w in
                initial_weights]

    return [glorot_uniform()(w.shape).numpy() if w.ndim > 1 else w for w in initial_weights]
