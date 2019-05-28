import keras.backend as K
import tensorflow as tf


# Huber loss for DQN
def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta
    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
    return K.mean(tf.where(cond, squared_loss, quadratic_loss))
