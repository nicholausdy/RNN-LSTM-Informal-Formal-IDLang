# Code is modified from the link below
# Source: https://www.tensorflow.org/text/tutorials/text_classification_rnn

import tensorflow as tf
from tensorflow import keras

class RNNLSTM():
    def __init__(self, vocab_size = 660, embed_dim = 32):
        inputs = tf.keras.layers.Input(shape=(vocab_size,))
        x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embed_dim))(x)
        x = tf.keras.layers.Dense(embed_dim, activation="relu")(x)
        x = tf.keras.layers.Dropout(rate = 0.1)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
