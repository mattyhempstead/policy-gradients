import tensorflow as tf
from tensorflow import keras

from loss import loss

class Model:
    def __init__(self):
        self.model = keras.Sequential()

        self.model.add(keras.layers.Flatten(input_shape=(4,)))
        self.model.add(keras.layers.Dense(8, activation='relu'))
        self.model.add(keras.layers.Dense(16, activation='relu'))
        self.model.add(keras.layers.Dense(8, activation='relu'))
        self.model.add(keras.layers.Dense(2, activation=None))

        self.model.compile(
            optimizer = tf.keras.optimizers.Adam(),
            loss = loss,
            metrics = [],
        )

    def select_action(self, logits):
        # `tf.random.categorical` returns index of randomly selected value after applying softmax to logits
        # `tf.squeeze` strips tensor of all single dimentions (in this case returns scaler)
        return tf.squeeze(tf.random.categorical(logits, 1))

    def predict(self, inputs):
        return self.model.predict(inputs)
    