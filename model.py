from abc import ABC

import tensorflow as tf
from keras.layers import StringLookup


class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units, dropout_rate=0.1):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, name="embedding")
        self.lstm1 = tf.keras.layers.LSTM(rnn_units,
                                          return_sequences=True,
                                          return_state=True,
                                          name="lstm_1")
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.lstm2 = tf.keras.layers.LSTM(rnn_units,
                                          return_sequences=True,
                                          return_state=True,
                                          name="lstm_2")
        self.dense = tf.keras.layers.Dense(vocab_size, name="dense_1")

    @tf.function
    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            state1, state2 = self.lstm1.get_initial_state(x)
            state3, state4 = self.lstm2.get_initial_state(x)
            states = (state1, state2, state3, state4)
        x, state1, state2 = self.lstm1(x, initial_state=states[:2], training=training)
        x = self.dropout(x, training=training)
        x, state3, state4 = self.lstm2(x, initial_state=states[2:], training=training)
        states = (state1, state2, state3, state4)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


class OneStep(tf.keras.Model, ABC):
    def __init__(self, model: MyModel,
                 chars_from_ids: StringLookup,
                 ids_from_chars: StringLookup,
                 temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # Create a mask to prevent "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put an -inf at each bad index.
            values=[-float('inf')] * len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)
        inputs = tf.TensorSpec(shape=[None], dtype=tf.string)
        states = [tf.TensorSpec(shape=[None, None], dtype=tf.float32, name='state1'),
                  tf.TensorSpec(shape=[None, None], dtype=tf.float32, name='state2'),
                  tf.TensorSpec(shape=[None, None], dtype=tf.float32, name='state3'),
                  tf.TensorSpec(shape=[None, None], dtype=tf.float32, name='state4')]
        self.generate_one_step.get_concrete_function(inputs, states)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs.
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(inputs=input_ids, states=states,
                                              return_state=True)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits / self.temperature
        # Apply the prediction mask: prevent "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states
