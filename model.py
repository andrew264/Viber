from typing import Optional

import keras
from keras import Sequential
from keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from keras.losses import CategoricalCrossentropy


class MyModel(keras.Model):
    def __init__(self, max_sequence_len: int, total_words: int, rnn_units: int = 512,
                 embedding_dim: int = 128, dropout_rate: float = 0.2):
        super(MyModel, self).__init__()
        self.max_sequence_len = max_sequence_len - 1
        self.total_words = total_words
        self.rnn_units = rnn_units
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim
        self.embedding = Embedding(self.total_words, self.embedding_dim, input_length=self.max_sequence_len)
        self.dropout = Dropout(self.dropout_rate)
        self.lstm1 = Bidirectional(LSTM(self.rnn_units, return_sequences=True))
        self.lstm2 = Bidirectional(LSTM(self.rnn_units))
        self.dense = Dense(self.total_words, activation='softmax')

    def call(self, inputs, training: Optional[bool] = None):
        x = self.embedding(inputs)
        mask = self.embedding.compute_mask(inputs)
        x = self.dropout(x, training=training)
        x = self.lstm1(x, mask=mask, training=training)
        x = self.dropout(x, training=training)
        x = self.lstm2(x, mask=mask, training=training)
        x = self.dropout(x, training=training)
        return self.dense(x)

    def get_config(self):
        return dict(max_sequence_len=self.max_sequence_len,
                    total_words=self.total_words,
                    rnn_units=self.rnn_units,
                    embedding_dim=self.embedding_dim,
                    dropout_rate=self.dropout_rate
                    )

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def create_model(max_sequence_len: int, total_words: int) -> Sequential:
    """
    Create a model for training.
    """
    # return MyModel(max_sequence_len, total_words)
    # Dropouts are used to prevent overfitting
    dropout_rate = 0.1

    # The embedding dimension
    embedding_dim = 256

    # Number of RNN units
    rnn_units = 1024

    model = Sequential()
    model.add(Embedding(total_words, embedding_dim, input_length=max_sequence_len - 1, mask_zero=True))
    model.add(Bidirectional(LSTM(rnn_units, return_sequences=True)))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(rnn_units)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss=CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
    return model


def load_model():
    return None
