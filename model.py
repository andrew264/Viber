from keras import Sequential
from keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from keras.losses import CategoricalCrossentropy

# Dropouts are used to prevent overfitting
dropout_rate = 0.2

# The embedding dimension
embedding_dim = 128

# Number of RNN units
rnn_units = 512


def create_model(max_sequence_len: int, total_words: int) -> Sequential:
    """
    Create a model for training.
    """
    model = Sequential()
    model.add(Embedding(total_words, embedding_dim, input_length=max_sequence_len - 1))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(rnn_units, return_sequences=True)))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(rnn_units)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss=CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
    return model


def load_model():
    return None
