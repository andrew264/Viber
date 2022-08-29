from keras import Sequential
from keras.layers import Embedding, LSTM, Bidirectional, Dense
from keras.losses import CategoricalCrossentropy


def create_model(max_sequence_len: int, total_words: int) -> Sequential:
    """
    Create a model for training.
    """
    model = Sequential()
    model.add(Embedding(total_words, 64, input_length=max_sequence_len - 1))
    model.add(Bidirectional(LSTM(20)))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss=CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
    return model
