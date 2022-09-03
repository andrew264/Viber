import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import pad_sequences
from matplotlib.pyplot import ylabel, plot, show, xlabel

from model import create_model
from preprocessing import create_lyrics_corpus, create_tokenizer, create_sequence, create_tokenized_corpus

DELIMITER = "|"

if __name__ == '__main__':
    dataset_df = pd.read_csv('dataset.csv', dtype=str, delimiter=DELIMITER)
    # Create the corpus using the 'text' column containing lyrics
    corpus = create_lyrics_corpus(dataset_df, 'lyrics')
    del dataset_df
    # Tokenize the corpus
    tokenizer = create_tokenizer(corpus)

    total_words = tokenizer.get_vocab_size()
    print(total_words)

    tokenized_corpus = create_tokenized_corpus(tokenizer, corpus)
    sequences, max_sequence_length = create_sequence(tokenized_corpus)
    print(f"Total Sequences: {tf.shape(sequences)[0]}")
    print(f"Max Sequence length: {max_sequence_length}")
    del corpus, tokenized_corpus

    # Split sequences between the "input" sequence and "output" predicted word
    input_sequences, labels = sequences[:, :-1], sequences[:, -1]
    # One-hot encode the labels
    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=total_words)

    model = create_model(max_sequence_length, total_words)
    dataset = tf.data.Dataset.from_tensor_slices((input_sequences, one_hot_labels))
    dataset = dataset.batch(512)
    del input_sequences, one_hot_labels, sequences, labels

    history = model.fit(dataset, epochs=50, verbose=1)


    def plot_graphs(history, string):
        plot(history.history[string])
        xlabel("Epochs")
        ylabel(string)
        show()


    plot_graphs(history, 'accuracy')

    next_words = 50

    while True:
        seed_text = input("Enter seed text: ")
        if seed_text == '':
            break
        token_list = []
        token_list.extend(tokenizer.tokenize(seed_text)[0].numpy().tolist())
        for i in range(next_words):
            tokens = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding='pre')
            predicted_probs = model.predict(tokens)[0]
            predicted = np.random.choice([x for x in range(len(predicted_probs))],
                                         p=predicted_probs)
            token_list.append(predicted)
        print("Output: " + tokenizer.detokenize([token_list])[0].numpy().decode())
