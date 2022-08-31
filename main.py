import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import pad_sequences
from matplotlib.pyplot import ylabel, plot, show, xlabel

from model import create_model
from preprocessing import create_lyrics_corpus, tokenize_corpus

DELIMITER = "|"

if __name__ == '__main__':
    dataset_df = pd.read_csv('dataset.csv', dtype=str, delimiter=DELIMITER)[:180]
    # Create the corpus using the 'text' column containing lyrics
    corpus = create_lyrics_corpus(dataset_df, 'lyrics')
    # Tokenize the corpus
    tokenizer = tokenize_corpus(corpus)

    total_words = len(tokenizer.word_index) + 1

    print(tokenizer.word_index)
    print(total_words)

    sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            sequences.append(n_gram_sequence)

    # Pad sequences for equal input length
    max_sequence_len = max([len(seq) for seq in sequences])
    sequences: tf.Tensor = tf.convert_to_tensor(pad_sequences(sequences, maxlen=max_sequence_len, padding='pre'),
                                                dtype=tf.int8)

    # Split sequences between the "input" sequence and "output" predicted word
    input_sequences, labels = sequences[:, :-1], sequences[:, -1]
    # One-hot encode the labels
    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=total_words)

    model = create_model(max_sequence_len, total_words)
    dataset = tf.data.Dataset.from_tensor_slices((input_sequences, one_hot_labels))
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(512)
    del input_sequences, one_hot_labels, sequences, labels, corpus

    history = model.fit(dataset, epochs=100, verbose=1)


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
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
            predicted_probs = model.predict(token_list)[0]
            predicted = np.random.choice([x for x in range(len(predicted_probs))],
                                         p=predicted_probs)
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " " + output_word

        print("Output: " + seed_text)
