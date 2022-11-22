import pandas as pd
import tensorflow as tf
from matplotlib.pyplot import ylabel, plot, show, xlabel

from model import create_model
from preprocessing import create_lyrics_corpus, create_tokenizer, create_sequence, create_tokenized_corpus
from run_model import run

DELIMITER = "|"
max_seq_len = 48

if __name__ == '__main__':
    dataset_df = pd.read_csv('dataset.csv', dtype=str, delimiter=DELIMITER).sample(n=250)
    # Create the corpus using the 'text' column containing lyrics
    corpus = create_lyrics_corpus(dataset_df, 'lyrics')
    del dataset_df
    # Tokenize the corpus
    tokenizer = create_tokenizer(corpus, num_words=(2 ** 12))

    total_words = tokenizer.get_vocab_size()
    print(total_words)

    tokenized_corpus = create_tokenized_corpus(tokenizer, corpus)
    sequences = create_sequence(tokenized_corpus, max_seq_len)
    print(f"Total Sequences: {tf.shape(sequences)[0]}")
    # print(f"Max Sequence length: {max_sequence_length}")
    del corpus, tokenized_corpus

    # Split sequences between the "input" sequence and "output" predicted word
    input_sequences, labels = sequences[:, :-1], sequences[:, -1]
    print(f"Input sequences shape: {tf.shape(input_sequences)}")
    print(f"Labels shape: {tf.shape(labels)}")
    # One-hot encode the labels
    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=total_words)

    model = create_model(max_seq_len, total_words)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.build(input_shape=(None, max_seq_len - 1))
    dataset = tf.data.Dataset.from_tensor_slices((input_sequences, one_hot_labels))
    dataset = dataset.batch(512, drop_remainder=True)
    del input_sequences, one_hot_labels, sequences, labels

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./checkpoints/weights_{epoch:02d}', monitor='accuracy', save_best_only=True, mode='max',
        save_weights_only=True, )
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

    history = model.fit(dataset, epochs=50, verbose=1,
                        callbacks=[checkpoint_callback, early_stop])


    def plot_graphs(_history, string):
        plot(_history.history[string])
        xlabel("Epochs")
        ylabel(string)
        show()


    plot_graphs(history, 'accuracy')

    run(tokenizer, max_seq_len)
