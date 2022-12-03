import datetime
import gc
import os.path
from typing import Optional

import pandas as pd
import tensorflow as tf
from matplotlib.pyplot import ylabel, plot, show, xlabel

from bert_tokenizer import BERTTokenizer
from model import create_model
from preprocessing import create_lyrics_corpus, create_tokenizer, make_sequences_and_labels_from_df, generate_dataset_from_sequences
from run_model import run

DELIMITER = "|"
max_seq_len = 48

# tf memory management
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


def plot_graphs(_history, string):
    plot(_history.history[string])
    xlabel("Epochs")
    ylabel(string)
    show()


if __name__ == '__main__':
    dataset_df = pd.read_csv('dataset.csv', dtype=str, delimiter=DELIMITER).sample(frac=1)
    print("Dataset Loaded")
    # Tokenizer
    if os.path.exists('vocab.txt'):
        print("Using existing vocabulary...")
        tokenizer = BERTTokenizer('vocab.txt')
    else:
        # Create the corpus using the 'lyrics' column containing lyrics
        print("Creating Vocabulary...")
        corpus = create_lyrics_corpus(dataset_df, 'lyrics')
        tokenizer = create_tokenizer(corpus)
        del corpus

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./checkpoints/weights_{epoch:02d}', monitor='accuracy', save_weights_only=True, )
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # creating input_sequences: Tuple, labels: Tuple
    input_sequences, labels = make_sequences_and_labels_from_df(dataset_df, tokenizer, max_seq_len)

    # Start training
    print("Starting Training...")

    model = create_model(max_seq_len, tokenizer.get_vocab_size())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.build(input_shape=(None, max_seq_len - 1))
    start_time = datetime.datetime.now()
    datasets = generate_dataset_from_sequences(input_sequences=input_sequences,
                                               labels=labels,
                                               vocab_size=tokenizer.get_vocab_size())
    for i, dataset in enumerate(datasets):
        dataset = dataset.prefetch(128)
        print(f"Training on dataset {i + 1} / {len(datasets)}")

        history = model.fit(dataset, epochs=10, verbose=1,
                            callbacks=[checkpoint_callback, early_stop, tensorboard_callback])

        tf.keras.backend.clear_session()
        datasets[i] = None
        gc.collect()
    del datasets
    print(f"Training Completed in {(datetime.datetime.now() - start_time).seconds} seconds")

    plot_graphs(history, 'accuracy')

    run(tokenizer, max_seq_len)
