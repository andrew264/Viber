import datetime
import gc
import math
import os.path

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.pyplot import ylabel, plot, show, xlabel

from bert_tokenizer import BERTTokenizer
from model import create_model
from preprocessing import create_lyrics_corpus, create_tokenizer, make_sequences_and_labels_from_df
from run_model import run

DELIMITER = "|"
max_seq_len = 64
batch_size = 256
songs_per_dataset = 512


def plot_graphs(_history, string):
    plot(_history.history[string])
    xlabel("Epochs")
    ylabel(string)
    show()


if __name__ == '__main__':
    # # tf memory management
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    # tf.config.run_functions_eagerly(False)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # load data
    dataset_df = pd.read_csv('dataset.csv', dtype=str, delimiter=DELIMITER).sample(frac=1)
    print("Dataset Loaded")
    # Tokenizer
    if os.path.exists('vocab.txt'):
        print("Using existing vocabulary...")
        tokenizer = BERTTokenizer('vocab.txt')
    else:
        # Create the corpus using the 'lyrics' column containing lyrics
        corpus = create_lyrics_corpus(dataset_df, 'lyrics')
        tokenizer = create_tokenizer(corpus)
        del corpus

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./checkpoints/weights_{epoch:02d}', monitor='accuracy', save_weights_only=True, )
    # early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch='500,510')

    # Start training
    print("Starting Training...")

    model = create_model(max_seq_len, tokenizer.get_vocab_size())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.build(input_shape=(None, max_seq_len - 1))
    start_time = datetime.datetime.now()

    # Split dataset
    no_of_parts = math.ceil(len(dataset_df) / songs_per_dataset)
    datasets = np.array_split(dataset_df, no_of_parts)
    del dataset_df

    for i in range(0, len(datasets)):
        gc.collect()
        tf.keras.backend.clear_session()
        print(f"Training on dataset {i + 1} / {len(datasets)}")
        input_sequences, labels = make_sequences_and_labels_from_df(datasets[i], tokenizer, max_seq_len)
        with tf.device('/CPU:0'):
            dataset = tf.data.Dataset.from_tensor_slices(
                (input_sequences, tf.one_hot(labels, depth=tokenizer.get_vocab_size()))).batch(batch_size)
        del input_sequences, labels
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        history = model.fit(dataset, epochs=2, verbose=1,
                            callbacks=[checkpoint_callback, ])

    print(f"Training Completed in {(datetime.datetime.now() - start_time).min} minutes")

    plot_graphs(history, 'accuracy')

    run(tokenizer, max_seq_len)
