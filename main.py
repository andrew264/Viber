import datetime
import gc
import os.path
from typing import Optional

import pandas as pd
import tensorflow as tf
from matplotlib.pyplot import ylabel, plot, show, xlabel

from bert_tokenizer import BERTTokenizer
from model import create_model
from preprocessing import create_lyrics_corpus, create_tokenizer, create_dataset_from_df
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
    dataset_df = pd.read_csv('dataset.csv', dtype=str, delimiter=DELIMITER)
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

    # creating the dataset
    start_time = datetime.datetime.now()
    dataset_dfs = [dataset_df[index:index + 200].copy() for index in range(0, len(dataset_df), 200)]
    datasets: list[Optional[tf.data.Dataset]] = [create_dataset_from_df(df, tokenizer, max_seq_len)
                                                 for df in dataset_dfs]
    del dataset_dfs
    print(f"Dataset Created in {(datetime.datetime.now() - start_time).seconds} seconds")

    # Start training
    print("Starting Training...")

    model = create_model(max_seq_len, tokenizer.get_vocab_size())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.build(input_shape=(None, max_seq_len - 1))
    start_time = datetime.datetime.now()
    for i, dataset in enumerate(datasets):
        print(f"Training on dataset {i + 1} of {len(datasets)}")

        history = model.fit(dataset, epochs=20, verbose=1,
                            callbacks=[checkpoint_callback, early_stop])

        tf.keras.backend.clear_session()
        datasets[i] = None
        gc.collect()
    del datasets
    print(f"Training Completed in {(datetime.datetime.now() - start_time).seconds} seconds")

    plot_graphs(history, 'accuracy')

    run(tokenizer, max_seq_len)
