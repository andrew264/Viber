import datetime
import multiprocessing
import os
import re
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import pad_sequences
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert

from bert_tokenizer import BERTTokenizer

CONTRACTIONS: tuple[tuple[str, str], ...] = \
    (("i'm", "I am"), ("he's", "he is"), ("she's", "she is"), ("it's", "it is"), ("that's", "that is"),
     ("what's", "what is"), ("when's", "when is"), ("where's", "where is"), ("how's", "how is"), ("who's", "who is"),
     ("ain't", "is not"), ("can't", "can not"), ("won't", "will not"), ("n't", " not"), ("ya'll", "you all"),
     ("y'all", "you all"), ("'ll", " will"), ("'cause", "because"), ("'em", "them"), ("'til", "until"),
     ("'ve", " have"), ("'re", " are"), ("'d", " would"), ("in'", "ing"), ("'bout", "about"), ("there's", "there is"),
     ("'cause", "because"), ("cuz", "because"), ("in'", "ing"), ("let's", "let us"), ("y'know", "you know"),
     ("'round", "around"), ("gon'", "gonna"), ("lil'", "little"), ("yo'", "your"), ("'fore", "before"),
     ("wit'", "with"),
     ("hol'", "hold"), ("here's", "here is"), ("one's", "one is"), ("life's", "life is"), ("you's", "you"),
     ("love's", "love is"), ("ing's", "ing is"), ("c'mon", "come on"), ("ol'", "old"), (" im ", " i am "),
     ("shoulda", "should have"), ("I'mma", "I am going to"))


def create_tokenizer(corpus, num_words=None) -> BERTTokenizer:
    if os.path.exists("./vocab.txt"):
        print("Using existing vocabulary...")
    else:
        print("Creating Vocabulary...")

        reserved_tokens = ["[PAD]", "[UNK]"]
        bert_vocab_args = dict(
            vocab_size=num_words or (2 ** 13),
            reserved_tokens=reserved_tokens,
            bert_tokenizer_params=dict(lower_case=False),
            learn_params={},
        )
        dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(corpus)
        dataset.batch(1024)
        vocab = bert.bert_vocab_from_dataset(dataset, **bert_vocab_args)
        del dataset
        with open("./vocab.txt", 'w') as vocab_file:
            for token in vocab:
                print(token, file=vocab_file)
    return BERTTokenizer("./vocab.txt")


def create_lyrics_corpus(dataset: pd.DataFrame, field: str):
    lyrics_list = []
    for index, row in dataset.iterrows():
        lyrics = row[field]
        if isinstance(lyrics, str):
            lyrics_list.extend(cleanup_lyrics(lyrics))

    return lyrics_list


def cleanup_lyrics(lyrics: str) -> list[str]:
    NEWLINE = "\n"
    for (contraction, expansion) in CONTRACTIONS:
        lyrics = re.sub(contraction, expansion, lyrics, flags=re.IGNORECASE)

    # Keep space, newlines, Letters, and select punctuation.
    lyrics = re.sub('[^ a-zA-Z0-9\n.?!,¿/-]', '', lyrics)
    # Add spaces around punctuation.
    lyrics = re.sub(r"([?.!,¿/-])", r" \1 ", lyrics)

    list_of_lyrics = []
    for lyric in lyrics.split("\n\n"):
        lyric = lyric.strip()
        if not lyric:
            continue

        lyric = lyric.replace("\n", NEWLINE)
        # check if the lyric more than 1 line
        if NEWLINE in lyric:
            list_of_lyrics.append(lyric)
        else:
            if not list_of_lyrics:
                list_of_lyrics.append(lyric)
            else:
                list_of_lyrics[-1] += NEWLINE + lyric

    return list_of_lyrics


def _corpus_to_ngram(corpus) -> list:
    ngram_seqs_list = []
    for line in corpus:
        for j in range(1, len(line)):
            n_gram_sequence = line[:j + 1]
            ngram_seqs_list.append(n_gram_sequence)
    return ngram_seqs_list


def create_sequence(tokenized_corpus: tf.RaggedTensor, max_seq_len: int) -> tuple[tf.Tensor, int]:
    """
    Create Padded Sequence from Tokenized Corpus
    Returns padded_sequence, max_sequence_length
    """
    sequences = []
    print("Creating Sequences...")
    start = datetime.datetime.now()
    # split tokenized_corpus into cpu_count parts
    cpu_count = multiprocessing.cpu_count()
    split_corpus = np.array_split(tokenized_corpus.numpy(), cpu_count)

    with multiprocessing.Pool(cpu_count) as pool:
        sequences.extend(pool.map(_corpus_to_ngram, split_corpus))

    sequences = [item for sublist in sequences for item in sublist]

    print(f"Created {len(sequences)} sequences in {(datetime.datetime.now() - start).seconds} seconds")
    print("Padding Sequences...")
    start = datetime.datetime.now()
    padded_sequence = pad_sequences(sequences, maxlen=max_seq_len)
    print(f"Padded {len(padded_sequence)} sequences in {(datetime.datetime.now() - start).seconds} seconds")
    del sequences, start
    return padded_sequence


def make_sequences_and_labels_from_df(df: pd.DataFrame, tokenizer: BERTTokenizer,
                                      max_seq_len: int) -> Tuple[Tuple, Tuple]:
    print("Creating Corpus...")
    start = datetime.datetime.now()
    corpus = create_lyrics_corpus(df, "lyrics")
    print(f"Tokenizing {len(corpus)} lines of lyrics")
    tokenized_corpus = tokenizer.tokenize(corpus)
    del corpus
    print(f"Tokenized Corpus in {(datetime.datetime.now() - start).seconds} seconds")
    sequences = create_sequence(tokenized_corpus, max_seq_len)
    del tokenized_corpus
    return sequences[:, :-1], sequences[:, -1]


def generate_dataset_from_sequences(input_sequences: Tuple, labels: Tuple, vocab_size: int) -> [tf.data.Dataset]:
    """yields Dataset in portions from a pandas DataFrame"""
    print("Creating Dataset...")
    seqs_per_dataset = 512 * 512  # batch_size * no_of_batches per dataset
    datasets = []
    with tf.device("/cpu:0"):
        for i in range(0, len(input_sequences), seqs_per_dataset):
            d = tf.data.Dataset.from_tensor_slices((input_sequences[i:i + seqs_per_dataset],
                                                    tf.one_hot(labels[i: i + seqs_per_dataset],
                                                               depth=vocab_size))).batch(512)
            tf.keras.backend.clear_session()
            datasets.append(d)
    del input_sequences, labels
    return datasets
