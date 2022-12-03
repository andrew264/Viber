import datetime
import os
import re

import pandas as pd
import tensorflow as tf
from keras.utils import pad_sequences
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert

from bert_tokenizer import BERTTokenizer

CONTRACTIONS: tuple[tuple[str, str], ...] = \
    (("i'm", "i am"), ("he's", "he is"), ("she's", "she is"), ("it's", "it is"), ("that's", "that is"),
     ("what's", "what is"), ("when's", "when is"), ("where's", "where is"), ("how's", "how is"), ("who's", "who is"),
     ("ain't", "is not"), ("can't", "can not"), ("won't", "will not"), ("n't", " not"), ("ya'll", "you all"),
     ("y'all", "you all"), ("'ll", " will"), ("'cause", "because"), ("'em", "them"), ("'til", "until"),
     ("'ve", " have"), ("'re", " are"), ("'d", " would"), ("in'", "ing"), ("'bout", "about"), ("there's", "there is"),
     ("'cause", "because"), ("cuz", "because"), ("in'", "ing"), ("let's", "let us"), ("y'know", "you know"),
     ("'round", "around"), ("gon'", "gonna"), ("lil'", "little"), ("yo'", "your"), ("'fore", "before"), ("wit'", "with"),
     ("hol'", "hold"), ("here's", "here is"), ("one's", "one is"), ("life's", "life is"), ("you's", "you"),
     ("love's", "love is"), ("ing's", "ing is"), ("c'mon", "come on"), ("ol'", "old"), ("im", "i am"),
     ("shoulda", "should have"))


def create_tokenizer(corpus, num_words=None) -> BERTTokenizer:
    if os.path.exists("./vocab.txt"):
        print("Using existing vocabulary...")
    else:
        print("Creating Vocabulary...")

        reserved_tokens = ["[PAD]", "[UNK]", ]
        bert_vocab_args = dict(
            vocab_size=num_words or (2 ** 13),
            reserved_tokens=reserved_tokens,
            bert_tokenizer_params=dict(lower_case=True),
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
    # Make it lowercase
    dataset[field] = dataset[field].str.lower()
    # Make it one long string to split by line
    lyrics = dataset[field].str.cat()

    return cleanup_lyrics(lyrics)


def cleanup_lyrics(lyrics: str) -> list[str]:
    for (contraction, expansion) in CONTRACTIONS:
        lyrics = re.sub(contraction, expansion, lyrics)

    lyrics = lyrics.lower().splitlines()
    corpus = []
    for line in lyrics:
        if line.strip() == "":
            continue
        # replace - with space
        # Keep space, a to z, and select punctuation.
        line = re.sub('[^ a-z.?!,¿/-]', '', line)
        # Add spaces around punctuation.
        line = re.sub(r"([?.!,¿/-])", r" \1 ", line)
        corpus.append(line)

    return corpus


def create_sequence(tokenized_corpus: tf.RaggedTensor, max_seq_len) -> tuple[tf.Tensor, int]:
    """
    Create Padded Sequence from Tokenized Corpus
    Returns padded_sequence, max_sequence_length
    """
    sequences = []
    print("Creating Sequences...")
    start = datetime.datetime.now()
    for line in tokenized_corpus:
        for j in range(1, len(line)):
            n_gram_sequence = line[:j + 1]
            sequences.append(n_gram_sequence)

    print(f"Created {len(sequences)} sequences in {(datetime.datetime.now() - start).seconds} seconds")
    print("Padding Sequences...")
    start = datetime.datetime.now()
    padded_sequence = pad_sequences(sequences, maxlen=max_seq_len)
    print(f"Padded {len(padded_sequence)} sequences in {(datetime.datetime.now() - start).seconds} seconds")
    del sequences, start
    return padded_sequence


def create_dataset_from_df(df: pd.DataFrame, tokenizer: BERTTokenizer, max_seq_len: int) -> [tf.data.Dataset]:
    """Create a Dataset from a pandas DataFrame"""
    print("Creating Corpus...")
    start = datetime.datetime.now()
    corpus = create_lyrics_corpus(df, "lyrics")
    print(f"Tokenizing {len(corpus)} lines of lyrics")
    tokenized_corpus = tokenizer.tokenize(corpus)
    print(f"Tokenized Corpus in {(datetime.datetime.now() - start).seconds} seconds")
    print(f"Padding {tokenized_corpus.shape[0]} lines of lyrics")
    sequences = create_sequence(tokenized_corpus, max_seq_len)
    input_sequences, labels = sequences[:, :-1], sequences[:, -1]
    del sequences, start
    print("Creating Dataset...")
    datasets: [tf.data.Dataset] = []
    seqs_per_dataset = 512 * 512  # batch_size * no_of_batches per dataset
    with tf.device("/cpu:0"):
        for i in range(0, len(input_sequences), seqs_per_dataset):
            d = tf.data.Dataset.from_tensor_slices((input_sequences[i:i + seqs_per_dataset],
                                                    tf.one_hot(labels[i: i + seqs_per_dataset],
                                                               depth=tokenizer.get_vocab_size()))).batch(512)
            datasets.append(d)
            tf.keras.backend.clear_session()
    return datasets
