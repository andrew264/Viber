import os
import re

import tensorflow as tf
from keras.utils import pad_sequences
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert

from bert_tokenizer import BERTTokenizer

CONTRACTIONS = (("i'm", "i am"), ("he's", "he is"), ("she's", "she is"), ("it's", "it is"), ("that's", "that is"),
                ("what's", "what is"), ("when's", "when is"), ("where's", "where is"), ("how's", "how is"),
                ("ain't", "is not"), ("can't", "can not"), ("won't", "will not"), ("n't", " not"), ("'ll", " will"),
                ("'ve", " have"), ("'re", " are"), ("'d", " would"), ("n'", "ng"), ("'bout", "about"),
                ("'cause", "because"), ("cuz", "because"), ("in'", "ing"))


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


def create_lyrics_corpus(dataset, field):
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
        # Keep space, a to z, and select punctuation.
        line = re.sub('[^ a-z.?!,¿]', '', line)
        # Add spaces around punctuation.
        line = re.sub(r"([?.!,])", r" \1 ", line)
        if len(line.split()) > 50:
            line = ''.join(line.split()[:50])
        corpus.append(line)

    return corpus


def create_sequence(tokenized_corpus: tf.RaggedTensor) -> tuple[tf.Tensor, int]:
    """
    Create Padded Sequence from Tokenized Corpus
    Returns padded_sequence, max_sequence_length
    """
    sequences = []
    max_sequence_length = 0
    for line in tokenized_corpus:
        for j in range(1, len(line)):
            n_gram_sequence = line[:j + 1]
            if len(n_gram_sequence) > max_sequence_length:
                max_sequence_length = len(n_gram_sequence)
            sequences.append(n_gram_sequence)

    print("Padding Sequences")
    padded_sequence = pad_sequences(sequences, maxlen=max_sequence_length, padding='pre')
    del sequences
    return padded_sequence, max_sequence_length


def create_tokenized_corpus(tokenizer: BERTTokenizer, corpus: list) -> tf.RaggedTensor:
    """Create a Tokenized Corpus"""
    dataset = tf.data.Dataset.from_tensor_slices(corpus)
    tokenized_corpus = []
    for batch in dataset.batch(256):
        batch = tokenizer.tokenize(batch)
        tokenized_corpus.extend(batch)

    del dataset
    return tf.ragged.stack(tokenized_corpus)
