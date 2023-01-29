import collections

import six
import tensorflow as tf
import unicodedata
import sentencepiece as spm
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

SPIECE_UNDERLINE = u"▁".encode("utf-8")


def preprocess_text(inputs, remove_space=True, lower=False):
    """preprocess data by removing extra space and normalize data."""
    outputs = inputs
    if remove_space:
        outputs = " ".join(inputs.strip().split())

    outputs = unicodedata.normalize("NFKD", outputs)
    outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
    if lower:
        outputs = outputs.lower()

    return outputs


def encode_pieces(sp_model, text, sample=False):
    """turn sentences into word pieces."""

    if not sample:
        pieces = sp_model.EncodeAsPieces(text)
    else:
        pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)
    new_pieces = []
    for piece in pieces:
        piece = printable_text(piece)
        if len(piece) > 1 and piece[-1] == "," and piece[-2].isdigit():
            cur_pieces = sp_model.EncodeAsPieces(
                six.ensure_binary(piece[:-1]).replace(SPIECE_UNDERLINE, b""))
            if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                if len(cur_pieces[0]) == 1:
                    cur_pieces = cur_pieces[1:]
                else:
                    cur_pieces[0] = cur_pieces[0][1:]
            cur_pieces.append(piece[-1])
            new_pieces.extend(cur_pieces)
        else:
            new_pieces.append(piece)

    return new_pieces


def encode_ids(sp_model, text, sample=False):
    pieces = encode_pieces(sp_model, text, sample=sample)
    ids = [sp_model.PieceToId(piece) for piece in pieces]
    return ids


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with tf.io.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip().split()[0] if token.strip() else " "
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output


class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, spm_model_file):
        self.sp_model = spm.SentencePieceProcessor()
        tf.print("loading sentence piece model")
        # Handle cases where SP can't load the file, but gfile can.
        sp_model_ = tf.io.gfile.GFile(spm_model_file, "rb").read()
        self.sp_model.LoadFromSerializedProto(sp_model_)
        self.vocab = {self.sp_model.IdToPiece(i): i for i in range(self.sp_model.GetPieceSize())}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        if isinstance(text, tf.Tensor):
            text = text.numpy().tolist()
        tokens = self.sp_model.EncodeAsIds(text)
        return tf.convert_to_tensor(tokens, dtype=tf.int16)

    def detokenize(self, tokens):
        if isinstance(tokens, tf.Tensor):
            tokens = tokens.numpy().tolist()
        return self.sp_model.DecodeIds(tokens)

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, tf.Tensor):
            tokens = tokens.numpy().tolist()
        ids = [self.sp_model.PieceToId(printable_text(token)) for token in tokens]
        return tf.convert_to_tensor(ids, dtype=tf.int16)

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, tf.Tensor):
            ids = ids.numpy().tolist()
        tokens = [self.sp_model.IdToPiece(id_) for id_ in ids]
        return tf.convert_to_tensor(tokens, dtype=tf.string)

    @property
    def vocab_size(self):
        return self.sp_model.vocab_size()


if __name__ == '__main__':
    import pandas as pd

    dataset = pd.read_csv('dataset.csv', delimiter='|')
    lyrics = dataset['lyrics'].str.cat()
    with open('lyrics.txt', 'w') as f:
        f.write(lyrics)

    spm.SentencePieceTrainer.train(input='lyrics.txt', model_prefix='sp_model', vocab_size=512, model_type='unigram', shuffle_input_sentence=True, num_threads=20, user_defined_symbols=['\n'])
