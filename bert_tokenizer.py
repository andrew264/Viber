import pathlib
import re
from abc import ABC

import tensorflow as tf
import tensorflow_text as text


class BERTTokenizer(tf.Module, ABC):
    def __init__(self, vocab_path):
        super().__init__()
        self.tokenizer = text.BertTokenizer(vocab_path, lower_case=True)
        self._reserved_tokens = ["[PAD]", "[UNK]", "[NEWLINE]"]
        self._NEWLINE = tf.argmax(tf.constant(self._reserved_tokens) == "[NEWLINE]")
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        vocab = pathlib.Path(vocab_path).read_text().splitlines()
        self.vocab = tf.Variable(vocab)

        # Create the signatures for export:

        # Include a tokenize signature for a batch of strings.
        self.tokenize.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string))

        # Include `detokenize` and `lookup` signatures for:
        #   * `Tensors` with shapes [tokens] and [batch, tokens]
        #   * `RaggedTensors` with shape [batch, tokens]
        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.detokenize.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        self.lookup.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.lookup.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        # These `get_*` methods take no arguments
        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_reserved_tokens.get_concrete_function()

    @tf.function
    def tokenize(self, strings) -> tf.RaggedTensor:
        strings = tf.strings.regex_replace(strings, r"\n", self._NEWLINE)
        enc: tf.RaggedTensor = self.tokenizer.tokenize(strings)
        # Merge the `word` and `word-piece` axes.
        return enc.merge_dims(-2, -1)

    @tf.function
    def detokenize(self, tokenized) -> str:
        words = self.tokenizer.detokenize(tokenized)
        # Drop the reserved tokens, except for "[UNK]".
        bad_tokens = [re.escape(tok) for tok in self._reserved_tokens if tok != "[UNK]"]
        bad_token_re = "|".join(bad_tokens)

        bad_cells = tf.strings.regex_full_match(words, bad_token_re)
        result = tf.ragged.boolean_mask(words, ~bad_cells)
        result = tf.strings.reduce_join(result, separator=' ', axis=-1)

        # Join them into strings.
        return tf.strings.regex_replace(result, self._NEWLINE, "\n")

    @tf.function
    def lookup(self, token_ids):
        return tf.gather(self.vocab, token_ids)

    @tf.function
    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]

    @tf.function
    def get_vocab_path(self):
        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self):
        return tf.constant(self._reserved_tokens)
