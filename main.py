import tensorflow as tf

import os
import pandas as pd

from model import MyModel, OneStep

DELIMITER = '|'
seq_length = 100

BATCH_SIZE = 128
BUFFER_SIZE = 10000
EPOCHS = 10

dataset_df = pd.read_csv('dataset.csv', dtype=str, delimiter=DELIMITER).sample(frac=1)

lyrics = dataset_df['lyrics'].str.cat()
vocab = sorted(set(lyrics))
print(f'{len(vocab)} unique characters')

ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None)
chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)


def text_from_ids(_ids):
    return tf.strings.reduce_join(chars_from_ids(_ids), axis=-1)


all_ids = ids_from_chars(tf.strings.unicode_split(lyrics, 'UTF-8'))
all_ids = tf.cast(all_ids, dtype=tf.int8)

ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)


def split_input_target(sequence: list[str]) -> tuple[list[str], list[str]]:
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


dataset = (
    sequences
    .map(split_input_target)
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

# Length of the vocabulary in StringLookup Layer
vocab_size = len(ids_from_chars.get_vocabulary())

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

model = MyModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)

if os.path.exists(checkpoint_dir):
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

one_step_model = OneStep(model, chars_from_ids, ids_from_chars, temperature=0.40)
tf.saved_model.save(one_step_model, "one_step")
