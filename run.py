import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf

from model import OneStep

# gpu enable memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def get_initial_state(rnn_units):
    return [tf.zeros([1, rnn_units]) for _ in range(4)]


if not os.path.exists("one_step"):
    raise ValueError("You need to train a model first.")
one_step_model: OneStep = tf.saved_model.load("one_step")

while True:
    print("_" * 80)
    string_input = input("\nEnter seed text: ")
    states = get_initial_state(rnn_units=1024)
    if string_input == "":
        break
    print(string_input, end="")
    string_input = tf.constant([string_input])
    for n in range(500):
        next_char, states = one_step_model.generate_one_step(string_input,
                                                             states=states)
        states = [tf.convert_to_tensor(state) for state in states]
        string_input = string_input + next_char
        print(next_char.numpy()[0].decode(), end="")
    print("\n")
