import tensorflow as tf

from model import OneStep

states = [tf.zeros([1, 1024], dtype=tf.float32), tf.zeros([1, 1024], dtype=tf.float32)]
one_step_model: OneStep = tf.saved_model.load("one_step")

while True:
    print("_"*80)
    string_input = input("\nEnter seed text: ")
    states = [tf.zeros([1, 1024], dtype=tf.float32), tf.zeros([1, 1024], dtype=tf.float32)]
    if string_input == "":
        break
    print(string_input, end="")
    string_input = tf.constant([string_input])
    for n in range(500):
        next_char, states = one_step_model.generate_one_step(string_input, states=states)
        states = [tf.convert_to_tensor(state) for state in states]
        string_input = string_input + next_char
        print(next_char.numpy()[0].decode(), end="")
    print("\n")

