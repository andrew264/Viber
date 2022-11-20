import os.path

import numpy as np
import tensorflow as tf
from keras.utils import pad_sequences

from bert_tokenizer import BERTTokenizer
from model import create_model

seq_len = 100
randomize = False

# run in CPU only mode
tf.config.set_visible_devices([], 'GPU')


def run(tokenizer: BERTTokenizer, max_seq_len: int) -> None:
    """
    Run the model.
    """
    print("Running model...")
    if not os.path.exists("./checkpoints"):
        raise FileNotFoundError("No checkpoint found. Please train the model first using main.py.")
    model = create_model(max_seq_len, tokenizer.get_vocab_size())
    model.load_weights(tf.train.latest_checkpoint('./checkpoints'))
    model.build(tf.TensorShape([1, None]))
    while True:
        seed_text = input("Enter seed text: ")
        if seed_text == '':
            break
        token_list = []
        token_list.extend(tokenizer.tokenize(seed_text)[0].numpy().tolist())
        for i in range(seq_len):
            inp = [token_list[-8:]]
            tokens = pad_sequences(inp, maxlen=max_seq_len - 1, padding='pre')
            predicted_probs = model.predict(tokens, verbose=0, use_multiprocessing=True)[0]
            predicted = np.random.choice([x for x in range(len(predicted_probs))],
                                         p=predicted_probs)
            token_list.append(predicted)
        print("Output: " + tokenizer.detokenize([token_list])[0].numpy().decode())


if __name__ == "__main__":
    run(BERTTokenizer("./vocab.txt"), seq_len)
