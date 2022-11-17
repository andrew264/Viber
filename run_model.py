import os.path

import numpy as np
from keras import Sequential
from keras.utils import pad_sequences

from bert_tokenizer import BERTTokenizer
from model import load_model

next_words = 50
randomize = False


def run(model: Sequential, tokenizer: BERTTokenizer, max_seq_len: int) -> None:
    """
    Run the model.
    """
    print("Running model...")
    while True:
        seed_text = input("Enter seed text: ")
        if seed_text == '':
            break
        token_list = []
        token_list.extend(tokenizer.tokenize(seed_text)[0].numpy().tolist())
        for i in range(next_words):
            tokens = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')
            predicted_probs = model.predict(tokens, verbose=0)[0]
            predicted = np.random.choice([x for x in range(len(predicted_probs))],
                                         p=predicted_probs)
            token_list.append(predicted)
        print("Output: " + tokenizer.detokenize([token_list])[0].numpy().decode())


if __name__ == "__main__":

    run(load_model(), BERTTokenizer("./vocab.txt"))
