import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from main import DELIMITER
from preprocessing import CONTRACTIONS, cleanup_lyrics


def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


def word_counts(dataset: pd.DataFrame, x="-") -> dict[str, int]:
    dic = dict()
    lyrics = "".join(dataset["lyrics"].str.lower())
    for (contraction, expansion) in CONTRACTIONS:
        lyrics = re.sub(contraction, expansion, lyrics)
    for _line in lyrics.splitlines():
        for word in _line.split():
            if x in word.lower():
                if word.lower() in dic:
                    dic[word.lower()] += 1
                else:
                    dic[word.lower()] = 1

    return dict(sorted(dic.items(), key=lambda x: x[1]))


def replace_words_in_dataset(dataset: pd.DataFrame, old: str, new: str = "") -> pd.DataFrame:
    """replace words in dataset"""
    dataset["lyrics"] = dataset["lyrics"].str.replace(old, new)
    return dataset


def word_frequency(dataset: pd.DataFrame) -> None:
    """ """
    word_freq = {}
    clean_lyrics = cleanup_lyrics(dataset['lyrics'].str.cat())
    for _line in clean_lyrics:
        for _word in _line.split():
            if _word in word_freq:
                word_freq[_word] += 1
            else:
                word_freq[_word] = 1

    _dataset = pd.DataFrame.from_dict(word_freq, orient="index")
    _dataset.to_csv("word-frequency.csv", header=["FREQUENCY"])


if __name__ == '__main__':
    lines = []
    dataset_df = pd.read_csv('dataset.csv', dtype=str, delimiter=DELIMITER)
    # get all artists
    artists = dataset_df["artist"].unique()
    print(f"Total artists: {len(artists)}")
    print(f"Total songs: {len(dataset_df)}")
    word_frequency(dataset_df)

    # iterate over dataset
    index: int
    for index, row in dataset_df.iterrows():
        for line in row['lyrics'].split('\n'):
            if line.strip() != '':
                lines.append(line)
    print("Total lines: ", len(lines))
    print("Total unique lines: ", len(set(lines)))
    length_of_lines = np.array([len(line) for line in lines])
    mean = np.mean(length_of_lines)
    print("Average length of lines: ", mean)
    print("Median length of lines: ", np.median(length_of_lines))
    print("Max length of lines: ", np.max(length_of_lines))
    print("Min length of lines: ", np.min(length_of_lines))
    print("95th percentile length of lines: ", np.percentile(length_of_lines, 95))
    # Plot the distribution of line lengths
    plt.hist(length_of_lines, bins=np.max(length_of_lines))
    plt.title("Distribution of line lengths")
    plt.xlabel("Length of line")
    plt.ylabel("Number of lines")
    plt.axvline(mean, linestyle='dashed', color='black', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(length_of_lines.mean() * 1.1, max_ylim * 0.9, f'Mean: {length_of_lines.mean():.2f}')
    plt.show()
