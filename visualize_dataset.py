import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from main import DELIMITER
from preprocessing import CONTRACTIONS


def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


def _count_words(dataset: pd.DataFrame, x="-") -> dict[str, int]:
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

    return dict(sorted(dic.items(), key=lambda x:x[1]))


if __name__ == '__main__':
    lines = []
    dataset_df = pd.read_csv('dataset.csv', dtype=str, delimiter=DELIMITER)
    # count words
    words = _count_words(dataset_df)
    pretty(words)
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
