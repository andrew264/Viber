import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from main import DELIMITER

if __name__ == '__main__':
    lines = []
    dataset_df = pd.read_csv('dataset.csv', dtype=str, delimiter=DELIMITER)
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
    # Plot the distribution of line lengths
    plt.hist(length_of_lines, bins=np.max(length_of_lines))
    plt.title("Distribution of line lengths")
    plt.xlabel("Length of line")
    plt.ylabel("Number of lines")
    plt.axvline(mean, linestyle='dashed', color='black', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(length_of_lines.mean() * 1.1, max_ylim * 0.9, f'Mean: {length_of_lines.mean():.2f}')
    plt.show()
