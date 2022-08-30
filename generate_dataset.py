import json
import os
import re

import pandas as pd
from lyricsgenius import Genius

from lyricsgenius.artist import Artist
from lyricsgenius.song import Song

DELIMITER = "|"


def cleanup_lyrics(lyrics: str) -> str:
    """
    Cleans up lyrics by removing the following:
        - words in brackets
        - repeated newlines
        - first line of lyrics
    """
    lyrics = lyrics.split("\n", 1)[1]
    lyrics = re.sub(r"[\(\[].*?[\)\]]", "", lyrics)
    lyrics.replace("\n\n", "\n")
    lyrics = re.sub(r"[0-9]*Embed*", "", lyrics)
    lyrics = re.sub(r"URLCopyEmbedCopy", "", lyrics)
    return lyrics


# Creating Dataset
if not os.path.exists('dataset.csv'):
    with open('dataset.csv', 'w') as file:
        print('Creating dataset.csv')
        file.close()
with open('dataset.csv', 'r') as dataset_file:
    if dataset_file.read() == '':
        print('Adding headers to dataset.csv')
        dataset_file.close()
        pd.DataFrame(columns=['artist', 'title', 'lyrics']).to_csv('dataset.csv', index=False, sep=DELIMITER)

# Getting Genius API Token
with open("config.json", "r") as configFile:
    config: dict = json.load(configFile)
    configFile.close()
genius = Genius(config['GENIUS_API_TOKEN'], sleep_time=0.75, timeout=15)

artist: str = (input('Enter artist name: ')).lower()

# Getting artist
artist: Artist = genius.search_artist(artist, per_page=1, get_full_info=False, max_songs=50)

# Getting songs
song: Song
dataset = pd.read_csv('dataset.csv', sep=DELIMITER)
for song in artist.songs:
    if artist.name in dataset['artist'].values and song.title in dataset['title'].values:
        # Checking if song is already in dataset
        continue
    print(f'Adding {song.title} to dataset.csv')
    dataset.loc[len(dataset)] = [artist.name, song.title, cleanup_lyrics(song.lyrics)]

dataset.to_csv('dataset.csv', index=False, sep=DELIMITER)
