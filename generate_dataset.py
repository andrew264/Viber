import json
import os
import re

import pandas as pd
from lyricsgenius import Genius

from lyricsgenius.artist import Artist
from lyricsgenius.song import Song

DELIMITER = "|"
MAX_SONGS = 50
ARTISTS = []


def cleanup_lyrics(lyrics: str) -> str:
    """
    Cleans up lyrics by removing the following:
        - words in brackets
        - repeated newlines
        - first line of lyrics
    """
    try:
        lyrics = lyrics.split("\n", 1)[1]
    except IndexError:
        pass
    lyrics = re.sub(r"[\(\[].*?[\)\]]", "", lyrics)
    lyrics = re.sub(r"[0-9]*URLCopyEmbedCopy", '', lyrics)
    return break_long_lines(lyrics)


def break_long_lines(lyrics: str) -> str:
    """
    Breaks up long lines into multiple lines
    and remove one word lines too
    """
    lines = []
    for line in lyrics.splitlines():
        line = line.strip()
        if len(line.split()) == 1:
            continue
        if len(line) > 100:
            for sub_line in line.split('. '):
                lines.append(sub_line + '.')
        else:
            lines.append(line)
    return '\n'.join(lines)


if __name__ == '__main__':
    # Creating Dataset
    if not os.path.exists('dataset.csv'):
        with open('dataset.csv', 'w') as file:
            print('Creating dataset.csv')
            file.close()
    with open('dataset.csv', 'r', encoding='utf8') as dataset_file:
        if dataset_file.read() == '':
            print('Adding headers to dataset.csv')
            dataset_file.close()
            pd.DataFrame(columns=['artist', 'title', 'lyrics']).to_csv('dataset.csv', index=False, sep=DELIMITER)

    # Getting Genius API Token
    with open("config.json", "r") as configFile:
        config: dict = json.load(configFile)
        configFile.close()
    genius = Genius(config['GENIUS_API_TOKEN'], sleep_time=0.5, timeout=15, remove_section_headers=True,
                    excluded_terms=['(Remix)', '(Live)', '(Demo)', '(Cover)', '(Instrumental)', '(Clean)', '(Version)',
                                    '(Extended)', '(Original)', '(Silent)', '(Radio Edit)'],
                    retries=3)

    artists: list[str] = input('Enter artist name: ').split(',') or ARTISTS

    # Getting songs
    song: Song
    dataset = pd.read_csv('dataset.csv', sep=DELIMITER)
    for artist in artists:
        # Getting artist
        artist: Artist = genius.search_artist(artist, per_page=1, get_full_info=False, max_songs=MAX_SONGS)
        for song in artist.songs:
            if artist.name in dataset['artist'].values and song.title in dataset['title'].values:
                # Checking if song is already in dataset
                continue
            print(f'Adding {song.title} to dataset.csv')
            dataset.loc[len(dataset)] = [artist.name.strip(), song.title.strip(), cleanup_lyrics(song.lyrics)]
        dataset.sort_values(by=['artist', 'title'], inplace=True)
        dataset.to_csv('dataset.csv', index=False, sep=DELIMITER)

# lyrics = ''.join(dataset['lyrics'].values)
# with open('lyrics.txt', 'w', encoding='utf8') as file:
#     file.write(lyrics)
#     file.close()
