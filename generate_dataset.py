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
    # try:
    #     lyrics = lyrics.split("\n", 1)[1]
    # except IndexError:
    #     pass
    lyrics = re.sub(r"[\(\[].*?[\)\]]", '', lyrics)
    lyrics = re.sub(r"You might also like", '', lyrics)
    lyrics = re.sub(r"[0-9]*URLCopyEmbedCopy", '', lyrics)
    lyrics = re.sub(r"[0-9]*Embed", '', lyrics)
    # if first line has "Lyrics" in it, remove it
    if len(lyrics.splitlines()) > 1 and "lyrics" in lyrics.splitlines()[0].lower():
        lyrics = lyrics.split("\n", 1)[1]
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
    return '\n'.join(lines).encode('ascii', errors='ignore').decode()


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
    artists = list(set(artists))
    repr(artists)

    if not artists or artists == ['']:
        print('cleaning dataset.csv')
        dataset_df = pd.read_csv('dataset.csv', dtype=str, delimiter=DELIMITER)
        dataset_df = dataset_df.drop_duplicates(subset=['artist', 'title', 'lyrics'])
        for index, row in dataset_df.iterrows():
            if row['lyrics'] == 'nan' or row['lyrics'] is None or row['lyrics'] == '':
                # remove rows with no lyrics
                dataset_df.drop(index, inplace=True)
                continue
            if len(dataset_df.at[index, 'lyrics'].splitlines()) < 3:
                # remove rows with less than 3 lines of lyrics
                dataset_df.drop(index, inplace=True)
                continue
            dataset_df.at[index, 'lyrics'] = cleanup_lyrics(row['lyrics'])
            dataset_df.at[index, 'artist'] = row['artist'].encode('ascii', errors='ignore').decode()
            dataset_df.at[index, 'title'] = row['title'].encode('ascii', errors='ignore').decode()
        dataset_df.to_csv('dataset.csv', index=False, sep=DELIMITER)
        exit(1)

    # Getting songs
    song: Song
    dataset = pd.read_csv('dataset.csv', sep=DELIMITER)
    print(f"Total Artists: {len(artists)}")
    for i in range(len(artists)):
        print(f"Progress: {i + 1 / len(artists) / 100:.0%} %")
        # Getting artist
        artist: Artist = genius.search_artist(artists[i], per_page=1, get_full_info=False, max_songs=MAX_SONGS)
        for song in artist.songs:
            if artist.name in dataset['artist'].values and song.title in dataset['title'].values:
                # Checking if song is already in dataset
                continue
            if song.lyrics is None or song.lyrics == '':
                # Checking if song has lyrics
                continue
            print(f'Adding {song.title} to dataset.csv')
            dataset.loc[len(dataset)] = [artist.name.encode('ascii', errors='ignore').decode(),
                                         song.title.encode('ascii', errors='ignore').decode(),
                                         cleanup_lyrics(song.lyrics)]
        dataset.sort_values(by=['artist', 'title'], inplace=True)
        dataset.to_csv('dataset.csv', index=False, sep=DELIMITER)

# lyrics = ''.join(dataset['lyrics'].values)
# with open('lyrics.txt', 'w', encoding='utf8') as file:
#     file.write(lyrics)
#     file.close()
