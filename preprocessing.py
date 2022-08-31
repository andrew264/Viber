import re

from keras.preprocessing.text import Tokenizer

CONTRACTIONS = (("i'm", "i am"), ("he's", "he is"), ("she's", "she is"), ("it's", "it is"), ("that's", "that is"),
                ("what's", "what is"), ("when's", "when is"), ("where's", "where is"), ("how's", "how is"),
                ("ain't", "is not"), ("can't", "can not"), ("won't", "will not"), ("n't", " not"), ("'ll", " will"),
                ("'ve", " have"), ("'re", " are"), ("'d", " would"), ("n'", "ng"), ("'bout", "about"),
                ("'cause", "because"), ("cuz", "because"), ("in'", "ing"))


def tokenize_corpus(corpus, num_words=-1):
    # Fit a Tokenizer on the corpus
    if num_words > -1:
        tokenizer = Tokenizer(num_words=num_words)
    else:
        tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    return tokenizer


def create_lyrics_corpus(dataset, field):
    # Make it lowercase
    dataset[field] = dataset[field].str.lower()
    # Make it one long string to split by line
    lyrics = dataset[field].str.cat()

    return cleanup_lyrics(lyrics)


def cleanup_lyrics(lyrics: str) -> list[str]:
    for (contraction, expansion) in CONTRACTIONS:
        lyrics = re.sub(contraction, expansion, lyrics)

    lyrics = lyrics.lower().splitlines()
    corpus = []
    for line in lyrics:
        if line.strip() == "":
            continue
        # Keep space, a to z, and select punctuation.
        line = re.sub('[^ a-z.?!,¿]', '', line)
        # Add spaces around punctuation.
        line = re.sub(r"([?.!,])", r" \1 ", line)
        corpus.append(line)

    return corpus
