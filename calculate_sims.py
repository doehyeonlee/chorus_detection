import os
import numpy as np
import chardet
from nltk import download, pos_tag, word_tokenize
from scipy.spatial.distance import cosine, euclidean
from fastdtw import fastdtw
from gensim.models import Word2Vec, KeyedVectors
from Levenshtein import distance as lsd
import phonetics
import pyphen

# Define paths for saved models
word2vec_model_path = 'saved_word2vec_model.model'
# context2vec_model_path = 'saved_context2vec_model.model'

# Check if the models exist
if os.path.exists(word2vec_model_path):
    word2vec_model = KeyedVectors.load(word2vec_model_path)
else:
    word2vec_model = KeyedVectors.load_word2vec_format("wiki-news-300d-1M-subword.vec", binary=False)
    word2vec_model.save(word2vec_model_path)

# if os.path.exists(context2vec_model_path):
#     context2vec_model = Word2Vec.load(context2vec_model_path)
# else:
#     context2vec_model = Word2Vec.load('context2vec')
#     context2vec_model.save(context2vec_model_path)

def load_pronunciation_dict(file_path):
    """Load Pronunciation Dictionary."""
    pronunciation_dict = {}

    # First, detect the file encoding
    with open(file_path, 'rb') as f:
        detected_encoding = chardet.detect(f.read())['encoding']

    # Now, read the file using the detected encoding
    with open(file_path, 'r', encoding=detected_encoding) as f:
        for line in f:
            # Skip lines that start with ";;;"
            if line.startswith(";;;"):
                continue

            parts = line.strip().split('  ')
            if len(parts) == 2:  # ensuring only valid lines are processed
                word, phonemes = parts
                pronunciation_dict[word] = phonemes.split()

    return pronunciation_dict

cmu_dict = load_pronunciation_dict('model/cmudict.txt')
download('punkt')
download('averaged_perceptron_tagger')

def simstr(line1, line2):
    """String similarity using normalized Levenshtein edit distance."""
    return lsd(line1, line2)


def simhead(line1, line2):
    """Head similarity."""
    words1, words2 = word_tokenize(line1)[:2], word_tokenize(line2)[:2]
    return simstr(' '.join(words1), ' '.join(words2))


def simtail(line1, line2):
    """Tail similarity."""
    words1, words2 = word_tokenize(line1)[-2:], word_tokenize(line2)[-2:]
    return simstr(' '.join(words1), ' '.join(words2))


def simphone(line1, line2):
    """Phonetic similarity."""

    def get_phonetic(line):
        words = word_tokenize(line)
        return [cmu_dict.get(word, [''])[0] for word in words]

    return simstr(' '.join(get_phonetic(line1)), ' '.join(get_phonetic(line2)))


def simpos(line1, line2):
    """Part-of-speech similarity."""
    pos1, pos2 = [pos for word, pos in pos_tag(word_tokenize(line1))], [pos for word, pos in
                                                                        pos_tag(word_tokenize(line2))]
    return simstr(' '.join(pos1), ' '.join(pos2))

def simw2v(line1, line2):
    """Word vector similarity."""

    def get_avg_vector(line, model):
        vectors = [model[word] for word in word_tokenize(line) if word in model.key_to_index]
        return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

    vec1, vec2 = get_avg_vector(line1, word2vec_model), get_avg_vector(line2, word2vec_model)
    return 1 - cosine(vec1, vec2)

# def simc2v(line1, line2):
#     """Context vector similarity."""
#     def get_avg_vector(line, model):
#         vectors = [model[word] for word in word_tokenize(line) if word in model.vocab]
#         return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)
#
#     vec1, vec2 = get_avg_vector(line1, context2vec_model), get_avg_vector(line2, context2vec_model)
#     return 1 - cosine(vec1, vec2)

dic = pyphen.Pyphen(lang='en')

def count_syllables(line):
    """Counts the syllables in a line."""
    words = word_tokenize(line)
    return sum([len(dic.inserted(word).split('-')) for word in words])

def get_syllable_count(word):
    """Returns the syllable count for a word."""
    return len(dic.inserted(word).split('-'))


def simsyW(line1, line2):
    """Word syllable count similarity using DTW."""
    syllables1 = [get_syllable_count(word) for word in word_tokenize(line1)]
    syllables2 = [get_syllable_count(word) for word in word_tokenize(line2)]
    print(syllables1)
    print(type(syllables1))
    print(syllables2)
    distance, _ = fastdtw(syllables1, syllables2)
    return 1 / (1 + distance)

def sliding_window(lyrics, window_size=4):
    for i in range(len(lyrics) - window_size + 1):
        yield lyrics[i:i + window_size]

def simsyl(lyrics):
    # Split lyrics into lines and count syllables for each line
    lines = lyrics.split("\n")
    syllable_counts = [count_syllables(line) for line in lines]

    similarities = []

    for win1 in sliding_window(syllable_counts):
        for win2 in sliding_window(syllable_counts):
            distance, _ = fastdtw(win1, win2)
            similarities.append((1 / (1 + distance)))  # Similarity is the inverse of distance

    return np.mean(similarities)
