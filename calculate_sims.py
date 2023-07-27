import os
import numpy as np
import chardet
from nltk import download, pos_tag, word_tokenize
from scipy.spatial.distance import cosine, euclidean
from fastdtw import fastdtw
from gensim.models import Word2Vec, KeyedVectors
from Levenshtein import distance as lsd
from sentence_transformers import SentenceTransformer
from hyphenate import hyphenate_word

# word2vec model download
word2vec_model_path = 'saved_word2vec_model.model'
if os.path.exists(word2vec_model_path):
    word2vec_model = KeyedVectors.load(word2vec_model_path)
else:
    word2vec_model = KeyedVectors.load_word2vec_format("wiki-news-300d-1M-subword.vec", binary=False)
    word2vec_model.save(word2vec_model_path)

# sen2vec model download
sen2vec_model = SentenceTransformer('sentence-transformers/sentence-t5-base')


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
    try:
        # Actually, just computing the Normalized Levenshtein edit distance
        return lsd(line1, line2) / max(len(line1), len(line2))
    except ZeroDivisionError:
        print(line1, ":", len(line1), line2, ":", len(line2))

        # If both strings are empty, they are considered similar
        return 1 if not line1 and not line2 else 0

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
    line1 = line1.upper()
    line2 = line2.upper()
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

def sims2v(line1, line2):
    """Context vector similarity."""
    vec1, vec2 = sen2vec_model.encode(line1), sen2vec_model.encode(line2)
    return 1 - cosine(vec1, vec2)


def get_syllable_count(word):
    """Returns the syllable count for a word."""
    return len(hyphenate_word(word))

def count_syllables(line):
    """Counts the syllables in a line."""
    words = word_tokenize(line)
    return sum([len(hyphenate_word(word)) for word in words])


def simsyW(line1, line2):
    """Word syllable count similarity using DTW."""
    syllables1 = [get_syllable_count(word) for word in word_tokenize(line1)]
    syllables2 = [get_syllable_count(word) for word in word_tokenize(line2)]
    distance, _ = fastdtw(syllables1, syllables2)
    return 1 / (1 + distance)


def simsyl(lyrics):
    lyrics = str(lyrics)
    lines = lyrics.split("\n")
    # Calculate the total syllable count for each lyric line
    syllable_counts = [count_syllables(line) for line in lines if count_syllables(line)>0]
    T = len(syllable_counts)
    # Create a matrix to store the similarities
    similarity_matrix = np.zeros((T, T))

    # Calculate the similarity using DTW and fill in the matrix
    for i in range(T):
        for j in range(T):
            window1 = syllable_counts[i:i+4]
            window2 = syllable_counts[j:j+4]
            distance, _ = fastdtw(window1, window2)
            similarity_matrix[i, j] = 1 / (1 + distance)  # Similarity is the inverse of distance
    return similarity_matrix
