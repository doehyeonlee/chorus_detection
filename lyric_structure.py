import numpy as np
import re
from functools import reduce
import calculate_sims as cal
from fastdtw import fastdtw

def tree_structure(text):
    #normalize segment border encoding
    segment_border_encoder = '<segmentborder>'
    line_border_encoder = '<lineborder>'
    tree_string = re.sub('(( )*\n( )*){2,}', segment_border_encoder, text)
    tree_string = re.sub('( )*\n( )*', line_border_encoder, tree_string)
    #parse tree_string
    segment_structure = tree_string.split(segment_border_encoder)
    tree_structure = list(map(lambda segment: segment.split(line_border_encoder), segment_structure))
    return tree_structure

#removed fancy stuff like bracket removal here, until we have stable results
def normalize_lyric(lyric):
    return lyric.lower()

#given a text tree structure, print it nicely
def pretty_print_tree(text_tree):
    space_between = '    '
    res = ''
    output_separator = '\n'
    block_index = 0
    line_index = 0
    for block in text_tree:
        if not block:
            continue
        line_in_block_index = 0
        for line in block:
            line = line.strip()
            if not line:
                continue
            line_pretty = space_between + str(block_index) + '.' + str(line_in_block_index)\
                          + space_between + str(line_index) + space_between + line + output_separator
            res += line_pretty
            line_in_block_index += 1
            line_index += 1
        block_index += 1
        res += output_separator
    return space_between + res

#The indices of lines that end a segment
def segment_borders(lyric):
    normalized_lyric = normalize_lyric(lyric)
    segment_lengths = reduce(lambda x, block: x + [len(block)], tree_structure(lyric), [])
    segment_indices = []
    running_sum = -1
    for i in range(len(segment_lengths)):
        running_sum += segment_lengths[i]
        segment_indices.append(running_sum)
    return segment_indices[:-1]
def segment_count(lyric):
     return 1 + len(segment_borders(lyric))

#flattened tree structure, does not differentiate between segment and line border
def line_structure(text):
    return reduce(lambda x, segment: x + segment, tree_structure(text), [])

def self_similarity_matrix(items, metric):
    return np.array([[metric(x, y) for x in items] for y in items])

def calculate_str(lyric):
    normalized_lyric = normalize_lyric(lyric)
    line_encoding_string = line_structure(normalized_lyric)
    ssm_lines_string = self_similarity_matrix(
        line_encoding_string, metric=lambda x, y: pow(cal.simstr(x, y), 1)
    )
    return ssm_lines_string

def calculate_head(lyric):
    normalized_lyric = normalize_lyric(lyric)
    line_encoding_string = line_structure(normalized_lyric)
    ssm_lines_string = self_similarity_matrix(
        line_encoding_string, metric=lambda x, y: pow(cal.simhead(x, y), 1)
    )
    return ssm_lines_string

def calculate_tail(lyric):
    normalized_lyric = normalize_lyric(lyric)
    line_encoding_string = line_structure(normalized_lyric)
    ssm_lines_string = self_similarity_matrix(
        line_encoding_string, metric=lambda x, y: pow(cal.simtail(x, y), 1)
    )
    return ssm_lines_string

def calculate_phone(lyric):
    normalized_lyric = normalize_lyric(lyric)
    line_encoding_string = line_structure(normalized_lyric)
    ssm_lines_string = self_similarity_matrix(
        line_encoding_string, metric=lambda x, y: pow(cal.simphone(x, y), 1)
    )
    return ssm_lines_string

def calculate_pos(lyric):
    normalized_lyric = normalize_lyric(lyric)
    line_encoding_string = line_structure(normalized_lyric)
    ssm_lines_string = self_similarity_matrix(
        line_encoding_string, metric=lambda x, y: pow(cal.simpos(x, y), 1)
    )
    return ssm_lines_string

def calculate_w2v(lyric):
    normalized_lyric = normalize_lyric(lyric)
    line_encoding_string = line_structure(normalized_lyric)
    ssm_lines_string = self_similarity_matrix(
        line_encoding_string, metric=lambda x, y: pow(cal.simw2v(x, y), 1)
    )
    return ssm_lines_string

def calculate_s2v(lyric):
    normalized_lyric = normalize_lyric(lyric)
    line_encoding_string = line_structure(normalized_lyric)
    ssm_lines_string = self_similarity_matrix(
        line_encoding_string, metric=lambda x, y: pow(cal.sims2v(x, y), 1)
    )
    return ssm_lines_string


def calculate_syW(lyric):
    normalized_lyric = normalize_lyric(lyric)
    line_encoding_string = line_structure(normalized_lyric)
    ssm_lines_string = self_similarity_matrix(
        line_encoding_string, metric=lambda x, y: pow(cal.simsyW(x, y), 1)
    )
    return ssm_lines_string

def calculate_syl(lyric):
    return cal.simsyl(lyric)
