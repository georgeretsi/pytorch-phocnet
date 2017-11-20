'''
Created on Dec 17, 2015

@author: ssudholt
'''
import logging

import numpy as np
import tqdm

def get_unigrams_from_strings(word_strings, split_character=None):
    unigrams = None
    if split_character is not None:
        unigrams = [elem for word_string in word_strings for elem in word_string.split(split_character)]
    else:
        unigrams = [elem for word_string in word_strings for elem in word_string]
    unigrams = sorted(set(unigrams))
    return unigrams

def get_most_common_n_grams(words, num_results=50, len_ngram=2):
    '''
    Calculates the 50 (default) most common bigrams (default) from a
    list of pages, where each page is a list of WordData objects.

    @param words: (list of str)
        List containing the words from which to extract the bigrams
    @param num_results: (int)
        Number of n-grams returned.
    @param len_ngram: (int)
        length of n-grams.
    @return most common <n>-grams
    '''
    ngrams = {}
    for word in words:
        w_ngrams = get_n_grams(word, len_ngram)
        for ngram in w_ngrams:
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
    sorted_list = sorted(ngrams.items(), key=lambda x: x[1], reverse=True)
    top_ngrams = sorted_list[:num_results]
    return {k: i for i, (k, _) in enumerate(top_ngrams)}


def build_phoc_descriptor(words, phoc_unigrams, unigram_levels,  #pylint: disable=too-many-arguments, too-many-branches, too-many-locals
                          bigram_levels=None, phoc_bigrams=None,
                          split_character=None, on_unknown_unigram='nothing',
                          phoc_type='phoc'):
    '''
    Calculate Pyramidal Histogram of Characters (PHOC) descriptor (see Almazan 2014).

    Args:
        word (str): word to calculate descriptor for
        phoc_unigrams (str): string of all unigrams to use in the PHOC
        unigram_levels (list of int): the levels to use in the PHOC
        split_character (str): special character to split the word strings into characters
        on_unknown_unigram (str): What to do if a unigram appearing in a word
            is not among the supplied phoc_unigrams. Possible: 'warn', 'error', 'nothing'
        phoc_type (str): the type of the PHOC to be build. The default is the
            binary PHOC (standard version from Almazan 2014).
            Possible: phoc, spoc
    Returns:
        the PHOC for the given word
    '''
    # prepare output matrix
    logger = logging.getLogger('PHOCGenerator')
    if on_unknown_unigram not in ['error', 'warn','nothing']:
        raise ValueError('I don\'t know the on_unknown_unigram parameter \'%s\'' % on_unknown_unigram)
    phoc_size = len(phoc_unigrams) * np.sum(unigram_levels)
    if phoc_bigrams is not None:
        phoc_size += len(phoc_bigrams) * np.sum(bigram_levels)
    phocs = np.zeros((len(words), phoc_size))
    # prepare some lambda functions
    occupancy = lambda k, n: [float(k) / n, float(k + 1) / n]
    overlap = lambda a, b: [max(a[0], b[0]), min(a[1], b[1])]
    size = lambda region: region[1] - region[0]

    # map from character to alphabet position
    char_indices = {d: i for i, d in enumerate(phoc_unigrams)}

    # iterate through all the words
    for word_index, word in enumerate(tqdm.tqdm(words)):
        if split_character is not None:
            word = word.split(split_character)

        n = len(word) #pylint: disable=invalid-name
        for index, char in enumerate(word):
            char_occ = occupancy(index, n)
            if char not in char_indices:
                if on_unknown_unigram == 'warn':
                    logger.warn('The unigram \'%s\' is unknown, skipping this character', char)
                    continue
                elif on_unknown_unigram == 'error':
                    logger.fatal('The unigram \'%s\' is unknown', char)
                    raise ValueError()
                else:
                    continue
            char_index = char_indices[char]
            for level in unigram_levels:
                for region in range(level):
                    region_occ = occupancy(region, level)
                    if size(overlap(char_occ, region_occ)) / size(char_occ) >= 0.5:
                        feat_vec_index = sum([l for l in unigram_levels if l < level]) * len(phoc_unigrams) + region * len(phoc_unigrams) + char_index
                        if phoc_type == 'phoc':
                            phocs[word_index, feat_vec_index] = 1
                        elif phoc_type == 'spoc':
                            phocs[word_index, feat_vec_index] += 1
                        else:
                            raise ValueError('The phoc_type \'%s\' is unknown' % phoc_type)
        # add bigrams
        if phoc_bigrams is not None:
            ngram_features = np.zeros(len(phoc_bigrams) * np.sum(bigram_levels))
            ngram_occupancy = lambda k, n: [float(k) / n, float(k + 2) / n]
            for i in range(n - 1):
                ngram = word[i:i + 2]
                if phoc_bigrams.get(ngram, 0) == 0:
                    continue
                occ = ngram_occupancy(i, n)
                for level in bigram_levels:
                    for region in range(level):
                        region_occ = occupancy(region, level)
                        overlap_size = size(overlap(occ, region_occ)) / size(occ)
                        if overlap_size >= 0.5:
                            if phoc_type == 'phoc':
                                ngram_features[region * len(phoc_bigrams) + phoc_bigrams[ngram]] = 1
                            elif phoc_type == 'spoc':
                                ngram_features[region * len(phoc_bigrams) + phoc_bigrams[ngram]] += 1
                            else:
                                raise ValueError('The phoc_type \'%s\' is unknown' % phoc_type)
            phocs[word_index, -ngram_features.shape[0]:] = ngram_features
    return phocs

def build_correlated_phoc(words, phoc_unigrams, n_levels, split_character=None, on_unknown_unigram='error'):
    '''
    class 0 (00) means current level attribute is 0 and father level attribute is 0
    class 1 (01) means current level attribute is 0 and father level attribute is 1
    class 2 (11) means current level attribute is 1 and father level attribute is 1
    '''
    phoc_unigram_levels = [2 ** i for i in range(n_levels)]
    phocs = build_phoc_descriptor(words=words,
                                 phoc_unigrams=phoc_unigrams,
                                 unigram_levels=phoc_unigram_levels,
                                 split_character=split_character,
                                 on_unknown_unigram=on_unknown_unigram)
    phocs = phocs.astype(np.uint8)
    phoc_splits = np.split(ary=phocs, indices_or_sections=np.sum(phoc_unigram_levels), axis=1)
    correlated_phocs = []
    for cur_split_id in range(1, len(phoc_splits)):
        # the father of the current split_id is always split_id/2
        father_id = cur_split_id / 2
        cur_correlated_phoc = np.bitwise_or(phoc_splits[cur_split_id] * 2, phoc_splits[father_id])
        correlated_phocs.append(cur_correlated_phoc)
    correlated_phocs = np.hstack(correlated_phocs)
    return correlated_phocs


def get_n_grams(word, len_ngram):
    '''
    Calculates list of ngrams for a given word.

    @param word: (str)
        Word to calculate ngrams for.
    @param len_ngram: (int)
        Maximal ngram size: n=3 extracts 1-, 2- and 3-grams.
    @return:  List of ngrams as strings.
    '''
    return [word[i:i + len_ngram]for i in range(len(word) - len_ngram + 1)]
