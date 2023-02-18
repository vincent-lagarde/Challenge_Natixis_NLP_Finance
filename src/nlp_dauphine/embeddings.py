from collections import Counter, OrderedDict
from nltk.tokenize import word_tokenize
import numpy as np
from tqdm import tqdm


def vocabulary(corpus, voc_threshold=0):
    """
    Function using word counts to build a vocabulary - can be improved with a second parameter for
    setting a frequency threshold
    Params:
        corpus (list of list of strings): corpus of sentences
        voc_threshold (int): maximum size of the vocabulary (0 means no limit !)
    Returns:
        vocabulary (dictionary): keys: list of distinct words across the corpus
                                 values: indexes corresponding to each word sorted by frequency
    """
    # Setting limits
    voc_threshold = voc_threshold if voc_threshold else int(10e5)

    # Count each words of our corpus
    word_counts = dict(Counter(word_tokenize(" ".join(corpus))))

    # Sort them decreasingly
    sorted_word_counts = OrderedDict(
        sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    )

    # Create our vocabulary (no more than the parameter voc_threshold)
    vocabulary_word_counts = dict(list(sorted_word_counts.items())[:voc_threshold])

    vocabulary = {
        list(vocabulary_word_counts.keys())[i]: i
        for i in range(len(vocabulary_word_counts))
    }

    # Add 'Unknown' word
    vocabulary = {**vocabulary, "UNK": len(vocabulary_word_counts)}
    vocabulary_word_counts = {**vocabulary_word_counts, "UNK": 0}

    return vocabulary, vocabulary_word_counts


def co_occurence_matrix(corpus, vocabulary, window=0, distance_weighting=False):
    """
    Params:
        corpus (list of list of strings): corpus of sentences
        vocabulary (dictionary): words to use in the matrix
        window (int): size of the context window; when 0, the context is the whole sentence
        distance_weighting (bool): indicates if we use a weight depending on the distance between words for co-oc counts
    Returns:
        matrix (array of size (len(vocabulary), len(vocabulary))): the co-oc matrix, using the same ordering as the vocabulary given in input
    """
    l = len(vocabulary)
    M = np.zeros((l, l))
    for sent in tqdm(corpus, desc="co-occurence matrix"):
        # Get the sentence as a list of words
        sent = word_tokenize(sent)
        # Obtain the indexes of the words in the sentence from the vocabulary
        sent_idx = [vocabulary[word] if word in vocabulary else -1 for word in sent]
        # Go through the indexes and add 1 / dist(i,j) to M[i,j] if words of index i and j appear in the same window
        for i, idx in enumerate(sent_idx):
            # If we consider a limited context:
            if window > 0:
                # Create a list containing the indexes that are on the left of the current index 'idx_i'
                l_ctx_idx = sent_idx[max(i - window, 0) : i]
            # If the context = ... is the entire document:
            else:
                # The list containing the left context is easier to create
                l_ctx_idx = sent_idx[:i]
            # Go through the list and update M[i,j]:
            for j, idx_j in enumerate(l_ctx_idx):
                # We know the word is in the vocabulary
                if j > -1:
                    weight = 1.0
                    if distance_weighting:
                        weight /= abs(i - j)
                    M[idx, idx_j] += weight * 1.0
                    M[idx_j, idx] += weight * 1.0
                # Unkwnown word
                else:
                    weight = 1.0
                    if distance_weighting:
                        weight /= abs(i - j)
                    M[idx, l] += weight * 1.0
                    M[l, idx] += weight * 1.0
    return M


def euclidean(u, v):
    return np.linalg.norm(u - v)


def length_norm(u):
    return u / np.sqrt(u.dot(u))


def cosine(u, v):
    return 1.0 - length_norm(u).dot(length_norm(v))


def sentence_representations(texts, vocabulary, embeddings, np_func=np.mean):
    """
    Represent the sentences as a combination of the vector of its words.
    Parameters
    ----------
    texts : a list of sentences
    vocabulary : dict
        From words to indexes of vector.
    embeddings : Matrix containing word representations
    np_func : function (default: np.sum)
        A numpy matrix operation that can be applied columnwise,
        like `np.mean`, `np.sum`, or `np.prod`.
    Returns
    -------
    np.array, dimension `(len(texts), embeddings.shape[1])`
    """
    representations = np.zeros((len(texts), embeddings.shape[1]))
    count = 0
    for text in tqdm(texts):
        vec_rep = list(
            map(
                lambda x: embeddings[vocabulary[x]]
                if (x in vocabulary.keys())
                else embeddings[vocabulary["UNK"]],
                text.split(),
            )
        )
        vec_rep_2D = np.stack(vec_rep, axis=0)
        transform_vec = np_func(vec_rep_2D, axis=0)
        representations[count] = transform_vec
        count += 1
    return representations
