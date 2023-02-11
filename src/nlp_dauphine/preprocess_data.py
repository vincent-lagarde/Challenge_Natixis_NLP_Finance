import re
import contractions
import nltk
import pandas as pd

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def load_data(dict_path):
    df_ecb = pd.read_csv(dict_path["ecb"], index_col=0)
    df_fed = pd.read_csv(dict_path["fed"], index_col=0)
    df_train_series = pd.read_csv(dict_path["train_series"], index_col=0)
    return df_ecb, df_fed, df_train_series


def get_wordnet_pos(tag: str) -> str:
    """_summary_
    TODO docstring

    Parameters
    ----------
    tag : str
        _description_

    Returns
    -------
    str
        _description_
    """
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def text_cleaning(
    corpus: str,
    negation_set: set[str],
    fg_stop_words: bool = False,
    fg_lemmatization: bool = False,
) -> tuple[str, list[str], list[str]]:
    """Text cleaning of a corpus and extraction of mentions and hashtags

    Parameters
    ----------
    corpus : str
        String to clean
    negation_set : set[str]
        Negation words
    fg_stop_words : bool, optional
        Remove or not stop words, by default False
    fg_lemmatization : bool, optional
        Apply or not lemmatization, by default False

    Returns
    -------
    tuple[str, list[str], list[str]]
        corpus : Cleaned string
        mentions : List of mentionned users in the corpus (@'s)
        hashtags : List of hashtags in the corpuss (#'s)
    """

    # lowercase
    corpus = corpus.lower()

    # remove extra newlines
    corpus = re.sub(r"[\r|\n|\r\n]+", " ", corpus)

    # remove URL
    corpus = re.sub(r"https?://[\S]+", "", corpus)

    # remove contractions
    corpus = " ".join([contractions.fix(x) for x in corpus.split()])

    # Remove @ # and any special chars
    corpus = re.sub(r"[\W_]+", " ", corpus)

    # tokenization
    corpus_words = word_tokenize(corpus)

    if fg_stop_words:
        # remove stop words
        stop_words = set(stopwords.words("english")).difference(negation_set)
        corpus_words = [word for word in corpus_words if word not in stop_words]

    if fg_lemmatization:
        # lemmatization
        corpus_pos_tag = nltk.tag.pos_tag(corpus_words)
        corpus_pos_tag = [
            (word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in corpus_pos_tag
        ]
        wordnet_lemmatizer = WordNetLemmatizer()
        corpus_words = [
            wordnet_lemmatizer.lemmatize(word, tag) for (word, tag) in corpus_pos_tag
        ]

    return " ".join(corpus_words)


def concatenate_texts(df_train_series, index_name_column, df_text):
    l_text = []
    l_index = []
    for index, row in df_train_series.iterrows():
        text = ""
        for index_txt in row[index_name_column]:
            text += " " + df_text.iloc[int(index_txt), :].text
        l_text.append(text)
        l_index.append(index)
    df_conc = pd.DataFrame({"index": l_index, "text": l_text})
    return df_conc


def suppr_footnotes(text):
    """
    Description : fonction pour supprimer les footnotes d'un texte (références biblio, etc)  (à vérifier si permet de supprimer toutes les footnotes)
    - input : colonne 'text'
    - output : colonne 'text' modifiée afin de supprimer les footnotes / citations

    """

    txt = text

    try:

        txt_list = txt.split("References", maxsplit=1)
        if len(txt) > 1:
            txt = txt_list[0]
        else:
            1 / 0
    except:
        pass

    try:
        txt_list = txt.split("Footnotes", maxsplit=1)
        if len(txt) > 1:
            txt = txt_list[0]
        else:
            1 / 0
    except:
        pass

    try:
        txt_list = txt.split("      [1]", maxsplit=1)
        if len(txt) > 1:
            txt = txt_list[0]
        else:
            1 / 0
    except:
        pass

    try:
        txt_list = txt.split("See also footnotes", maxsplit=1)
        if len(txt) > 1:
            txt = txt_list[0]
        else:
            1 / 0
    except:
        pass

    try:
        txt_list = txt.split(" References ", maxsplit=1)
        if len(txt) > 1:
            txt = txt_list[0]
        else:
            1 / 0
    except:
        pass

    try:
        txt_list = txt.split(" 1. ", maxsplit=1)
        if len(txt) > 1:
            txt = txt_list[0]
        else:
            1 / 0
    except:
        pass

    try:
        txt_list = txt.split("SEE ALSO", maxsplit=1)
        if len(txt) > 1:
            txt = txt_list[0]
        else:
            1 / 0
    except:
        pass

    try:
        txt_list = txt.split("See also", maxsplit=1)
        if len(txt) > 1:
            txt = txt_list[0]
        else:
            1 / 0
    except:
        pass
    try:
        txt_list = txt.split("Thank you. ", maxsplit=1)
        if len(txt) > 1:
            txt = txt_list[0]
        else:
            1 / 0
    except:
        pass

    try:
        txt_list = txt.split("Thank you for your attention.  ", maxsplit=1)
        if len(txt) > 1:
            txt = txt_list[0]
        else:
            1 / 0
    except:
        pass

    return txt

