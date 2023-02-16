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
    fg_no_numbers: bool = False,
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
    fg_no_numbers: bool, optional
        Remove the numbers
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

    # remove numbers
    if fg_no_numbers:
        corpus = re.sub(r" \d+", " ", corpus)

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


def link_texts_series(df_train_series, df_text, id_series, id_text):
    """
    Using the text indices of a time series, concatenate the text

    Arguments
    ---------
        df_train_series: pd.DataFrame
            Times Series with a text index column
        df_text: pd.DataFrame
            Dataframe storing the texts
        id_series: str
            Name of the column of the Unique identifier of the series
        id_text: str
            Name of the column of the Unique identifier of the texts
    Returns
    -------
        df_conc: pd.DataFrame
            Dataframe containing the time series and the concatenated text
    """
    # Copy our dataframe
    df_temp = df_text.copy()
    suff = id_text.split("_")[1]
    # Unnest the list of texts for each times series
    df_temp = df_train_series.explode(id_text)
    # Convert the Id to Int
    df_temp[id_text] = df_temp[id_text].astype("int64")
    # Join the list and series on the id_texts (previously unnested)
    df_temp = df_temp.merge(df_text, on=id_text, how="left")
    # Group By the series and aggregate on specific features (concatenate the text, list of the speakers)
    df_temp = (
        df_temp.groupby(id_series)
        .agg({"text": lambda x: " ".join(x), "speaker": lambda x: list(x)})
        .reset_index()
        .rename(
            columns={"text": "text_concat_" + suff, "speaker": "list_speakers_" + suff}
        )
    )
    return df_temp


def suppr_footnotes(text):
    """
    Description : fonction pour supprimer les footnotes d'un texte (références biblio, etc)  (à vérifier si permet de supprimer toutes les footnotes)
    - input : colonne 'text'
    - output : colonne 'text' modifiée afin de supprimer les footnotes / citations

    """

    txt = text
    #
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

    ##
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

    #
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
