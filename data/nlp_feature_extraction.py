"""
This script contains the classes for extracting LDA and W2V features from item descriptions.
"""

import numpy as np
import pandas as pd
import re

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem import WordNetLemmatizer

from gensim.models import LdaModel, KeyedVectors
import gensim.corpora as corpora


def get_item_descriptions(df):
    """
    Returns a dataframe containing each unique item and its most recent
    item description.
    Parameters:
      df (pd.DataFrame): full Amazon dataframe
    Returns:
      item_descriptions (pd.DataFrame): most recent item description for each
                                        unique item
    """
    most_recent_items = \
        df.loc[df.reset_index().groupby(['itemID'])['unixTime'].idxmax()]
    item_descriptions = most_recent_items[['itemID', 'description']]

    return item_descriptions


def preprocessing(text, digits, stop_words):
    """
    Performs lemmatisation, removes stopwords, replaces numbers with digits, lower-cases.
    Parameters:
      text (str): Text for preprocessing
    Returns:
      text (str): Preprocessed keywords from text
    """
    lemmatizer = WordNetLemmatizer()

    for key, value in digits.items():
        text = text.replace(key, value)

    text = text.lower()
    text = re.sub(r"[^a-z]", " ", text)
    text = [lemmatizer.lemmatize(word) for word in text.split()
            if (word not in stop_words) and (len(word) > 2)]

    return text


def preprocessing_df(item_descriptions):
    """
    Creates a column 'word_list' with preprocessed keywords from the 'description'
    column of a dataframe.
    Parameters:
      item_descriptions (pd.DataFrame): full dataframe with 'description' column
    """
    item_descriptions['description'] = item_descriptions['description'].astype(str)
    item_descriptions['word_list'] = \
        item_descriptions['description'].apply(lambda x: preprocessing(x))


class LDA:
    def __init__(self, n_topics):
        self.n_topics = n_topics

    def get_corpus(self, item_descriptions):
        """
        Obtains corpus of item descriptions in bag-of-words format.
        Parameters:
          item_descriptions (pd.DataFrame): preprocessed dataframe
        Returns:
          corpus (list): corpus of item descriptions
        """
        description_words = list(item_descriptions['word_list'])
        id_words_dict = corpora.Dictionary(description_words)

        corpus = [id_words_dict.doc2bow(text) for text in description_words]

        return corpus

    def get_lda_features(self, corpus):
        """
        Obtains features from LDA from each item description, as the vector of
        topic probabilities.
        Parameters:
          corpus (list): corpus of item descriptions
        Returns:
          lda_features (np.array): rows: descriptions, columns: probability of
                               description being in respective topic
        """
        lda_model = LdaModel(corpus=corpus, num_topics=self.n_topics)
        lda_features = np.zeros((len(corpus), self.n_topics))

        for description in range(len(corpus)):
            topic_dist = lda_model.get_document_topics(corpus[description])

            for topic, prob in topic_dist:
                lda_features[description, topic] = prob

        return lda_features


class TFIDF_W2V:

    def __init__(self, n_components):
        self.n_components = n_components

    def fit_tfidf(self, item_descriptions):
        """
        Fits tfidf model to item_descriptions.
        Parameters:
          item_descriptions (pd.DataFrame): preprocessed dataframe
        Returns:
          filtered_vocab (list): words with idf exceeding log(2)
          tfidf_model (model): trained tfidf model
          tfidf_vocab (list): full vocabulary from 'word_list' column of
                              item_descriptions
        """
        tfidf_model = TfidfVectorizer()
        tfidf_model.fit(item_descriptions['word_list'].apply(lambda x: ' '.join(x)))
        tfidf_vocab = list(tfidf_model.vocabulary_.keys())

        word_idf = pd.DataFrame()
        word_idf['word'] = list(tfidf_vocab)
        word_idf['idf'] = list(tfidf_model.idf_)
        word_idf['idf'] = word_idf['idf'].astype('float32')
        word_idf = word_idf[word_idf['idf'] > np.log(2)]

        filtered_vocab = list(word_idf['word'])

        return filtered_vocab, tfidf_model, tfidf_vocab

    def get_w2v_idf(self, filtered_vocab, tfidf_model, tfidf_vocab, w2v_model):
        """
        Creates array of Word2Vec embeddings for words, weighted by idf score.
        Parameters:
          filtered_vocab (list): words with idf exceeding log(2)
          tfidf_model (model): trained tfidf model
          tfidf_vocab (list): full vocabulary from 'word_list' column of
                              item_descriptions
          w2v_model (model): pretrained Word2Vec embeddings
        Returns:
          vocab (list): words contained in intersection of filtered_vocab and w2v_vocab
          w2v_idf (np.array): rows: words (contained in 'vocab');
                              columns: Word2Vec embedding coefficient
                              weighted by idf of respective word
        """
        w2v_vocab = w2v_model.vocab.keys()

        vocab = list(set(filtered_vocab).intersection(set(w2v_vocab)))

        embedding = np.zeros((300, len(vocab)))
        vocab_idf = []

        i = 0
        for word in vocab:
            embedding[:, i] = w2v_model.wv[word]
            index = tfidf_vocab.index(word)
            vocab_idf.append(tfidf_model.idf_[index])
            i += 1

        scaler = StandardScaler()
        embedding = scaler.fit_transform(embedding)

        pca = PCA(n_components=self.n_components)
        embedding = pca.fit_transform(X=embedding.transpose()).transpose()

        w2v_idf = np.multiply(embedding, vocab_idf)

        return w2v_idf, vocab

    def get_w2v_average_tfidf(self, description, w2v_idf, vocab):
        """
        Creates array of tfidf-weighted Word2Vec embeddings for description.
        Parameters:
          vocab (list): words contained in intersection of filtered_vocab and
                        w2v_vocab
          description (str): description
          w2v_idf (np.array): rows: words (contained in 'vocab');
                              columns: Word2Vec embedding coefficient,
                              weighted by idf of respective word
        Returns:
          description_w2v_idf (np.array): rows: descriptions;
          columns: averaged Word2Vec embedding coefficient, weighted by tfidf
                   score of words in description
        """
        description_w2v_idf = np.zeros((self.n_components,
                                        np.max(1, len(description))))

        i = 0
        for word in description:
            if word in vocab:
                description_w2v_idf[:, i] = w2v_idf[:, vocab.index(word)]
                i += 1

        description_w2v_tfidf = description_w2v_idf.mean(axis=1)

        return description_w2v_tfidf


def run_lda(item_descriptions, n_topics):
    """
    Returns an array of LDA features from item_descriptions.
    Parameters:
      item_descriptions (pd.DataFrame): most recent item description
                                        for each unique item
      n_topics (int): number of LDA topics
    Returns:
      features (np.array): rows: descriptions; columns: probability of
                           description being in respective topic
    """
    lda = LDA(n_topics)
    lda_corpus = lda.get_corpus(item_descriptions)
    features = lda.get_lda_features(lda_corpus)

    return features


def run_w2v(item_descriptions, n_components):
    """
    Returns an array of tfidf-weighted Word2Vec (W2V) features from
    item_descriptions.
    Parameters:
      item_descriptions (pd.DataFrame): most recent item description for each
                                        unique item
      n_components (int): features dimension (number of PCA components)
    Returns:
      features (np.array): rows: descriptions; columns: tfidf-weighted Word2Vec
                           features
    """
    tfidf_w2v = TFIDF_W2V(n_components)
    filtered_vocab, tfidf_model, tfidf_vocab = TFIDF_W2V.fit_tfidf(item_descriptions)

    w2v_idf, vocab = tfidf_w2v.get_w2v_idf(filtered_vocab, tfidf_model,
                                           tfidf_vocab, w2v_model)
    features = item_descriptions['word_list'].apply(
        lambda x: TFIDF_W2V.get_w2v_average_tfidf(x, w2v_idf, vocab))
    features = np.concatenate(features).reshape((item_descriptions.shape[0],
                                                 n_components))

    return features


def features_to_df(lda_features, w2v_features, item_descriptions, lda_columns, w2v_columns):
    """
    Appends LDA and W2V features as columns to item_descriptions.
    Parameters:
      lda_features (np.array): rows: descriptions; columns: LDA features
      w2v_features (np.array): rows: descriptions; columns: W2V features
      item_descriptions (pd.DataFrame): most recent item description for each
                                        unique item
    """
    for column in lda_columns:
        item_descriptions[column] = lda_features[:, lda_columns.index(column)]

    for column in w2v_columns:
        item_descriptions[column] = w2v_features[:, w2v_columns.index(column)]


def get_features_dict(features_df, features='lda', n_features=30):
    """
    Creates a dictionary mapping items to their respective features.
    Parameters:
      features_df (pd.DataFrame): dataframe containing itemID and features
                                  created by LDA or Word2Vec
      features (str): 'lda' for LDA features 'w2v' for Word2Vec features
      n_features (int): dimension of features
    Returns:
      features_dict (dict): dictionary mapping items to LDA or Word2Vec features
    """

    if features == 'lda':
        lda_columns = ['Topic_' + str(i) for i in range(n_features)]
        review_features = [np.array(row[lda_columns]) for index, row in
                           features_df.iterrows()]
    else:
        w2v_columns = ['W2V_Emb_' + str(i) for i in range(n_features)]
        review_features = [np.array(row[w2v_columns]) for index, row in
                           features_df.iterrows()]

    features_dict = dict(zip(list(features_df['itemID']), review_features))

    return features_dict
