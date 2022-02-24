"""
This script contains helper functions for the sampling of (u,i,j) for the
multiple feedback sampling scheme.
"""

import numpy as np
from collections import OrderedDict


def get_dist(weights, counts, levels=None):
    """
    Returns sampling distribution of levels, given level weights and respective
    counts.
    Parameters:
      weights (np.array): weights of levels
      counts (np.array): number of ratings
      levels (list): names of levels corresponding to weights
    Returns:
      dist (dict): sampling distribution for levels
    """
    probs = weights * counts / sum(weights * counts)

    if levels is None:
        dist = dict(zip(list(weights), probs))
    else:
        dist = dict(zip(levels, probs))

    return dist


def get_pos_df(df, rating_data=True, pos_levels=('transaction', 'add-to-cart')):
    """
    For Amazon datasets: returns positive level training dataset such that each
    user's ratings are greater or equal to their mean rating.
    Parameters:
      df (pd.DataFrame): training dataset in the format (user, item, rating)
      rating_data (bool): True if Amazon datasets, False if RetailRocket dataset
      pos_levels (tuple): names of positive levels in dataset; default
                          ('transaction', 'add-to-cart') for RetailRocket
    Returns:
      pos_df (pd.DataFrame): positive level training dataset
    """
    if rating_data is False:
        pos_df = df[df['rating'].isin(list(pos_levels))]
    else:
        user_mean_ratings = \
            df[['userID', 'rating']].groupby('userID').mean().reset_index()
        user_mean_ratings.rename(columns={'rating': 'mean_rating'}, inplace=True)

        df = df.merge(user_mean_ratings, on='userID')
        pos_df = df[df['rating'] >= df['mean_rating']]

    return pos_df


def get_pos_dist(pos_df, level_weights_dict=None):
    """
    Obtains positive level sampling distribution for discrete ratings.
    Parameters:
      pos_df (pd.DataFrame): positive level training dataset
      level_weights_dict (dict): dictionary mapping levels to weights for
                                 RetailRocket
    Returns:
      pos_dist (dict): positive level sampling distribution
    """
    pos_counts = pos_df['rating'].value_counts()

    if level_weights_dict is None:
        pos_dist = get_dist(pos_counts.index.values, pos_counts.values)
    else:
        levels = list(pos_counts.index.values)
        weights = [level_weights_dict[level] for level in levels]
        pos_dist = get_dist(weights, pos_counts.values, levels)

    return pos_dist


def get_pos_dict(pos_df):
    """
    Creates ordered dictionary mapping each rating in 'pos_df' to all observed
    (user, item) interactions with the corresponding rating.
    Parameters:
      pos_df (pd.DataFrame): positive level training dataset
    Returns:
      pos_dict (dict): collection of all (user, item) interaction tuples for
                       each positive feedback channel
    """

    pos_counts = pos_df['rating'].value_counts()
    pos_dict = OrderedDict()

    for rating in pos_counts.index.values:
        pairs = [tuple(x) for x in pos_df[pos_df['rating'] == rating]
        [['userID', 'itemID']].values]
        pos_dict[rating] = pairs

    return pos_dict


def get_user_reps(df, test_ratings, beta, rating_data=True):
    """
    Creates a dictionary, 'user_reps', mapping user ID (keys) to user-specific
    information (values):
    'mean_rating': the mean rating of the user
    'items': items interacted with by user (training)
    'all items': all items interacted with by user (training and test)
    'neg_items': negative items interacted with by user (training)
    'neg_dist': user's negative level sampling distribution
    Parameters:
      df (pd.DataFrame): training dataset in the format (user, item, rating)
      test_ratings (pd.DataFrame): test dataset in the format (user, item, rating)
      beta (float): unobserved sampling proportion for the negative level
      rating_data (bool): True if Amazon datasets, False if RetailRocket dataset
    Returns:
      user_reps (dict): representations for all `m` unique users
    """
    user_reps = {}
    users = list(df['userID'].unique())
    ratings = sorted(df['rating'].unique(), reverse=True)

    for user_id in users:
        user_reps[user_id] = {}
        user_item_ratings = df[df['userID'] == user_id][['itemID', 'rating']]
        user_reps[user_id]['mean_rating'] = user_item_ratings['rating'].mean()
        user_reps[user_id]['items'] = list(user_item_ratings['itemID'])
        user_reps[user_id]['all_items'] = list(set(user_reps[user_id]['items']).union(
            set(list(test_ratings[test_ratings['userID'] == user_id]['itemID']))))
        user_reps[user_id]['neg_items'] = OrderedDict()

        # Define sampling distribution for less preferred item, j
        if rating_data:
            for rating in ratings:
                if rating < user_reps[user_id]['mean_rating']:
                    user_reps[user_id]['neg_items'][rating] = \
                        list(user_item_ratings[user_item_ratings['rating'] == rating]['itemID'])

            neg_ratings = list(user_reps[user_id]['neg_items'].keys())
            neg_counts = [len(user_reps[user_id]['neg_items'][key]) for key in neg_ratings]

            if sum(neg_counts) != 0:
                if rating_data:
                    user_reps[user_id]['neg_dist'] = \
                        get_dist(1 / np.array(neg_ratings), neg_counts, levels=neg_ratings)
            else:
                user_reps[user_id]['neg_dist'] = {-1: 1.0}
                continue
        else:
            user_reps[user_id]['neg_dist']['view'] = 1

        for key in user_reps[user_id]['neg_dist'].keys():
            user_reps[user_id]['neg_dist'][key] = \
                user_reps[user_id]['neg_dist'][key] * (1 - beta)
            user_reps[user_id]['neg_dist'][-1] = beta


def get_item_reps(n, d):
    """
    FOR BPRMF
    Initializes item latent features from a Normal Distribution with mean = 0,
    sd = 0.1.
    Parameters:
      n (int): no. of unique items in the dataset
      d (int): no. of latent features for user and item representations
    Returns:
      item_reps ((n, d) np.array): d-dimensional initialised latent features for
                                   all n items
    """
    item_reps = np.random.normal(scale=0.1, size=(n, d))

    return item_reps


def sample_pos(pos_dist):
    """
    Samples a positive level P from p(P).
    Parameters:
      pos_dist (dict): positive level sampling distribution
    Returns:
      P (int): positive level
    """
    P = np.random.choice(list(pos_dist.keys()), p=list(pos_dist.values()))

    return P


def get_pos_user_item(P, pos_user_item_dict):
    """
    Given a positive level P, samples a user-item pair (u,i) from p(u,i|P).
    Parameters:
      P (int): positive level
      pos_user_item_dict (dict): dictionary of positive levels P (keys)
                                 mapping to all (user, item) interactions
                                 corresponding to P (values)
    Returns:
      (u, i) (int, int): user ID, positive item ID sampled from P
    """
    u, i = pos_user_item_dict[P][np.random.randint(0, len(pos_user_item_dict[P]))]

    return u, i


def sample_neg(user_rep):
    """
    Samples a negative level N, given the user_rep dictionary, which contains
    the user's negative level sampling distribution.
    Parameters:
      user_rep (dict): user representation (see get_user_reps)
    Returns:
      N (int): negative level
    """
    N = np.random.choice(list(user_rep['neg_dist'].keys()),
                         p=list(user_rep['neg_dist'].values()))

    return N


def sample_neg_item(user_rep, N, n_item):
    """
    Samples the negative item j to complete the triplet (u, i, j).
    If N is an explicit negative channel, sample uniformly from the user's
    negative items; otherwise sample uniformly from all items the user did not
    interact with.
    Parameters:
      user_rep (dict): user representation
      N (int): negative level, either -1 (unobserved) or explicit negative level
      n_item (int): no. of unique items in the dataset
    Returns:
      j (int): sampled negative item ID
    """
    if N != -1:
        j = np.random.choice(user_rep['neg_items'][N])
    else:
        j = np.random.choice(np.setdiff1d(np.arange(n_item), user_rep['items']))

    return j
