"""
This script contains the main class for the BPRMF model, using helper functions
from sampling.py. It includes the methods for hyperparameter tuning and model
evaluation.
"""

from hyperopt import hp, tpe, fmin, Trials

from data.sampling import *


class BPRMF:
    def __init__(self, n_user, n_item, beta, rating_data=True):
        """
        Parameters:
          n_user (int): no. of unique users in the dataset
          n_item (int): no. of unique items in the dataset
          beta (float): unobserved sampling proportion for the negative level
          rating_data (bool): True if Amazon datasets, False if RetailRocket dataset
        """
        self.beta = beta
        self.n_user = n_user
        self.n_item = n_item
        self.rating_data = rating_data

    def attach_data(self, train_ratings, test_ratings):
        """
        Attaches the data to the model class.
        Parameters:
          train_ratings (pd.DataFrame): training dataset in the format (user, item, rating)
          test_ratings (pd.DataFrame): test dataset in the format (user, item, rating)
        """
        self.pos_df = get_pos_df(train_ratings, self.rating_data)
        self.pos_dict = get_pos_dict(self.pos_df)
        self.user_reps = get_user_reps(train_ratings, test_ratings, self.beta,
                                       self.rating_data)
        self.test_ratings = test_ratings

    def fit(self, n_epochs, lr, reg_params, d, trans_cart_ratio=None):
        """
        Fits the BPR-MF model to the training data.
        Parameters:
          n_epochs (int): no. of training epochs
          lr (float): SGD learning rate
          reg_params (dict): regularisation parameters for user, positive item,
                             and negative item latent feature updates
          d (int): no. of latent features for user and item representations
          trans_cart_ratio (float): ratio of 'transaction' sampling weight to
                                    'add-to-cart' sampling weight
        """

        for user_id in list(self.user_reps.keys()):
            self.user_reps[user_id]['embed'] = np.random.normal(scale=0.1, size=(d,))

        self.item_reps = get_item_reps(self.n_item, d)

        if not self.rating_data:
            self.level_weights_dict = {'transaction': trans_cart_ratio, 'addtocart': 1}
        else:
            self.level_weights_dict = None

        self.pos_dist = get_pos_dist(self.pos_df, self.level_weights_dict)

        n_examples = self.pos_df.shape[0]

        for epoch in range(n_epochs):
            for instance in range(n_examples):
                P = sample_pos(self.pos_dist)
                u, i = get_pos_user_item(P, self.pos_dict)
                N = sample_neg(self.user_reps[u])
                j = sample_neg_item(self.user_reps[u], N, self.n_item)

                user_embed, pos_item_embed, neg_item_embed = \
                    sgd_update(self.user_reps[u]['embed'], self.item_reps[i],
                               self.item_reps[j], lr, reg_params)

                self.user_reps[u]['embed'] = user_embed
                self.item_reps[i] = pos_item_embed
                self.item_reps[j] = neg_item_embed

    def evaluate_bprmf(self, test_ratings, k=10, n_random=1000):
        """
        Computes mean recall, mean reciprocal rank and AUC score.
        Parameters:
          test_ratings (pd.DataFrame): test dataset in the format (user, item, rating)
          k (int): no. of most relevant items
          n_random (int): no. of unobserved items to sample to use for evaluation
        Returns:
          average_AUC (float): mean average AUC
          recall (float): mean average recall @ k
          mrr (float): mean reciprocal rank @ k
        """
        test_ratings_array = test_ratings[['userID', 'itemID']].values
        hits, rr_agg, auc_agg = 0, 0, 0
        m = test_ratings_array.shape[0]
        auc_list = np.zeros(m)

        for instance in range(m):
            u = test_ratings_array[instance, 0]
            i = test_ratings_array[instance, 1]
            user_embed = self.user_reps[u]['embed']
            user_items = self.user_reps[u]['all_items']
            uo_items = np.setdiff1d(np.arange(self.n_item), user_items)

            random_uo_items = np.random.choice(uo_items, replace=False,
                                               size=n_random)
            random_uo_items = np.array(list(random_uo_items) + [i])
            user_item_reps = self.item_reps[random_uo_items]

            user_item_scores = np.dot(user_item_reps, user_embed)
            sorted_user_items = random_uo_items[np.argsort(user_item_scores)[::-1]]

            position = np.where(i == sorted_user_items)[0] + 1

            if position <= k:
                hits += 1
                rr_agg += 1 / position

            auc_list[instance] = ((len(random_uo_items) - position)
                                  / len(random_uo_items))
            auc_agg += auc_list[instance]

        recall = hits / m
        mrr = rr_agg / m
        average_auc = auc_agg / m

        return average_auc, recall, mrr

    def objective(self, grid):
        """
        Returns (1 - mean average AUC) for model trained on grid parameters, on the
        validation set, as an objective function to minimise for Bayesian Optimisation.
        Parameters:
          grid (dict): dictionary with values of 'd', 'lr', reg_params for 'u', 'i'
                       and 'j', and for RetailRocket, 'trans_cart_ratio'
        Returns:
          1 - average_AUC (float): 1 - mean average AUC
        """
        d = grid['d']
        lr = grid['lr']
        reg_params = {'u': grid['u'], 'i': grid['i'], 'j': grid['j']}
        n_epochs = grid['n_epochs']

        if not self.rating_data:
            trans_cart_ratio = grid['trans_cart_ratio']
        else:
            trans_cart_ratio = None

        self.fit(n_epochs, lr, reg_params, d, trans_cart_ratio)

        auc, _, _ = self.evaluate_bprmf(self.test_ratings)

        return 1 - auc

    def optimise_params(self, grid, bo_iterations=100):
        """
        Fits Bayesian Optimisation with Tree Parzen estimators to optimise
        parameters taking values in 'grid', minimising (1 - mean average AUC) on
        the validation set, for model trained on grid parameters.
        Parameters:
          grid (dict): dictionary with values of 'd', 'lr', reg_params for 'u',
                      'i' and 'j', and for RetailRocket, 'trans_cart_ratio'
          bo_iterations (int): number of iterations of Bayesian Optimisation
        Returns:
          results (dict): optimal set of hyperparameters
        """
        trials = Trials()
        results = fmin(self.objective, grid, algo=tpe.suggest, trials=trials,
                       max_evals=bo_iterations)

        return results


def sgd_update(user_embed, pos_item_embed, neg_item_embed, lr, reg_params):
    """
    Performs one step of stochastic gradient descent on model parameters,
    optimising BPR-MIN for Matrix Factorisation.
    Parameters:
      user_embed (np.array): d-dimensional latent factors for a user
      pos_item_embed (np.array): d-dimensional latent factors for preferred item
      neg_item_embed (np.array): d-dimensional latent factors for less preferred item
      lr (float): learning rate for SGD
      reg_params (dict): regularisation parameters for user, positive item, and
                         negative item latent feature updates
    Returns:
      user_embed (np.array): latent user features after update
      pos_item_embed (np.array): latent positive item features after update
      neg_item_embed (np.array): latent negative item features after update
    """
    lambda_u, lambda_i, lambda_j = reg_params['u'], reg_params['i'], reg_params['j']

    rating_diff = np.dot(user_embed, pos_item_embed) \
                  - np.dot(user_embed, neg_item_embed)
    factor = 1 / (1 + np.exp(rating_diff))

    d_user = pos_item_embed - neg_item_embed
    d_pos_item = user_embed
    d_neg_item = -user_embed

    user_embed += lr * (factor * d_user - lambda_u * user_embed)
    pos_item_embed += lr * (factor * d_pos_item - lambda_i * pos_item_embed)
    neg_item_embed += lr * (factor * d_neg_item - lambda_j * neg_item_embed)

    return user_embed, pos_item_embed, neg_item_embed
