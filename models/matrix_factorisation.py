"""
This script contains the main class for the MF model using the implementation
in the surprise package. It includes the methods for hyperparameter tuning
and model evaluation. The experiments are performed on the Amazon: Groceries
dataset.
"""

import pandas as pd
import numpy as np

from surprise import Dataset
from surprise import Reader
from surprise import SVD

from hyperopt import hp, tpe, fmin, Trials
from functools import partial


def get_recs(predictions):
    """
    Returns the top-k recommendations for a user from a set of predictions.
    Parameters:
      predictions (list): List of predictions, as returned by the test method of
                          an algorithm.
    Returns:
    ` item_recs (list): list of test items, ranked in descending order by
                        predicted rating
    """

    items_ratings = []
    for userID, itemID, _, rating, _ in predictions:
        items_ratings.append((itemID, rating))

    items_ratings.sort(key=lambda x: x[1], reverse=True)
    item_recs = [x[0] for x in items_ratings]

    return item_recs


class MF:

    def __init__(self):
        pass

    def fit(self, train_ratings, lr, reg, d):
        """
        Fits the model using the SVD method to the training data
        Parameters:
          train_ratings (pd.DataFrame): training dataset in the format (user,
                                        item, rating)
          lr (float): learning rate for all model parameters
          reg (float): L2 regularisation for all model parameters
          d (float): number of latent factors
        """
        reader = Reader(rating_scale=(1, 5))
        trainset = Dataset.load_from_df(train_ratings, reader).build_full_trainset()

        self.model = SVD(lr_all=lr, reg_all=reg, n_factors=d)
        self.model.fit(trainset)

    def evaluate_mf(self, test_ratings, full_ratings, k=10, n_random=100):
        """
        Computes mean average recall, mean reciprocal rank and AUC score
        Parameters:
          test_ratings (pd.DataFrame): test dataset in the format (user, item, rating)
          full_ratings (pd.DataFrame): full dataset in the format (user, item, rating)
          k (int): no. of most relevant items
          n_random (int): no. of unobserved items to sample to use for evaluation
        Returns:
          average_AUC (float): mean average AUC
          recall (float): mean average recall @ k
          mrr (float): mean reciprocal rank @ k
        """
        test_ratings_array = test_ratings[['userID', 'itemID']].values
        m = test_ratings_array.shape[0]
        n_item = len(full_ratings['itemID'].unique())

        hits = 0
        rr_agg = 0
        auc_agg = 0
        auc_list = np.zeros(m)

        for instance in range(m):
            testset = []
            u = test_ratings_array[instance, 0]
            i = test_ratings_array[instance, 1]
            user_items = np.array(full_ratings[full_ratings['userID'] == u]['itemID'])
            uo_items = np.setdiff1d(np.arange(n_item), user_items)

            random_uo_items = np.random.choice(uo_items, replace=False, size=n_random)
            random_uo_items = np.array(list(random_uo_items) + [i])

            for item in random_uo_items:
                testset.append((u, item, 0))

            predictions = self.model.test(testset)
            item_recs = get_recs(predictions)

            position = int(np.where(i == np.array(item_recs))[0] + 1)
            if position <= k:
                hits += 1
                rr_agg += 1 / position

            auc_list[instance] = (n_random + 1 - position) / (n_random + 1)
            auc_agg += auc_list[instance]

        recall = hits / m
        mrr = rr_agg / m
        average_auc = auc_agg / m

        return average_auc, recall, mrr

    def objective_mf(self, grid, train_ratings, test_ratings, full_ratings):
        """
        Returns (1 - mean average AUC) for model trained on grid parameters,
        on the validation set, as an objective function to minimise for Bayesian
        Optimisation.
        Parameters:
          grid (dict): dictionary with values of 'd', 'lr', 'reg'
          train_ratings (pd.DataFrame): train dataset in the format (user, item, rating)
          test_ratings (pd.DataFrame): test dataset in the format (user, item, rating)
          full_ratings (pd.DataFrame): full dataset in the format (user, item, rating)
        Returns:
          1 - average_AUC (float): 1 - mean average AUC
        """
        lr = grid['lr']
        reg = grid['reg']
        d = grid['d']

        self.fit(train_ratings, lr, reg, d)

        auc, _, _ = self.evaluate_mf(test_ratings, full_ratings)
        return 1 - auc

    def optimise_params_mf(self, grid, bo_iterations, train_ratings,
                           test_ratings, full_ratings):
        """
        Fits Bayesian Optimisation with Tree Parzen estimators to optimise parameters
        taking values in 'grid', minimising (1 - mean average AUC) on the validation
        set, for model trained on grid parameters.
        Parameters:
          grid (dict): dictionary with values of 'd', 'lr', 'reg'
          bo_iterations (int): number of iterations of Bayesian Optimisation
          train_ratings (pd.DataFrame): train dataset in the format (user, item, rating)
          test_ratings (pd.DataFrame): test dataset in the format (user, item, rating)
          full_ratings (pd.DataFrame): full dataset in the format (user, item, rating)
        Returns:
          results (dict): optimal set of hyperparameters
        """
        fmin_objective = partial(self.objective_mf, train_ratings=train_ratings,
                                 test_ratings=test_ratings, full_ratings=full_ratings)

        trials = Trials()
        results = fmin(fmin_objective, grid, algo=tpe.suggest, trials=trials,
                       max_evals=bo_iterations)

        return results
