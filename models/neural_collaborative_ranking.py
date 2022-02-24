"""
This script contains the main class for the NCR model, that inherits from
MFBPR and uses helper functions from sampling.py. It includes the methods
for hyperparameter tuning and model evaluation.
"""

from keras.regularizers import l2
from keras.models import Model
from keras.layers import Embedding, Input, Dense, multiply, concatenate, \
    Flatten, Lambda
from keras.optimizers import Adagrad, SGD, RMSprop
from functools import partial

from models.bayesian_personalised_ranking import *


class NCR(BPRMF):
    def __init__(self, n_user, n_item, beta, rating_data=True, features_dict=None):
        """
        Parameters:
          n_user (int): no. of unique users in the dataset
          n_item (int): no. of unique items in the dataset
          beta (float): unobserved sampling proportion for the negative level
          rating_data (bool): True if rating data is used, False otherwise
          features_dict (dict): dictionary mapping items to features
        """
        super().__init__(n_user, n_item, beta, rating_data)
        self.n_user = n_user
        self.n_item = n_item
        self.beta = beta
        self.rating_data = rating_data
        self.features_dict = features_dict

    def get_model(self, learner, learning_rate, nbpr_dim, layers, reg_layers,
                  reg_embed_nbpr, reg_embed_mlp):
        """
        Returns NCR model, a concatenation of NBPR and MLP models.
        Parameters:
          learner (str): SGD learner: ['adagrad', 'rmsprop', 'sgd']
          learning_rate (float): SGD learning rate
          nbpr_dim (int): dimension of embedding for Neural BPR
          reg_embed_nbpr (float): L2 regularisation for Neural BPR embeddings
          reg_embed_mlp (float: L2 regulaisation for MLP embeddings
          layers (list of int): list of input dimensions for layers in DNCR
          reg_layers (float): L2 regularisation for layers in DNCR
        Returns:
          model (model): Keras NCR Model
        """
        user_input = Input(shape=(1,), dtype='int32')
        item_input_i = Input(shape=(1,), dtype='int32')
        item_input_j = Input(shape=(1,), dtype='int32')

        # NBPR Model
        nbpr_embedding_user = \
            Embedding(input_dim=self.n_user, output_dim=nbpr_dim,
                      embeddings_initializer='random_normal',
                      name='nbpr_user_embedding',
                      embeddings_regularizer=l2(reg_embed_nbpr),
                      input_length=1)
        nbpr_embedding_item = \
            Embedding(input_dim=self.n_item,
                      output_dim=nbpr_dim,
                      embeddings_initializer='random_normal',
                      name='nbpr_item_embedding',
                      embeddings_regularizer=l2(reg_embed_nbpr),
                      input_length=1)

        nbpr_user_latent = Flatten()(nbpr_embedding_user(user_input))
        nbpr_item_latent_i = Flatten()(nbpr_embedding_item(item_input_i))
        nbpr_item_latent_j = Flatten()(nbpr_embedding_item(item_input_j))

        prefer_i = multiply([nbpr_user_latent, nbpr_item_latent_i])
        prefer_j = multiply([nbpr_user_latent, nbpr_item_latent_j])
        prefer_j = Lambda(lambda x: -x)(prefer_j)
        nbpr_vector = concatenate([prefer_i, prefer_j])

        # MLP Model
        mlp_embedding_user = \
            Embedding(input_dim=self.n_user, output_dim=layers[0],
                      embeddings_initializer='random_normal',
                      name='mlp_user_embedding',
                      embeddings_regularizer=l2(reg_embed_mlp),
                      input_length=1)
        mlp_embedding_item = \
            Embedding(input_dim=self.n_item, output_dim=layers[0],
                      embeddings_initializer='random_normal',
                      name='mlp_item_embedding',
                      embeddings_regularizer=l2(reg_embed_mlp),
                      input_length=1)

        mlp_user_latent = Flatten()(mlp_embedding_user(user_input))
        mlp_item_latent_i = Flatten()(mlp_embedding_item(item_input_i))
        mlp_item_latent_j = Flatten()(mlp_embedding_item(item_input_j))
        mlp_item_latent_j = Lambda(lambda x: -x)(mlp_item_latent_j)

        mlp_vector = concatenate([mlp_user_latent, mlp_item_latent_i,
                                  mlp_item_latent_j])

        if self.features_dict is not None:
            i_j_features = Input(shape=(60,), dtype='float32')
            mlp_vector = concatenate([mlp_vector, i_j_features])

        for l in range(1, len(layers)):
            layer = Dense(layers[l], kernel_regularizer=l2(reg_layers),
                          activation='relu', kernel_initializer='he_normal',
                          name="layer%d" % l)
            mlp_vector = layer(mlp_vector)

        # Concatenate NBPR and MLP models
        concat_vector = concatenate([nbpr_vector, mlp_vector])
        prob = Dense(1, activation='sigmoid', kernel_initializer='random_normal',
                     name='prob')(concat_vector)

        if self.features_dict is not None:
            model = Model(inputs=[user_input, item_input_i, item_input_j,
                                  i_j_features], outputs=prob)

        else:
            model = Model(inputs=[user_input, item_input_i, item_input_j],
                          outputs=prob)

        if learner.lower() == "adagrad":
            model.compile(optimizer=Adagrad(lr=learning_rate),
                          loss='binary_crossentropy')

        elif learner.lower() == "rmsprop":
            model.compile(optimizer=RMSprop(lr=learning_rate),
                          loss='binary_crossentropy')

        else:
            model.compile(optimizer=SGD(lr=learning_rate),
                          loss='binary_crossentropy')

        return model

    def get_training_input(self, neg_sampling_ratio, trans_cart_ratio=None):
        """
        Obtains training examples for SGD, with 'neg_sampling_ratio' times more
        negative examples than positive examples.
        Parameters:
          neg_sampling_ratio (float): number of negative (u,j,i) samples to sample
                                      for every positive (u,i,j) sample
          trans_cart_ratio (float): ratio of 'transaction' sampling weight to
                                   'add-to-cart' sampling weight
        Returns:
          users (np.array): users in training data for NCR
          items_i (np.array): i_items in training data for NCR
          items_j (np.array): j_items in training data for NCR
          labels (np.array): 1 if (u,i,j) represents a positive sample, 0 if it
                             represents a negative sample
          i_j_features (np.array): concatenation of features for item i and j
        """
        n_examples = self.pos_df.shape[0]
        users, items_i, items_j, labels, i_j_features = [], [], [], [], []

        if not self.rating_data:
            self.level_weights_dict = {'transaction': trans_cart_ratio, 'view': 1}
        else:
            self.level_weights_dict = None

        self.pos_dist = get_pos_dist(self.pos_df, self.level_weights_dict)

        for instance in range(n_examples):
            L = sample_pos(self.pos_dist)
            u, i = get_pos_user_item(L, self.pos_dict)
            N = sample_neg(self.user_reps[u])
            j = sample_neg_item(self.user_reps[u], N, self.n_item)

            users.append(u)
            items_i.append(i)
            items_j.append(j)
            labels.append(1)
            if self.features_dict is not None:
                i_j_features.append(np.append(self.features_dict[i],
                                              self.features_dict[j]))

            for count in range(int(neg_sampling_ratio)):
                N = sample_neg(self.user_reps[u])
                j = sample_neg_item(self.user_reps[u], N, self.n_item)
                users.append(u)
                items_i.append(j)
                items_j.append(i)
                labels.append(0)
                if self.features_dict is not None:
                    i_j_features.append(np.append(self.features_dict[j],
                                                  self.features_dict[i]))

        if self.features_dict is not None:
            return np.array(users), np.array(items_i), np.array(items_j), \
                   np.array(labels), np.array(i_j_features)

        else:
            return np.array(users), np.array(items_i), np.array(items_j), \
                   np.array(labels)

    def fit_ncr(self, learner, learning_rate, nbpr_dim, layers, reg_layers,
                reg_embed_nbpr, reg_embed_mlp, n_epochs,
                neg_sampling_ratio, batch_size, trans_cart_ratio=None):
        """
        Fits the NCR model to the training data.
        Parameters:
          learner (str): SGD learner to use: ['adagrad', 'rmsprop', 'adam', 'sgd']
          learning_rate (float): SGD learning rate
          nbpr_dim (int): dimension of embedding for Neural BPR
          reg_embed_nbpr (float): L2 regularisation for Neural BPR embeddings
          reg_embed_mlp (float: L2 regulaisation for MLP embeddings
          layers (list of int): list of input dimensions for layers in DNCR
          reg_layers (float): L2 regularisation for layers in DNCR
          n_epochs (int): number of training epochs
          neg_sampling_ratio (float): number of negative (u,j,i) samples to sample
                                      for every positive (u,i,j) sample
          batch_size (int): number of training examples per iteration
          trans_cart_ratio (float): ratio of 'transaction' sampling weight to
                                    'add-to-cart' sampling weight
        """

        for epoch in range(n_epochs):
            if self.features_dict is not None:
                users, items_i, items_j, labels, i_j_features = \
                    self.get_training_input(neg_sampling_ratio=neg_sampling_ratio,
                                            trans_cart_ratio=trans_cart_ratio)

                self.model = self.get_model(learner, learning_rate, nbpr_dim, layers,
                                            reg_layers, reg_embed_nbpr, reg_embed_mlp)

                self.model.fit(x=[users, items_i, items_j, i_j_features], y=labels,
                               batch_size=batch_size, epochs=n_epochs, shuffle=True)
            else:
                users, items_i, items_j, labels = \
                    self.get_training_input(neg_sampling_ratio=neg_sampling_ratio,
                                            trans_cart_ratio=trans_cart_ratio)

                self.model = self.get_model(learner, learning_rate, nbpr_dim, layers,
                                            reg_layers, reg_embed_nbpr, reg_embed_mlp)

                self.model.fit(x=[users, items_i, items_j], y=labels,
                               batch_size=batch_size, epochs=n_epochs, shuffle=True)

    def evaluate_ncr(self, test_ratings, k=10, n_random=100):
        """
        Computes mean average recall, mean reciprocal rank and AUC score.
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
            user_items = self.user_reps[u]['all_items']

            uo_items = np.setdiff1d(np.arange(self.n_item), user_items)

            random_uo_items = np.random.choice(uo_items, replace=False,
                                               size=n_random)

            if self.features_dict is not None:
                position = get_position(u=u, i=i, uo_items=random_uo_items,
                                        model=self.model,
                                        features_dict=self.features_dict)
            else:
                position = get_position(u=u, i=i, uo_items=random_uo_items,
                                        model=self.model)

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

    def objective_ncr(self, grid, n_epochs, neg_sampling_ratio):
        """
        Returns (1 - mean average AUC) for NCR model trained on grid parameters,
        on the validation set, as an objective function to minimise for Bayesian
        Optimisation.
        Parameters:
          grid (dict): dictionary with values of 'learner', 'learning rate',
                      'nbpr_dim', 'layers', 'reg_layers', 'reg_embed_nbpr', 'reg_embed_mlp',
                      'batch_size', 'trans_cart_ratio'
          n_epochs (int): number of training epochs
          neg_sampling_ratio (float): number of negative (u,j,i) samples to
                                      sample for every positive (u,i,j) sample
          k (int): no. of most relevant items
        Returns:
          1 - average_AUC (float): 1 - mean average AUC
        """
        if not self.rating_data:
            trans_cart_ratio = grid['trans_cart_ratio']
        else:
            trans_cart_ratio = None

        self.fit_ncr(grid['learner'], grid['learning_rate'], grid['nbpr_dim'],
                     grid['layers'], grid['reg_layers'], grid['reg_embed_nbpr'],
                     grid['reg_embed_mlp'], n_epochs, neg_sampling_ratio,
                     grid['batch_size'], trans_cart_ratio)

        auc, _, _ = self.evaluate_ncr(self.test_ratings, self.user_reps)

        return 1 - auc

    def optimise_params_ncr(self, grid, n_epochs, neg_sampling_ratio,
                            bo_iterations=100):
        """
        Fits Bayesian Optimisation with Tree Parzen estimators to optimise
        parameters taking values in 'grid', minimising (1 - mean average AUC) on
        the validation set, for NCR model trained on grid parameters.
        Parameters:
          grid (dict): dictionary with values of 'learner', 'learning rate',
                      'nbpr_dim', 'layers', 'reg_layers', 'reg_embed_nbpr',
                      'reg_embed_mlp', 'batch_size', 'trans_cart_ratio'
          n_epochs (int): number of training epochs
          neg_sampling_ratio (float): number of negative (u,j,i) samples to sample
                                      for every positive (u,i,j) sample
          bo_iterations (int): number of iterations of Bayesian Optimisation
        Returns:
          results (dict): optimal set of hyperparameters
        """
        fmin_objective = partial(self.objective_ncr, n_epochs=n_epochs,
                                 neg_sampling_ratio=neg_sampling_ratio, k=k)

        trials = Trials()
        results = fmin(fmin_objective, grid, algo=tpe.suggest,
                       trials=trials, max_evals=bo_iterations)

        return results


def get_position(u, i, uo_items, model, features_dict=None):
    """
    Returns position of item i, in the list of unobserved items together with i.
    Parameters:
      u (int): userID
      i (int): itemID
      uo_items (list): list of (n_random) unobserved items to user u
      model (model): Trained keras NCR model
      features_dict (dict): dictionary mapping user-item tuples to features
    Returns:
      poition (int): position of i in [uo_items, i]
    """
    u_vector = np.full(len(uo_items), u, dtype='int32')
    i_vector = np.full(len(uo_items), i, dtype='int32')

    if features_dict is not None:
        i_items_features = [np.concatenate([features_dict[i], features_dict[item]])
                            for item in uo_items]
        items_i_features = [np.concatenate([features_dict[item], features_dict[i]])
                            for item in uo_items]
        score_i_items = model.predict([u_vector, i_vector, np.array(uo_items),
                                       np.array(i_items_features)])
        score_items_i = model.predict([u_vector, np.array(uo_items), i_vector,
                                       np.array(items_i_features)])
    else:
        score_i_items = model.predict([u_vector, i_vector, np.array(uo_items)])
        score_items_i = model.predict([u_vector, np.array(uo_items), i_vector])

    # Number of items that i is preferred over
    i_prefover_items = score_i_items - score_items_i

    position = sum(i_prefover_items < 0) + 1

    return int(position)
