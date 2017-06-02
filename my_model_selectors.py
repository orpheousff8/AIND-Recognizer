import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    def scoreBIC(self, logLike, number_of_parameters, number_of_data_points):
        return ((-2 * logLike) + (number_of_parameters * np.log(number_of_data_points)))
        
    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        
        model_list = {}
        n_components = self.min_n_components

        while n_components >= self.min_n_components and n_components <=self.max_n_components:
            hmm_model = self.base_model(n_components)
            try:
                test_score = hmm_model.score(self.X, self.lengths)
                bic = self.scoreBIC(test_score, (n_components**2) + (2 * len(self.X[0]) * n_components) -1, len(self.X))
                model_list[hmm_model] = bic
            except:
                pass
            n_components = n_components + 1
            
        if len(model_list) == 0:
            return self.base_model(self.n_constant)

        result_model = max(model_list, key=lambda i: model_list[i])
        return result_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_DIC_Score = float('-inf')
        M = len((self.words).keys()) 

        best_hmm_model = None

        for num_states in range(self.min_n_components, self.max_n_components + 1):

            try:

                hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

                LogLi = hmm_model.score(self.X, self.lengths)

            except:
                LogLi = float("-inf")

            SumLogL = 0

            for each_word in self.hwords.keys():
                X_each_word, lengths_each_word = self.hwords[each_word]

            try:
                SumLogL += hmm_model.score(X_each_word, lengths_each_word)

            except:
                SumLogL += 0

            DIC_Score = LogLi - (1 / (M - 1)) * (SumLogL - (0 if LogLi == float("-inf") else LogLi))

            if DIC_Score > best_DIC_Score:
                best_DIC_Score = DIC_Score
                best_hmm_model = hmm_model

        return best_hmm_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_CV_Score = float('-inf')

        best_hmm_model = None
        if len(self.sequences) < 2:
            return None

        split_method = KFold(n_splits=2)

        for num_states in range(self.min_n_components, self.max_n_components + 1):

            SumLogL = 0
            Counter = 0

            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):

                X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)

                try:
                    hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                    LogL = hmm_model.score(X_test, lengths_test)
                    Counter += 1

                except:
                    LogL = 0

                SumLogL += LogL


            CV_Score = SumLogL / (1 if Counter == 0 else Counter)

            if CV_Score > best_CV_Score:
                best_CV_Score = CV_Score
                best_hmm_model = hmm_model

        return best_hmm_model
