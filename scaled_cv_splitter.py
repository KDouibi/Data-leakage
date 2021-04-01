"""
--Created on 03/02/21
@author: Khalida Douibi
The present script aims to personalize the Scikit-learn splitter from Cross-validation class. To be easily integrated to Personnalized classfier and to avoid data leakage by scaling within each
KFold (overcome overfitting in production).
"""
import warnings
from abc import ABC
import numpy as np
from sklearn import metrics
from sklearn.metrics import auc, classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelBinarizer, MinMaxScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer, Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.utils import indexable
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples, check_array, check_random_state, _deprecate_positional_args, column_or_1d
from config import FREQUENCY_TO_TRIGGER
from maa.analysis.eeg.eeg_reader import get_xy
from sklearn.model_selection import StratifiedShuffleSplit, TimeSeriesSplit, KFold, StratifiedKFold, GroupKFold, RepeatedStratifiedKFold
from ml_config import TEST_SIZE, CV_SPLITS, RANDOM_STATE
import matplotlib.pyplot as plt

class ScaledStratifiedKFold(StratifiedKFold, ABC):

    def __init__(self, id_subject, sampling, n_splt, strat_scaler, acquisition):
        """sampling is the strategy to use: Cross validation or Train_test_split """
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.id_subject = id_subject
        self.sampling_strategy = sampling
        #self.type_crossv = type_cv
        self.acquisition_version = acquisition
        self.type_scaler = strat_scaler
        self.n_splits = n_splt
        self.random_state = RANDOM_STATE
        self.shuffle = False

    # from StratifiedKFold
    def _make_test_folds(self, X, y=None):
        rng = check_random_state(self.random_state)
        y = np.asarray(y)
        type_of_target_y = type_of_target(y)
        allowed_target_types = ('binary', 'multiclass')
        if type_of_target_y not in allowed_target_types:
            raise ValueError(
                'Supported target types are: {}. Got {!r} instead.'.format(
                    allowed_target_types, type_of_target_y))

        y = column_or_1d(y)

        _, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
        # y_inv encodes y according to lexicographic order. We invert y_idx to
        # map the classes so that they are encoded by order of appearance:
        # 0 represents the first label appearing in y, 1 the second, etc.
        _, class_perm = np.unique(y_idx, return_inverse=True)
        y_encoded = class_perm[y_inv]

        n_classes = len(y_idx)
        y_counts = np.bincount(y_encoded)
        min_groups = np.min(y_counts)
        if np.all(self.n_splits > y_counts):
            raise ValueError("n_splits=%d cannot be greater than the"
                             " number of members in each class."
                             % (self.n_splits))
        if self.n_splits > min_groups:
            warnings.warn(("The least populated class in y has only %d"
                           " members, which is less than n_splits=%d."
                           % (min_groups, self.n_splits)), UserWarning)

        # Determine the optimal number of samples from each class in each fold,
        # using round robin over the sorted y. (This can be done direct from
        # counts, but that code is unreadable.)
        y_order = np.sort(y_encoded)
        allocation = np.asarray(
            [np.bincount(y_order[i::self.n_splits], minlength=n_classes)
             for i in range(self.n_splits)])

        # To maintain the data order dependencies as best as possible within
        # the stratification constraint, we assign samples from each class in
        # blocks (and then mess that up when shuffle=True).
        test_folds = np.empty(len(y), dtype='i')
        for k in range(n_classes):
            # since the kth column of allocation stores the number of samples
            # of class k in each test set, this generates blocks of fold
            # indices corresponding to the allocation for class k.
            folds_for_class = np.arange(self.n_splits).repeat(allocation[:, k])
            if self.shuffle:
                rng.shuffle(folds_for_class)
            test_folds[y_encoded == k] = folds_for_class
        return test_folds

        # from StratifiedKFold

    # from StratifiedKFold
    def _iter_test_masks(self, X, y=None, groups=None):
        test_folds = self._make_test_folds(X, y)
        for i in range(self.n_splits):
            yield test_folds == i

    def get_n_splits(self, X, y, groups=None):
        # from stratifiedKFold
        return self.n_splits

    def split_BaseCrossValidator(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        indices = np.arange(_num_samples(X))
        for test_index in self._iter_test_masks(X, y, groups):
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            yield train_index, test_index

    def split_BaseKFold(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        if self.n_splits > n_samples:
            raise ValueError(
                ("Cannot have number of splits n_splits={0} greater"
                 " than the number of samples: n_samples={1}.")
                .format(self.n_splits, n_samples))

        for train, test in self.split_BaseCrossValidator(X, y, groups):
            yield train, test

    def split_StratifiedKFold(self, X, y, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.

        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """
        y = check_array(y, ensure_2d=False, dtype=None)
        return self.split_BaseKFold(X, y, groups)

    #personalized scaled splitter based stratifiedKFold
    def split(self, X, y=None, groups=None):
        # Here customized by Khalida
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        if self.n_splits > n_samples:
            raise ValueError(
                ("Cannot have number of splits n_splits={0} greater"
                 " than the number of samples: n_samples={1}.")
                    .format(self.n_splits, n_samples))
        #supe.split to get the indexes for train and test
        for train, test in self.split_StratifiedKFold(X, y, groups): #split1
            # Scaler for each Fold separately to avoid data leakage
            print("I am splitting here using personalized splitter")
            x_train_scaled, x_test_scaled = self.scale_data(X[train], X[test])
            yield x_train_scaled, x_test_scaled

    def make_scaled_cv(self, X, y):
        #crv = self.get_cv()
        #crv.get_n_splits(X, y)
        X = X.reshape(np.shape(X)[0], -1)
        y = LabelBinarizer(neg_label=0, pos_label=1).fit_transform(y)  # 0='10' 1='12'
        X_train_sc_all = []
        X_test_sc_all = []
        for train_index, test_index in self.split(X, y):
            # Scaler for each Fold separately to avoid data leakage
            print("I am splitting here using personalized splitter")
            # Check Instances for learning and Testing
            print("TRAIN:", train_index, "TEST:", test_index)
            x_train_sc, x_test_sc = self.scale_data(X[train_index], X[test_index])
            y_train, y_test = y[train_index], y[test_index]
            X_train_sc_all.append(np.array(x_train_sc))
            X_test_sc_all.append(np.array(x_test_sc))
            # print class-distribution within each Fold
            unique_tr, counts_tr, unique_tst, counts_tst = self.get_class_distribution(y_train, y_test)
            # print(f'Class distribution for train is Class {unique_tr[0]} with {counts_tr[0]} and Class {unique_tr[1]} with {counts_tr[1]}')
            print(f'Class distribution for TRAIN: class 10 with {counts_tr[0]} and class 12 with {counts_tr[1]} instances')
            print(f'Class distribution for TEST is class 10 with {counts_tst[0]} and class 12 with {counts_tst[1]} instances')
        return X_train_sc_all, X_test_sc_all

    def set_scaler(self, scl):
        """
        :'StandardScaler' removes the mean and scales the data to unit variance.-->cannot guarantee balanced feature scales in the presence of outliers.
        :'rescales' the data set such that all feature values are in the range [0, 1]--> very sensitive to the presence of outliers.
        :'MaxAbsScaler' scales and translates each feature individually such that the maximal absolute value of each feature in the training set will be 1.0. It does not shift/center the data,
                        and thus does not destroy any sparsity/  --> suffers from the presence of large outliers.
        :'RobustScaler' is based on percentiles and are therefore not influenced by a few number of very large marginal outliers. Consequently, the resulting range of the transformed feature values is
                        larger than for the previous scalers.
        : 'PowerTransformer': make the data more Gaussian-like in order to stabilize variance and minimize skewness. Currently the Yeo-Johnson and Box-Cox transforms are supported and the optimal
                        scaling factor is determined via maximum likelihood estimation in both methods.
                        :Box-Cox: can only be applied to strictly positive data
                        :Yeo-Johnson: prefered if negative values are present.
        : 'QuantileTransformer': applies a non-linear transformation such that the probability density function of each feature will be mapped to a uniform or Gaussian distribution.
                        In this case, all the data, including outliers, will be mapped to a uniform distribution with the range [0, 1], making outliers indistinguishable from inliers.--> robust to outliers
        : 'Normalizer' rescales the vector for each sample to have unit norm, independently of the distribution of the samples.
        """

        switcher ={
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'MaxAbsScaler': MaxAbsScaler(),
            'RobustScaler': RobustScaler(quantile_range=(25, 75)),
            'PowerTransformer': PowerTransformer(method='yeo-johnson'), #if negative values
            #'PowerTransformer': PowerTransformer(method='box-cox'), #strictly positive data
            #'QuantileTransformer': QuantileTransformer(output_distribution='uniform'),
            'QuantileTransformer': QuantileTransformer(output_distribution='normal'), #gaussian
            'Normalizer': Normalizer(),
        }
        scale_r = switcher.get(scl, lambda: 'Invalid')
        return scale_r

    def scale_data (self, xtr, xtst):
        """ if you normalize or standardize your entire dataset, then estimate the performance of your model using cross validation, you have committed the sin of data leakage.
            The data rescaling process that you performed had knowledge of the full distribution of data in the training dataset when calculating the scaling factors
            (like min and max or mean and standard deviation). This knowledge was stamped into the rescaled values and exploited by all algorithms in your cross validation test harness.
            A non-leaky evaluation of machine learning algorithms in this situation would calculate the parameters for rescaling data within each fold of the cross validation
            and use those parameters to prepare the data on the held out test fold on each cycle.
            --Doing Data Science: Straight Talk from the Frontline"""
        scaler = self.set_scaler(self.type_scaler)
        xtr_scaled = scaler.fit_transform(xtr.reshape(-1, 1))
        xtst_scaled = scaler.fit_transform(xtst.reshape(-1, 1))
        return xtr_scaled, xtst_scaled

    def plot_roc_cv(self, clf, name_clf, X, y):
        id_subject = self.id_subject
        crv = self.get_cv()
        cvm = self.type_crossv
        """plot for all Cross validation Receiver operating characteristic (ROC) """
        # https://scikit-learn.org/stable
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        fig, ax = plt.subplots()
        for i, (train, test) in enumerate(crv.split(X, y)):
            clf.fit(X[train], y[train])
            viz = metrics.plot_roc_curve(clf, X[test], y[test],
                                 name='ROC fold {}'.format(i),
                                 alpha=0.3, lw=1, ax=ax)
            yp = clf.predict(X[test])
            print(classification_report(y[test], yp))
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title="Receiver operating characteristic V"+str(self.acquisition_version)+" for S"+str(id_subject)+" using "+str(name_clf)+" with: "+str(cvm))
        ax.legend(loc="lower right")
        plt.show()
        return

    def get_cv(self, **kwargs): #get_train_testsamples
        # """ TimeSeriesSplit: This cross-validation object is a variation of :class:`KFold`.
        # In the kth split, it returns first k folds as train set and the
        # (k+1)th fold as test set. Note that unlike standard cross-validation methods, successive
        # training sets are supersets of those that come before them. This kind of data split is very important in the case of our EEG data"""
        # tcv = self.type_crossv
        # kwargs.get("cv_splits", CV_SPLITS)
        # kwargs.get("random_state", RANDOM_STATE)
        # kwargs.get("test_size", TEST_SIZE)
        # switcher = {
        #     'StratifiedShuffleSplit': StratifiedShuffleSplit(CV_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE),
        #     'KFold': KFold(n_splits=CV_SPLITS, random_state=None),
        #     'StratifiedKFold': StratifiedKFold(n_splits=CV_SPLITS, random_state=None),
        #     'TimeSeriesSplit': TimeSeriesSplit(max_train_size=None, n_splits=CV_SPLITS),
        #     'GroupKFold': GroupKFold(n_splits=CV_SPLITS),#not very stable
        #     'RepeatedStratifiedKFold': RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=10, random_state=RANDOM_STATE),
        # }
        # crosv = switcher.get(tcv, lambda: 'Invalid')
        return #crosv

    def get_class_distribution(self, y_train, y_test):
        """Compute class distribution for train and test"""
        #y_train =self.make_cross_validation(X, y)[2] #TODO later we should be able to use this function outside make_crossValidation()
        #y_test = self.make_cross_validation(X,y)[3]
        n_classes_train = len(np.unique(y_train))
        n_classes_test = len(np.unique(y_test))
        unique_tr, counts_tr = np.unique(y_train, return_counts=True)
        unique_tst, counts_tst = np.unique(y_test, return_counts=True)
        #frequencies_tr = np.asarray((unique_tr, (counts_tr/n_classes_train))).T
        #TODO finish class distribution we can compute proportion
        return unique_tr, counts_tr, unique_tst, counts_tst

    def get_train_test_samples(self, X, y):
        strategy = self.sampling_strategy
        if strategy == 'cross_validation':
            self.make_cross_validation(X, y)
            return
        elif strategy == 'train_test_split':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False)
            #TODO add validation sample (10%) #Customize the splitter to be scaled.
        else:
            #log.infos("Sampling strategy not allowed, try with cross_validation/ train_test_split")
            print("Sampling strategy not allowed, try with cross_validation/ train_test_split")
        return X_train, X_test, y_train, y_test

    def vis_cv_behavior(self, X, y):
        # #visualize
        # cvm = self.type_crossv()
        # crsv = self.get_cv()
        # id_subject = self.id_subject
        # fig, ax = plt.subplots(figsize=(10, 5))
        # for ii, (tr, tt) in enumerate(crsv.split(X, y)):
        #     # Plot training and test indices
        #     l1 = ax.scatter(tr, [ii] * len(tr), c=[cm.coolwarm(.1)], marker='_', lw = 6)
        #     l2 = ax.scatter(tt, [ii] * len(tt), c=[cm.coolwarm(.9)], marker='_', lw = 6)
        #     ax.set(ylim=[10, -1], title=str(cvm)+' behavior for S'+str(self.id_subject)+'V'+str(self.acquisition_version), xlabel='data index', ylabel='CV iteration')
        #     ax.legend([l1, l2], ['Training', 'Validation'])
        return

if __name__== '__main__':
    acquisition_version = '0.10'
    type_crossv = 'StratifiedKFold'  #Train_test_split #RepeatedStratifiedKFold
    strategy_sampling = 'cross_validation'
    strategy_scaler = 'MinMaxScaler'
    n_splits = 2
    clf = LinearDiscriminantAnalysis()
    # sc1 = ScaledCvSplitter(31, strategy_sampling, strategy_scaler, type_crossv, acquisition_version)
    sc1 = ScaledStratifiedKFold(31, strategy_sampling, n_splits, strategy_scaler, acquisition_version)
    X, y = get_xy(sc1.id_subject, target_events=[FREQUENCY_TO_TRIGGER[10], FREQUENCY_TO_TRIGGER[12]], version= sc1.acquisition_version)  # triggers_frequency=[10, 12]
    #cv = sc1.get_cv()
    #Previous strategy: Scale +StratifiedKFold split:
    X1 = X.reshape(np.shape(X)[0], -1)
    y = LabelBinarizer(neg_label=0, pos_label=1).fit_transform(y)
    scaler = sc1.set_scaler(sc1.type_scaler)
    scaled_x = scaler.fit_transform(X1) #scale before CV (wrong)
    x_train_all = []
    x_test_all = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=None)
    perf_list = []
    curr_fold = 0
    for train_index, test_index in skf.split(scaled_x, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        x_train_all.append(np.array(X_train))
        x_test_all.append(np.array(X_test))
    #     clf.fit(X_train, y_train)
    #     y_predicted = clf.predict(X_test)
    #     # Evaluate our models
    #     performance_fold = f1_score(y_test, y_predicted, average='micro')  # TODO use pipeline to test all measure for comparison
    #     # Scale and test each train and apply classifier separately
    #     perf_list.append(performance_fold)
    #     print("accuracy_fold_%s" % curr_fold, performance_fold)  # TODO use log later
    #     curr_fold += 1
    # average_perf = np.average(perf_list)
    # print("average accuracy without scale", average_perf)


    #New strartegy: StratifiedKFold split with scale per Fold
    #train_sc_all, test_sc_all = sc1.make_scaled_cv(X, y)
    train_cv, test_cv = sc1.split(X1, y)


#TODO tester sur les différentes versions et classifieurs, comparer les deux méthodes