# -*- coding: utf-8 -*-
"""
--Created on Wed 26/10/2020
for MAA Project
@author: Khalida Douibi
The present script aims to Spot-Check Classification Algorithms to discover which one performs better
for Offline SSVEP classification, then, will be applied for Online classification.

-First tests are conducted using get_xy after applying Epochs by defaults...
We tested also several splitter for re-sampling phase, the best one to choose for time series data StratifiedKFold splitter
--01/2021
-New version to change preprocessing pipeline to avoid Data Leakage:
A non-leaky evaluation of machine learning algorithms in this situation would calculate the parameters for rescaling data within each fold of the cross validation and
use those parameters to prepare the data on the held out test fold on each cycle. — Page 313, Doing Data Science: Straight Talk from the Frontline.
-Add a dummy classifier as baseline.
"""
import numpy as np
import matplotlib.pyplot as plt
from boto import sns
from skimage.metrics import mean_squared_error
from sklearn import metrics
from sklearn.metrics import auc, classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, r2_score, mean_absolute_error
from matplotlib import cm
from sklearn.cross_decomposition import CCA, PLSCanonical
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, LabelBinarizer, MinMaxScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer, Normalizer
from sklearn.svm import LinearSVC, SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, train_test_split

#from maa import log
from config import FREQUENCY_TO_TRIGGER
from maa.analysis.eeg.eeg_reader import get_xy, ACQUISITION_VERSION
from maa.config.prepro_config import *
from sklearn.model_selection import LeaveOneOut, StratifiedShuffleSplit, TimeSeriesSplit, KFold, StratifiedKFold, GroupKFold, RepeatedStratifiedKFold, cross_val_score
from mne.decoding import Vectorizer
from collections import OrderedDict
from ml_config import TEST_SIZE, CV_SPLITS, RANDOM_STATE


class EEGOfflineClassification:

    def __init__(self, id_subject, sampling, strat_scaler, type_cv, acquisition):
        self.id_subject = id_subject
        self.sampling_strategy = sampling
        self.type_crossv = type_cv
        self.acquisition_version = acquisition
        self.type_scaler = strat_scaler
        # self.data = data
        # self.pipeline = pipeline

    def plot_roc_cv(self, clf, name_clf, X, y):
        #TODO enelever le splitter d'ici et lier la fct directement à la classe
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

    def classify(self, xtr, xtst, ytr):
        #TODO use Pipeline from ML config
        #pipe = Pipeline(['classifier', LinearDiscriminantAnalysis()])  # juste un exemple après je rajoute la pipeline de classif
        clf = LinearDiscriminantAnalysis()
        clf.fit(xtr, ytr)
        predicted = clf.predict(xtst)
        return predicted

    def get_cv(self, **kwargs): #get_train_testsamples
        """ TimeSeriesSplit: This cross-validation object is a variation of :class:`KFold`.
        In the kth split, it returns first k folds as train set and the
        (k+1)th fold as test set. Note that unlike standard cross-validation methods, successive
        training sets are supersets of those that come before them. This kind of data split is very important in the case of our EEG data"""
        tcv = self.type_crossv
        kwargs.get("cv_splits", CV_SPLITS)
        kwargs.get("random_state", RANDOM_STATE)
        kwargs.get("test_size", TEST_SIZE)
        switcher = {
            'StratifiedShuffleSplit': StratifiedShuffleSplit(CV_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE),
            'KFold': KFold(n_splits=CV_SPLITS, random_state=None),
            'StratifiedKFold': StratifiedKFold(n_splits=CV_SPLITS, random_state=None),
            'TimeSeriesSplit': TimeSeriesSplit(max_train_size=None, n_splits=CV_SPLITS),
            'GroupKFold': GroupKFold(n_splits=CV_SPLITS),#not very stable
            'RepeatedStratifiedKFold': RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=10, random_state=RANDOM_STATE),
        }
        crosv = switcher.get(tcv, lambda: 'Invalid')
        return crosv

    def make_classification(self, X, y):
        crv = self.get_cv()
        #crv.get_n_splits(X, y)
        X = X.reshape(np.shape(X)[0], -1)
        # scaler = StandardScaler().fit(X) #scale before CV (wrong)
        # X = scaler.transform(X)
        # # Binarization, important step for some algorithms based probabilities
        y = LabelBinarizer(neg_label=0, pos_label=1).fit_transform(y)  # 0='10' 1='12'
        perf_list = []
        curr_fold = 0
        for train_index, test_index in crv.split(X, y):
            # Check Instances for learning and Testing
            print("TRAIN:", train_index, "TEST:", test_index)
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # print class-distribution within each Fold
            unique_tr, counts_tr, unique_tst, counts_tst = self.get_class_distribution(y_train, y_test)
            # print(f'Class distribution for train is Class {unique_tr[0]} with {counts_tr[0]} and Class {unique_tr[1]} with {counts_tr[1]}')
            print(f'Class distribution for TRAIN: class 10 with {counts_tr[0]} and class 12 with {counts_tr[1]} instances')
            print(f'Class distribution for TEST is class 10 with {counts_tst[0]} and class 12 with {counts_tst[1]} instances')
            # TODO customize pipeline (__init__, fit, transform)
            # pipeline = Pipeline(steps=[
            #     ('scale', self.scale_data(x_train, x_test)),
            #     ('classify', self.classify(x_train_scaled, x_test_scaled, y_train)),
            #     ('Evaluate', self.evaluate(y_test, y_predicted, 'f1-measure'))
            # ])
            # pipeline.fit(x_train, y_train)
            # pipeline.predict(x_test)
            #Scaler for each Fold separately to avoid data leakage
            x_train_scaled, x_test_scaled = self.scale_data(x_train, x_test)
            y_predicted = self.classify(x_train_scaled, x_test_scaled, y_train)
            # Evaluate our models
            performance_fold = self.evaluate(y_test, y_predicted, 'f1-measure') #TODO use pipeline to test all measure for comparison
            #Scale and test each train and apply classifier separately
            perf_list.append(performance_fold)
            print("accuracy_fold_%s" % curr_fold, performance_fold)# TODO use log later
            curr_fold += 1
        average_perf = np.average(perf_list)
        print("average accuracy", average_perf)
        return average_perf

    def get_class_distribution(self, y_train, y_test):
        """Compute class distribution for train and test"""
        #y_train =self.make_cross_validation(X, y)[2] #TODO later we should be able to use this function outside make_crossValidation()
        #y_test = self.make_cross_validation(X,y)[3]

        n_classes_train = len(np.unique(y_train))
        n_classes_test = len(np.unique(y_test))

        unique_tr, counts_tr = np.unique(y_train, return_counts=True)
        unique_tst, counts_tst = np.unique(y_test, return_counts=True)
        #frequencies_tr = np.asarray((unique_tr, (counts_tr/n_classes_train))).T
        #TODO finish class distribution for train and test
        return unique_tr, counts_tr, unique_tst, counts_tst

    def get_train_test_samples(self, X, y):
        strategy = self.sampling_strategy
        if strategy == 'cross_validation':
            self.make_cross_validation(X, y)
            return
        elif strategy == 'train_test_split':
            X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False)
            #TODO add validation sample (10%)
        else:
            #log.infos("Sampling strategy not allowed, try with cross_validation/ train_test_split")
            print("Sampling strategy not allowed, try with cross_validation/ train_test_split")
        return X_train, X_test, y_train, y_test

    def vis_cv_behavior(self, X, y):
        #visualize
        cvm = self.type_crossv()
        crsv = self.get_cv()
        id_subject = self.id_subject
        fig, ax = plt.subplots(figsize=(10, 5))
        for ii, (tr, tt) in enumerate(crsv.split(X, y)):
            # Plot training and test indices
            l1 = ax.scatter(tr, [ii] * len(tr), c=[cm.coolwarm(.1)], marker='_', lw = 6)
            l2 = ax.scatter(tt, [ii] * len(tt), c=[cm.coolwarm(.9)], marker='_', lw = 6)
            ax.set(ylim=[10, -1], title=str(cvm)+' behavior for S'+str(self.id_subject)+'V'+str(self.acquisition_version), xlabel='data index', ylabel='CV iteration')
            ax.legend([l1, l2], ['Training', 'Validation'])
        return

    def classif_report(self, clf, x_train, y_train, x_test, y_test):
        clf.fit(x_train, y_train)
        yp = clf.predict(x_test)
        print(classification_report(y_test, yp))
        return

    def evaluate(self, y_test, ypred, evaluator):
        """ :R2 (R squared): It shows if the model is a good fit observed values and how good of a “fit” it is.
                    High R² means that the correlation between observed and predicted values is high. High value means the variance in the model is similar to the variance in the true values
                    and if the R2 value is low it means that the two values are not much correlated. https://joydeep31415.medium.com/common-metrics-for-time-series-analysis-f3ca4b29fe42
            : MAE Mean absolute error is the average of the absolute values of the deviation. Its is quite robust to ourliers. Hence looking at MAE is useful if the training data is corrupted
                    with outliers and there are huge positive/negative values in our data
            :MSE: The mean squared error is the average of the square of the forecast error. As the square of the errors are taken, the effect is that larger errors have more weight on the score.
            """
        switcher = {
            #general evaluation measures
            'precision': precision_score(y_test, ypred),
            'recall': recall_score(y_test, ypred),
            'accuracy': accuracy_score(y_test, ypred),
            'f1-measure': f1_score(y_test, ypred),
            'confusion_matrix': confusion_matrix(y_test, ypred),
            'classification_report': classification_report(y_test, ypred),
            # metrics to measure the quality of the predictions
            'R2': r2_score(y_test, ypred),
            'MAE': mean_absolute_error(y_test, ypred),
            #'MSE': mean_squared_error(y_test, ypred),
        }
        eval = switcher.get(evaluator, lambda: 'Invalid')
        # Todo Personalized plot
        # sns.heatmap(eval, cmap='PuBu', annot=True, fmt='g', annot_kws={'size': 20})
        # plt.xlabel('predicted', fontsize=18)
        # plt.ylabel('actual', fontsize=18)
        # plt.title(str(evaluator), fontsize=18)
        # plt.show()
        return eval #print('results from:'+str(evaluator)+eval)

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
        xtr_scaled = scaler.fit_transform(xtr)
        xtst_scaled = scaler.fit_transform(xtst)
        return xtr_scaled, xtst_scaled

    def train_pipeline(self):
        # =============================================================================
        #                          Model selection/ Sampling methods
        #                   Choosing the right cross-validation object
        #                   is a crucial part of fitting a model properly
        # =============================================================================
        # =============================================================================
        #                       # Learning & Test
        # =============================================================================
        # TODO create a Pipeline to test all models and run it on the server by saving results using json
        ##Models
        # Linear Models
        # logistic = LogisticRegression(max_iter=1000, multi_class='ovr')#tester multi_class=”multinomial”
        # LDA = LinearDiscriminantAnalysis(solver='svd')
        # RLDA = QuadraticDiscriminantAnalysis()
        # lsvm = LinearSVC(max_iter=10000)
        # NonLinear Models
        # svm = SVC()
        ##TODO Add imbalanced classifier
        # Decomposition methods
        # cca = CCA(n_components=8, max_iter=1000, copy=False)
        # pls = PLSCanonical(n_components=8, max_iter=1000, copy=False)
        # =============================================================================
        #                       # Visualization of Evaluation Performance
        #               define model evaluation method/ tests of several strategies
        # =============================================================================
        # TODO add the pipeline
        # model = LDA
        # # off1.plot_roc_cv(model, 'LDA', cv, off1.type_crossv, X, y)
        # # off1.classif_report(model, off1.X_train, off1.y_train, off1.X_test, off1.y_test)
        # # print(cross_val_score(model, X, y, cv=10))
        # self.plot_roc_cv(model, 'LDA', cv, off1.type_crossv, X, y)
        # self.classif_report(model, self.X_train, self.y_train, self.X_test, self.y_test)
        # print(cross_val_score(model, X, y, cv=10))
        # for m in PIPELINES:
        #     ###-Plain
        #     #PIPELINES['logistic'] = make_pipeline(vectorizer, standard_scaler, logistic)
        #     PIPELINES['LDA'] = make_pipeline(vectorizer, standard_scaler, LDA)
        #     #PIPELINES['RLDA'] = make_pipeline(vectorizer, standard_scaler, RLDA)
        #     #PIPELINES['lsvm'] = make_pipeline(vectorizer, standard_scaler, lsvm)
        #     #PIPELINES['svm'] = make_pipeline(vectorizer, standard_scaler, svm)
        #     #PIPELINES['cca'] = make_pipeline(vectorizer, standard_scaler, cca)
        #     #PIPELINES['pls'] = make_pipeline(vectorizer, standard_scaler, pls)
        #     #res = cross_val_score(LDA, X, y, scoring=SCORING_ROC, cv=cv, n_jobs=-1)
        #     # summarize result
        #     #print('roc_auc: %.3f (%.3f)' % (mean(res), std(res)))
        #
        #     plot_rocCV(PIPELINES[m], cv, X, y)
        return



if __name__== '__main__':

    acquisition_version = '0.10'
    type_crossv = 'StratifiedKFold'  #Train_test_split #RepeatedStratifiedKFold
    strategy_sampling = 'cross_validation'
    strategy_scaler = 'StandardScaler'
    off1 = EEGOfflineClassification(31, strategy_sampling, strategy_scaler, type_crossv, acquisition_version)
    X, y = get_xy(off1.id_subject, target_events=[FREQUENCY_TO_TRIGGER[10], FREQUENCY_TO_TRIGGER[12]], version=off1.acquisition_version)  # triggers_frequency=[10, 12]
    cv = off1.get_cv()
    avr_accuracy = off1.make_classification(X, y)

    #off1.vis_cv_behavior(cv, off1.type_crossv, X, y)
    # off1.plot_roc_cv(model, 'LDA', cv, off1.type_crossv, X, y)
    # off1.classif_report(model, off1.X_train, off1.y_train, off1.X_test, off1.y_test)
    # print(cross_val_score(model, X, y, cv=10))
