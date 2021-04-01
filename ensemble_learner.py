"""
Created on 18/01/2021
for MAA Project
@author: Khalida Douibi
The idea here is to improve the results of best Personalized Classifier from calibration step by using Homogeneous Ensemble methods as Bagging, Boosting, RF, Extratrees (hierarchical classif)
Stacking was already tested before and does not improve the results for EEG data analysis.
"""
import joblib
import pandas as pd
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier

#from analysis import CV_DIR
from config import RANDOM_STATE
from maa.analysis.eeg.eeg_reader import get_xy, FILTER_LOW_PASS, FILTER_HIGH_PASS
from maa import BASE_DIR, OUTPUT_BASE_DIR, ACQUISITION_VERSION, ACQUISITION_VERSION_0_9, ACQUISITION_VERSION_0_10#, DEFAULT_SSVEP_CHANNELS_TYPES


class EnsembleLearner:

    def __init__(self, id_subject, strategy, base_dir=BASE_DIR, acquisition_version=ACQUISITION_VERSION, filter_range=None, n_learners=2):  # utiliser kwargs après avec le code RB
        self.id_subject = id_subject
        self.base_dir = base_dir
        self.acquisition_version = acquisition_version
        self.clf = []
        self.strategy = strategy
        self.n_learners = n_learners
        if filter_range is None:
            self.filter_range = f'[{FILTER_LOW_PASS}-{FILTER_HIGH_PASS}]'# should change it later
        else:
            self.filter_range = filter_range
        self.best_checkpoint = 0

    def get_best_classifier(self):
        from maa.analysis import get_cv_filename
        """get best classifier with best checkpoint from calibration phase to be enhanced and boosted"""
        #file = get_cv_filename(self.id_subject, directory= f'{CV_DIR}/PC_all',  params={'channels': DEFAULT_SSVEP_CHANNELS_TYPES})

        calib_output = pd.read_csv(f'{self.base_dir}/output/v{self.acquisition_version}/cv/PC_all/'
                                   f'PC_cv_v{self.acquisition_version}_s0{self.id_subject}_[O1,O2,Pz,P3,P4]_filter={self.filter_range}Hz_iir_0.0075_win=[0.0-all].dat',
                                   sep='\t', index_col=False)
        classif_checkpoints = calib_output.groupby('checkpoint').max().reset_index()
        best_results = classif_checkpoints.loc[classif_checkpoints['test_score'].idxmax()]
        best_clf = best_results['method']
        best_checkpoint = best_results['checkpoint']
        self.clf = best_clf
        self.best_checkpoint = best_checkpoint
        return best_clf, best_checkpoint

    def load_base_classifier(self):
        """load best classifier as base classifier for our ensemble method"""
        filename = f'{OUTPUT_BASE_DIR}/v{self.acquisition_version}/models/PC_{self.get_best_classifier()[0]}_v{self.acquisition_version}_s0{self.id_subject}_[O1,O2,Pz,P3,P4]_filter={self.filter_range}Hz_iir_0.0075_win=[0.0' \
                   f'-{self.get_best_classifier()[1]}]_baseline=(-0.4,0)_trainVersions={self.acquisition_version}.joblib.bz2'
        print(filename)
        from maa.analysis import get_model_filename

        # theFilename = get_model_filename(self.id_subject, self.acquisition_version, ext='.joblib.bz2', params={})
        # loadedClsf = joblib.load(theFilename)
        #SORRY RB je reprend à partir de mon code tu peux l'adapter après ;)
        loadedClsf = joblib.load(filename)
        self.clf = loadedClsf
        return loadedClsf

    def learn_ensemble(self, clf, X, y):
        """Learn the best """
        strategy = self.strategy
        switcher_strategy = {
            'bagging': BaggingClassifier(base_estimator=clf, n_estimators=self.n_learners, random_state=RANDOM_STATE).fit(X, y),
            'boosting': AdaBoostClassifier(base_estimator=clf, n_estimators=self.n_learners, random_state=RANDOM_STATE).fit(X, y),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=self.n_learners, max_depth=1, random_state=RANDOM_STATE).fit(X, y),  # TODO to be tuned and personnalized for our classif not DT/ hierarchical classif
            'RandomForestClassifier': RandomForestClassifier(max_depth=2, random_state=RANDOM_STATE).fit(X, y),  # TODO to be tuned and personnalized for our classif not DT
            'ExtraTreesClassifier': ExtraTreesClassifier(n_estimators=self.n_learners, random_state=RANDOM_STATE).fit(X, y)
            # the last ones include OOB test other vote
        }
        clf_ensemble = switcher_strategy.get(strategy, lambda: 'Invalid')

        return clf_ensemble


if __name__ == '__main__':
    """ two strategy for train_test could be tested, 
    Strategy 1: use best classifier from V0.9 and train/ EM and test for the same subject from V0.10: exemple S0.1 from V0.9 & same subject 31 from V0.10
    Strategy 2: use best classifier from V0.9 and train/test EM by using other subject data from V0.10 (interesting for SBC)"""
    # main
    acq_ver = ACQUISITION_VERSION_0_9  # use loop later to have an overall prog
    ens = EnsembleLearner(1, strategy='bagging', acquisition_version=acq_ver, n_learners=50)
    print(ens.get_best_classifier())
    base_clf = ens.load_base_classifier()

    # train/test EM
    acq_V_train_test = ACQUISITION_VERSION_0_10
    id_subject_train_test = 31 #[7, 8, 11, 21, 22]
    #id_subject_validation = [30]  # Demo
    X_train = []
    y_train = []
    # if online prendre partie des data du reste des checkpoints pour créer un ensemble method et best classifier prendre celui du checkpoint 1s ou bien selon threshold(exemple)
    # ici première simulation en offline
    xx, yy = get_xy(id_subject_train_test, version=acq_V_train_test)

    # for subject in id_subject_train_test:
    #     xx, yy = get_xy(subject, version=acq_V_train_test)
    #     X_train.append(xx)
    #     y_train.append(yy)

    # CV train & test
    # Ens1.learn_ensemble(base_clf, X, y)
    # get data for train/test from V0.10
