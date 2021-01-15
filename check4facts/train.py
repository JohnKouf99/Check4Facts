import os
import time

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from joblib import dump

from check4facts.config import DirConf


class Trainer:

    def __init__(self, **kwargs):
        self.classifiers_params = kwargs['classifiers']
        self.gs_params = kwargs['gs']
        self.features = kwargs['features']
        self.best_model = None

    def save_best_model(self, path):
        dump(self.best_model['best_estimator'], path)

    def gs(self, X, y):
        gs_results = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
        for clf in self.classifiers_params:
            gs = GridSearchCV(
                estimator=globals()[clf['class']](),
                param_grid=clf['params'],
                scoring=self.gs_params['scoring'],
                refit=self.gs_params['refit'],
                cv=skf).fit(X, y)
            gs_results.append({
                **{'clf': clf['name'], 'best_estimator': gs.best_estimator_,
                   'best_params': gs.best_params_},
                **{scorer: gs.cv_results_[f'mean_test_{scorer}'][
                    gs.best_index_] for scorer in self.gs_params['scoring']}})
        return pd.DataFrame(gs_results)

    def run(self, X, y):
        gs_results_df = self.gs(X, y).sort_values(
            by=self.gs_params['refit'], ascending=False).reset_index(drop=True)
        print(gs_results_df)
        self.best_model = gs_results_df.iloc[0]
        return

    def run_dev(self):
        start_time = time.time()
        if not os.path.exists(DirConf.TRAINER_RESULTS_DIR):
            os.mkdir(DirConf.TRAINER_RESULTS_DIR)
        statement_df = pd.read_csv(DirConf.CSV_FILE)
        features_df = pd.DataFrame([pd.read_json(os.path.join(
            DirConf.FEATURES_RESULTS_DIR, f'{s_id}.json'), typ='series')
            for s_id in statement_df['Fact id']], columns=self.features)
        mask = statement_df['Verdict'] == 'UNKNOWN'
        X = np.vstack(features_df[~mask].apply(np.hstack, axis=1))
        y = statement_df['Verdict'][~mask].astype(int)
        self.run(X, y)
        fname = self.best_model['clf'] + '_' + time.strftime(
            '%Y-%m-%d-%H:%M') + '.joblib'
        path = os.path.join(DirConf.TRAINER_RESULTS_DIR, fname)
        self.save_best_model(path)
        stop_time = time.time()
        print(f'Model training done in {stop_time-start_time:.2f} secs.')
