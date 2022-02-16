import os
import time

import numpy as np
import pandas as pd
from joblib import load

from check4facts.config import DirConf


def proba_to_rating(true_proba):
    if true_proba < 0.25:
        rating = 1
    elif true_proba < 0.5:
        rating = 2
    elif true_proba < 0.75:
        rating = 3
    else:
        rating = 4
    return rating


class Predictor:

    def __init__(self, **kwargs):
        self.model_params = kwargs['model']
        self.features = kwargs['features']
        self.model = load(self.model_params['path'])

    def prepare_data(self, features_list):
        features_df = pd.DataFrame(features_list, columns=self.features)
        # Drop statements with no resources
        # idx = list(features_df.dropna().index)
        # features_df = features_df[features_df.index.isin(idx)]
        x = np.vstack(features_df.apply(np.hstack, axis=1)).astype('float')
        return features_df.index, x

    def run(self, features_list):
        idx, x = self.prepare_data(features_list)
        preds = self.model.predict_proba(x)
        ratings = [proba_to_rating(p) for p in preds[:, 1]]
        # 'Statement id' col contains the 0-based index of the passed
        # statements for prediction. This is NOT equivalent to the
        # 'Fact id' property.
        result_df = pd.DataFrame(
            {'Statement id': idx, 'pred_0': preds[:, 0],
             'pred_1': preds[:, 1], 'rating': ratings})
        return result_df

    def run_dev(self):
        start_time = time.time()
        if not os.path.exists(DirConf.PREDICTOR_RESULTS_DIR):
            os.mkdir(DirConf.PREDICTOR_RESULTS_DIR)
        statement_df = pd.read_csv(DirConf.CSV_FILE).head(60)
        features_list = [pd.read_json(os.path.join(
            DirConf.FEATURES_RESULTS_DIR, f'{s_id}.json'), typ='series')
            for s_id in statement_df['Fact id']]
        result_df = self.run(features_list)
        path = os.path.join(DirConf.PREDICTOR_RESULTS_DIR, 'clf_results.csv')
        result_df.to_csv(path, index=False)
        stop_time = time.time()
        print(f'Model prediction done in {stop_time-start_time:.2f} secs.')
