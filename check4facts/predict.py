import os
import time

import numpy as np
import pandas as pd

import check4facts.models as models
from check4facts.config import DirConf


class Predictor:

    def __init__(self, **kwargs):
        self.model_params = kwargs['model']
        self.features = kwargs['features']
        self.model = self.get_model()

    def get_model(self):
        model_class = getattr(models, self.model_params['name'])
        model = model_class().load(self.model_params['path'])
        return model

    def prepare_data(self, features_list):
        features_df = pd.DataFrame(features_list, columns=self.features)
        mask = features_df.isna().any(axis=1)
        x = np.vstack(features_df[~mask].apply(np.hstack, axis=1))
        # TODO investigate why (eg check null values for s_id=1 in
        #  articles.body.emotion.anger). For now just set nones to 0.0
        x[x == None] = 0.0
        return x

    def run(self, features_list):
        x = self.prepare_data(features_list)
        # TODO we lost here the alignment after removing statements with no
        #  resources. Check why there is a 'TRUE' pred label in dev results.
        return self.model.predict_proba(x)

    def run_dev(self):
        start_time = time.time()
        if not os.path.exists(DirConf.PREDICTOR_RESULTS_DIR):
            os.mkdir(DirConf.PREDICTOR_RESULTS_DIR)
        statement_df = pd.read_csv(DirConf.CSV_FILE)
        features_list = [pd.read_json(os.path.join(
            DirConf.FEATURES_RESULTS_DIR, f'{s_id}.json'), typ='series')
            for s_id in statement_df['Fact id']]
        result = self.run(features_list)
        path = os.path.join(DirConf.PREDICTOR_RESULTS_DIR, 'results.csv')
        pd.DataFrame(result).to_csv(path, index=False)
        stop_time = time.time()
        print(f'Model prediction done in {stop_time-start_time:.2f} secs.')
