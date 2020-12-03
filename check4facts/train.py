import os
import time

import numpy as np
import pandas as pd

import check4facts.models as models
from check4facts.config import DirConf


class Trainer:

    def __init__(self, **kwargs):
        self.model_params = kwargs['model']
        self.metrics_params = kwargs['metrics']
        self.save_params = kwargs['save']
        self.features = kwargs['features']
        self.model = self.get_model()

    def get_model(self):
        model_class = getattr(models, self.model_params['name'])
        model = model_class(**self.model_params['params'])
        return model

    def save_model(self, path):
        self.model.save(path)

    def run(self, x, y):
        return self.model.fit(x, y)

    def run_dev(self):
        start_time = time.time()
        if not os.path.exists(DirConf.TRAINER_RESULTS_DIR):
            os.mkdir(DirConf.TRAINER_RESULTS_DIR)
        statement_df = pd.read_csv(DirConf.CSV_FILE)
        features_df = pd.DataFrame([pd.read_json(os.path.join(
            DirConf.FEATURES_RESULTS_DIR, f'{s_id}.json'), typ='series')
            for s_id in statement_df['Fact id']], columns=self.features)
        mask = (features_df.isna().any(axis=1)) | (
                statement_df['Verdict'] == 'UNKNOWN')
        x = np.vstack(features_df[~mask].apply(np.hstack, axis=1))
        # TODO investigate why (eg check null values for s_id=1 in
        #  artciles.body.emotion.anger). For now just set nones to 0.0
        # x[x == None] = 0.0
        x = np.nan_to_num(x)
        y = statement_df['Verdict'][~mask]
        self.run(x, y)
        fname = self.save_params['prefix'] + time.strftime(
            self.save_params['datetime']) + self.save_params['suffix']
        path = os.path.join(DirConf.TRAINER_RESULTS_DIR, fname)
        self.save_model(path)
        stop_time = time.time()
        print(f'Model training done in {stop_time-start_time:.2f} secs.')
