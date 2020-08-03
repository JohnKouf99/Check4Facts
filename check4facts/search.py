import os
import string

import pandas as pd
from googleapiclient.discovery import build

from check4facts.config import DirConf


class CustomSearchEngine:

    def __init__(self, **kwargs):
        self.basic_params = kwargs['basic']
        self.api_specific_params = kwargs['api_specific']
        self.standard_query_params = kwargs['standard_query']

        self.statements_df_ = None
        self.service_ = None

    @property
    def statements_df(self):
        if self.statements_df_ is None:
            path = os.path.join(
                DirConf.DATA_DIR, self.basic_params['filename'])
            self.statements_df_ = pd.read_csv(path)
            self.statements_df_['Text'] = \
                self.statements_df_['Text'].map(self.preprocess)
        return self.statements_df_

    @property
    def service(self):
        if self.service_ is None:
            self.service_ = build(
                'customsearch', 'v1',
                developerKey=self.standard_query_params['api_key'])
        return self.service_

    @staticmethod
    def preprocess(text):
        # Punctuation removal
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        return text

    def search(self, text):
        params = self.api_specific_params
        params['q'] = text

        results, start = [], 1
        while start - 1 < self.basic_params['total_num']:
            try:
                params['start'] = start
                # there is also extra metadata available outside ['items'] key
                results += self.service.cse().list(**params).execute()['items']
                start += self.api_specific_params['num']
            except KeyError:
                break
        return pd.DataFrame(results)

    def run(self):
        if not os.path.exists(DirConf.SEARCH_RESULTS_DIR):
            os.mkdir(DirConf.SEARCH_RESULTS_DIR)

        for row in self.statements_df.head(20).itertuples():
            statement_id, statement_text = row[1], row[2]
            print('Search text:', statement_text)

            results_df = self.search(statement_text)
            print('Search results:', len(results_df))

            results_df['statement_id'] = statement_id
            results_df['statement_text'] = statement_text

            out = os.path.join(
                DirConf.SEARCH_RESULTS_DIR, '{}.csv'.format(statement_id))
            results_df.to_csv(out, index=False)
        return
