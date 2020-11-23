import os
import string
import time

import pandas as pd
import googleapiclient.errors
from googleapiclient.discovery import build

from check4facts.config import DirConf


class SearchEngine:

    def __init__(self, **kwargs):
        self.basic_params = kwargs['basic']
        self.api_specific_params = kwargs['api_specific']
        self.standard_query_params = kwargs['standard_query']
        self.service = build(
            'customsearch', 'v1',
            developerKey=self.standard_query_params['api_key'])

    @staticmethod
    def text_preprocess(text):
        # Punctuation removal
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    def google_search(self, search_text):
        params = self.api_specific_params
        params['q'], params['start'] = search_text, 1
        search_results = []
        while params['start'] - 1 < self.basic_params['total_num']:
            try:
                response = self.service.cse().list(**params).execute()
                search_results += response['items']
                next_page = response['queries']['nextPage'][0]
                params['start'] = next_page['startIndex']
            except KeyError:
                break
            except googleapiclient.errors.HttpError as e:
                print(type(e), '::', e)
                time.sleep(60 * 60 * 24)
        result = pd.DataFrame(search_results).reset_index()
        return result

    def run(self, claim_texts):
        return [self.google_search(self.text_preprocess(t))
                for t in claim_texts]

    def run_dev(self):
        start_time = time.time()
        if not os.path.exists(DirConf.SEARCH_RESULTS_DIR):
            os.mkdir(DirConf.SEARCH_RESULTS_DIR)
        path = os.path.join(DirConf.DATA_DIR, self.basic_params['filename'])
        claims_df = pd.read_csv(path).head(10)
        for c_id, c_text in zip(claims_df['Fact id'], claims_df['Text']):
            t0 = time.time()
            result = self.run([c_text])[0]
            t1 = time.time()
            print(f'Claim id {c_id}: Found {len(result)} search results '
                  f'in {t1-t0:.2f} secs.')
            out = os.path.join(DirConf.SEARCH_RESULTS_DIR, f'{c_id}.csv')
            result.to_csv(out, index=False)
        stop_time = time.time()
        print(f'Search done in {stop_time-start_time:.2f} secs.')
