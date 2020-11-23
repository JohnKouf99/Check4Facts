import os
import glob
import string
import datetime
import time

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

from check4facts.config import DirConf


class Harvester:

    def __init__(self, **kwargs):
        self.basic_params = kwargs['basic']
        self.html_params = kwargs['html']
        self.similarity_params = kwargs['similarity']
        if self.similarity_params['metric'] == 'emb':
            self.nlp = spacy.load('el_core_news_lg')

    @staticmethod
    def text_preprocess(text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.lower()
        text = ' '.join([word for word in text.split() if
                         word not in stopwords.words('greek')])
        return text

    def most_similar(self, texts, text):
        all_texts = list(map(self.text_preprocess, texts + [text]))
        if self.similarity_params['metric'] == 'bow':
            vec = CountVectorizer().fit_transform(all_texts)
            idx = np.argmax(cosine_similarity(vec.toarray())[-1][:-1])
        else:
            docs = [self.nlp(t[:self.nlp.max_length]) for t in all_texts]
            idx = np.argmax([docs[-1].similarity(d) for d in docs[:-1]])
        return texts[idx]

    @staticmethod
    def save_content_to_xml(content, f_name):
        out = os.path.join(DirConf.HARVEST_XML_DIR, f_name)
        with open(out, 'w') as out_f:
            out_f.write(content)
        return

    def harvest_article(self, c_id, c_text, a_idx, a_url, save=False):
        try:
            response = requests.get(a_url, timeout=self.html_params['timeout'])
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            if save: self.save_content_to_xml(
                f'{type(e)} :: {e}', f'{c_id}_{a_idx}.xml')
            result = {'title': None, 'body': None,
                      'sim_paragraph': None, 'sim_sentence': None}
            return result

        soup = BeautifulSoup(response.content, self.html_params['parser'])
        if save: self.save_content_to_xml(
            soup.prettify(), f'{c_id}_{a_idx}.xml')
        for element in soup(self.html_params['blacklist']):
            element.extract()

        title = soup.title.text.strip() if soup.title else None
        all_lines = []
        if soup.body:
            for child in soup.body.children:
                if isinstance(child, Tag):
                    lines = [line.strip()
                             for line in child.get_text().splitlines()
                             if len(line.split()) > 10]
                    all_lines += lines
        body = '\n'.join(all_lines) if all_lines else None
        if body:
            paragraphs = body.splitlines()
            sim_par = self.most_similar(paragraphs, c_text)
            # TODO Replace below with sent_tokenize on each paragraph
            #  instead of joining paragraphs first. In order to decide,
            #  check the quality of results
            sentences = sent_tokenize(' '.join(paragraphs))
            sim_sent = self.most_similar(sentences, c_text)
        else:
            sim_par, sim_sent = None, None
        result = {'title': title, 'body': body,
                  'sim_paragraph': sim_par, 'sim_sentence': sim_sent}
        return result

    @staticmethod
    def filter_article_format(df, format_):
        if 'fileFormat' not in df: df['fileFormat'] = 'html'
        df['fileFormat'] = df['fileFormat'].fillna('html')
        df = df[df['fileFormat'] == format_]
        return df

    def harvest_articles(self, d, save=False):
        c_id, c_text, c_articles = d['c_id'], d['c_text'], d['c_articles']
        c_articles = self.filter_article_format(
            c_articles, self.basic_params['article_format'])
        a_idxs = c_articles['index'] if not c_articles.empty else []
        a_urls = c_articles['link'] if not c_articles.empty else []
        now = datetime.datetime.utcnow()
        data = [{**{'index': a_idx, 'url': a_url, 'harvest_date': now,
                    'file_format': self.basic_params['article_format']},
                 **self.harvest_article(c_id, c_text, a_idx, a_url, save)}
                for a_idx, a_url in zip(a_idxs, a_urls)]
        result = pd.DataFrame(data)
        if result.empty: result = result.reset_index()
        return result

    def run(self, claim_dicts, save=False):
        return [self.harvest_articles(d, save) for d in claim_dicts]

    def run_dev(self):
        start_time = time.time()
        if not os.path.exists(DirConf.HARVEST_RESULTS_DIR):
            os.mkdir(DirConf.HARVEST_RESULTS_DIR)
        if not os.path.exists(DirConf.HARVEST_XML_DIR):
            os.mkdir(DirConf.HARVEST_XML_DIR)
        claims_df = pd.read_csv(os.path.join(
            DirConf.DATA_DIR, self.basic_params['filename']))
        path = os.path.join(DirConf.DATA_DIR, self.basic_params['dir_name'])
        csv_files = glob.glob(os.path.join(path, '*.csv'))
        for csv_file in csv_files:
            t0 = time.time()
            c_id = os.path.basename(csv_file).split('.')[0]
            c_text = claims_df.loc[
                claims_df['Fact id'] == int(c_id), 'Text'].iloc[0]
            df = pd.read_csv(csv_file)
            claim_dict = {'c_id': c_id, 'c_text': c_text, 'c_articles': df}
            result = self.run([claim_dict], save=True)[0]
            t1 = time.time()
            print(f'Claim id {c_id}: Harvested {len(result)} articles '
                  f'in {t1-t0:.2f} secs.')
            out = os.path.join(DirConf.HARVEST_RESULTS_DIR, f'{c_id}.csv')
            result.to_csv(out, index=False)
        stop_time = time.time()
        print(f'Harvest done in {stop_time-start_time:.2f} secs.')
