import os
import glob
import string

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
            response = requests.get(a_url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            if save: self.save_content_to_xml(
                f'{type(e)} :: {e}', f'{c_id}_{a_idx}.xml')
            return None, None, None, None

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
            sentences = sent_tokenize(' '.join(paragraphs))
            sim_sent = self.most_similar(sentences, c_text)
        else:
            sim_par, sim_sent = None, None
        return title, body, sim_par, sim_sent

    def harvest_articles(self, d, save=False):
        data = []
        if not d['articles'].empty:
            # TODO use all articles per claim
            a_idxs, a_urls = d['articles']['index'].head(10), d['articles']['link'].head(10)
            for a_idx, a_url in zip(a_idxs, a_urls):
                title, body, sim_par, sim_sent = self.harvest_article(
                    d['c_id'], d['c_text'], a_idx, a_url, save)
                data.append({
                    'index': a_idx, 'url': a_url, 'title': title,
                    'body': body, 'sim_par': sim_par, 'sim_sent': sim_sent})
        result = pd.DataFrame(data)
        if result.empty: result = result.reset_index()
        return result

    def run(self, articles, save=False):
        return [self.harvest_articles(d, save) for d in articles]

    def run_dev(self):
        if not os.path.exists(DirConf.HARVEST_RESULTS_DIR):
            os.mkdir(DirConf.HARVEST_RESULTS_DIR)
        if not os.path.exists(DirConf.HARVEST_XML_DIR):
            os.mkdir(DirConf.HARVEST_XML_DIR)
        claims_df = pd.read_csv(os.path.join(
            DirConf.DATA_DIR, self.basic_params['filename']))
        path = os.path.join(DirConf.DATA_DIR, self.basic_params['dir_name'])
        csv_files = glob.glob(os.path.join(path, '*.csv'))
        for csv_file in csv_files:
            c_id = os.path.basename(csv_file).split('.')[0]
            c_text = claims_df.loc[
                claims_df['Fact id'] == int(c_id), 'Text'].iloc[0]
            df = pd.read_csv(csv_file)
            if 'fileFormat' not in df: df['fileFormat'] = 'html'
            df['fileFormat'] = df['fileFormat'].fillna('html')
            df = df[df['fileFormat'] == 'html']
            articles = {'c_id': c_id, 'c_text': c_text, 'articles': df}
            result = self.run([articles], save=True)[0]
            print(f'Claim id {c_id}: Harvested {len(result)} articles.')
            out = os.path.join(DirConf.HARVEST_RESULTS_DIR, f'{c_id}.csv')
            result.to_csv(out, index=False)
