import os
import string
import datetime
import time

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag, NavigableString, Comment
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
        if not texts:
            return None
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

    @staticmethod
    def get_resource_title(html):
        title = None
        if html.title:
            title = html.title.text.strip()
        elif html.find('meta', property='og:title'):
            title = html.find(
                'meta', property='og:title').get('content').strip()
        elif html.find('h1'):
            title = html.find('h1').text.strip()
        return title

    @staticmethod
    def get_resource_body_paragraphs(html):
        if not html.body: return None

        texts = [' '.join(text.split()) for text in html.body.find_all(
            text=lambda text: not isinstance(text, Comment))]
        texts = [text for text in texts if text]

        # texts = []
        # for child in html.descendants:
        #     if type(child) == NavigableString:
        #         parents = [t for t in child.parents if type(t) == Tag]
        #         if any(parent.name in self.html_params['blacklist']
        #                for parent in parents):
        #             continue
        #         text = ' '.join(child.string.split())
        #         if text:
        #             if child.parent.name == 'a':
        #                 a_tag = child.parent
        #                 text = a_tag.get_text().strip()
        #                 if (type(a_tag.previous_sibling) == NavigableString and
        #                         a_tag.previous_sibling.string.strip()):
        #                     texts[-1] = texts[-1] + ' ' + text
        #                     continue
        #             elif child.previous_sibling and \
        #                     child.previous_sibling.name == 'a':
        #                 texts[-1] = texts[-1] + ' ' + text
        #                 continue
        #             texts += [text]
        # texts = [' '.join(text.split()) for text in texts if text]
        return texts

    def harvest_resource(self, s_id, s_text, r_idx, r_url, save=False):
        result = {
            'title': None, 'body': None, 'sim_par': None, 'sim_sent': None}

        try:
            response = requests.get(
                r_url,
                headers=self.html_params['headers'],
                timeout=self.html_params['timeout'])
            response.raise_for_status()

            # Skipping aggregator web resources
            if len(response.content) > self.html_params['max_content_size']:
                if save: self.save_content_to_xml(
                    'Too large content. Skipped.', f'{s_id}_{r_idx}.xml')
                return result

        except requests.exceptions.RequestException as e:
            if save: self.save_content_to_xml(
                f'{type(e)} :: {e}', f'{s_id}_{r_idx}.xml')
            return result

        html = BeautifulSoup(response.content, self.html_params['parser'])
        if save: self.save_content_to_xml(
            html.prettify(), f'{s_id}_{r_idx}.xml')

        for element in html(self.html_params['blacklist']):
            element.extract()

        paragraphs = self.get_resource_body_paragraphs(html)
        if paragraphs:
            paragraphs = [p for p in paragraphs if len(p.split()) > 3]
        if paragraphs:
            sentences = [s for p in paragraphs for s in sent_tokenize(p)]
            sentences = [s for s in sentences if len(s.split()) > 3]
            sim_par = self.most_similar(paragraphs, s_text)
            sim_sent = self.most_similar(sentences, s_text)
            body = '\n'.join(paragraphs)
        else:
            body, sim_par, sim_sent = None, None, None

        result = {
            'title': self.get_resource_title(html),
            'body': body,
            'sim_par': sim_par,
            'sim_sent': sim_sent
        }
        return result

    @staticmethod
    def filter_resource_format(df, format_):
        if 'fileFormat' not in df: df['fileFormat'] = 'html'
        df['fileFormat'] = df['fileFormat'].fillna('html')
        df = df[df['fileFormat'] == format_]
        return df

    def harvest_resources(self, d, save=False):
        s_id, s_text, s_resources = d['s_id'], d['s_text'], d['s_resources']
        s_resources = self.filter_resource_format(
            s_resources, self.basic_params['resource_format'])
        r_idxs = s_resources['index'] if not s_resources.empty else []
        r_urls = s_resources['link'] if not s_resources.empty else []
        now = datetime.datetime.utcnow()
        data = [{**{'index': r_idx, 'url': r_url, 'harvest_date': now,
                    'file_format': self.basic_params['resource_format']},
                 **self.harvest_resource(s_id, s_text, r_idx, r_url, save)}
                for r_idx, r_url in zip(r_idxs, r_urls)]
        result = pd.DataFrame(data)
        if result.empty: result = result.reset_index()
        return result

    def run(self, statement_dicts, save=False):
        return [self.harvest_resources(d, save) for d in statement_dicts]

    def run_dev(self):
        start_time = time.time()
        if not os.path.exists(DirConf.HARVEST_RESULTS_DIR):
            os.mkdir(DirConf.HARVEST_RESULTS_DIR)
        if not os.path.exists(DirConf.HARVEST_XML_DIR):
            os.mkdir(DirConf.HARVEST_XML_DIR)
        statement_df = pd.read_csv(DirConf.CSV_FILE)
        for s_id, s_text in zip(statement_df['Fact id'], statement_df['Text']):
            t0 = time.time()
            df = pd.read_csv(
                os.path.join(DirConf.SEARCH_RESULTS_DIR, f'{s_id}.csv'))
            statement_dict = {
                's_id': s_id, 's_text': s_text, 's_resources': df}
            result = self.run([statement_dict], save=True)[0]
            t1 = time.time()
            print(f'Statement id {s_id}: Harvested {len(result)} resources '
                  f'in {t1-t0:.2f} secs.')
            out = os.path.join(DirConf.HARVEST_RESULTS_DIR, f'{s_id}.csv')
            result.to_csv(out, index=False)
        stop_time = time.time()
        print(f'Harvest done in {stop_time-start_time:.2f} secs.')
