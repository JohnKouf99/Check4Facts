import os
import string
import time

import numpy as np
import pandas as pd
import spacy
from nltk.corpus import stopwords

from check4facts.config import DirConf


def flatten_dict(dd, separator='_', prefix=''):
    return {prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()} \
        if isinstance(dd, dict) else {prefix: dd}


class FeaturesExtractor:

    def __init__(self, **kwargs):
        self.basic_params = kwargs['basic']
        self.emb_params = kwargs['embeddings']
        self.sim_params = kwargs['similarity']
        self.subj_params = kwargs['subjectivity']
        self.sent_params = kwargs['sentiment']
        self.emo_params = kwargs['emotion']

        self.nlp = spacy.load(self.basic_params['model'])
        self.lexicon_ = None

    @property
    def lexicon(self):
        if self.lexicon_ is None:
            self.lexicon_ = pd.read_csv(self.basic_params['lexicon'], sep='\t')
            self.lexicon_.columns = self.lexicon_.columns.str.lower()
            self.lexicon_ = self.lexicon_.fillna('N/A')
            self.lexicon_['lemma'] = self.lexicon_['term'].apply(
                lambda x: self.nlp(x.lower().split()[0])[0].lemma_)

            for feat in ['subj', 'sent', 'emo']:
                params = getattr(self, feat + '_params')
                prefixes, scores = params['prefixes'], params['scores']
                cols = [prefix + str(i) for prefix in prefixes for i in
                        range(1, 5)]
                for col in cols:
                    self.lexicon_[col] = self.lexicon_[col].map(scores)
        return self.lexicon_

    @staticmethod
    def text_preprocess(text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.lower()
        text = ' '.join([word for word in text.split() if
                         word not in stopwords.words('greek')])
        return text

    @staticmethod
    def get_embedding(sent_doc):
        return sent_doc.vector

    def get_similarity(self, sent_doc, statement):
        statement_doc = self.nlp(self.text_preprocess(statement))
        return sent_doc.similarity(statement_doc)

    def get_subjectivity(self, annots):
        cols = [col for col in self.lexicon if col.startswith('subjectivity')]
        scores = [np.mean(a[cols].values) if not a.empty else 0.5 for a in annots]
        return np.mean(scores)

    def get_subjectivity_counts(self, annots):
        cols = [col for col in self.lexicon if col.startswith('subjectivity')]
        scores = [np.mean(a[cols].values) if not a.empty else 0.5 for a in annots]
        obj_tokens = sum(s < self.subj_params['thr']['OBJ'] for s in scores)
        subj_tokens = sum(s > self.subj_params['thr']['SUBJ'] for s in scores)
        return np.array([obj_tokens, subj_tokens])

    def get_sentiment(self, annots):
        cols = [col for col in self.lexicon if col.startswith('polarity')]
        scores = [np.mean(a[cols].values) if not a.empty else 0.5 for a in annots]
        return np.mean(scores)

    def get_sentiment_counts(self, annots):
        cols = [col for col in self.lexicon if col.startswith('polarity')]
        scores = [np.mean(a[cols].values) if not a.empty else 0.5 for a in annots]
        neg_tokens = sum(s < self.sent_params['thr']['NEG'] for s in scores)
        pos_tokens = sum(s > self.sent_params['thr']['POS'] for s in scores)
        return np.array([neg_tokens, pos_tokens])

    def get_emotion(self, annots):
        emotions = {}
        for emotion in self.emo_params['prefixes']:
            cols = [col for col in self.lexicon if col.startswith(emotion)]
            scores = [np.mean(a[cols].values) if not a.empty else 0.0 for a in annots]
            emotions[emotion] = np.array(
                [np.min(scores), np.mean(scores), np.max(scores)])
        return emotions

    def aggregate_features(self, feats_list):
        aggr_feats = {}

        if 'embedding' in self.basic_params['included_feats']:
            feats = [d['embedding'] for d in feats_list]
            aggr_feats['embedding'] = np.mean(feats, axis=0)
        if 'similarity' in self.basic_params['included_feats']:
            feats = [d['similarity'] for d in feats_list]
            aggr_feats['similarity'] = np.mean(feats, axis=0)
        if 'subjectivity' in self.basic_params['included_feats']:
            feats = [d['subjectivity'] for d in feats_list]
            aggr_feats['subjectivity'] = np.mean(feats, axis=0)
        if 'subjectivity_counts' in self.basic_params['included_feats']:
            feats = [d['subjectivity_counts'] for d in feats_list]
            aggr_feats['subjectivity_counts'] = np.array(
                np.mean(feats, axis=0))
        if 'sentiment' in self.basic_params['included_feats']:
            feats = [d['sentiment'] for d in feats_list]
            aggr_feats['sentiment'] = np.mean(feats, axis=0)
        if 'sentiment_counts' in self.basic_params['included_feats']:
            feats = [d['sentiment_counts'] for d in feats_list]
            aggr_feats['sentiment_counts'] = np.array(np.mean(feats, axis=0))
        if 'emotion' in self.basic_params['included_feats']:
            aggr_feats['emotion'] = {}
            for emotion in self.emo_params['prefixes']:
                feats = [d['emotion'][emotion] for d in feats_list]
                min_ = np.min(feats, axis=0)[0]
                avg_ = np.mean(feats, axis=0)[1]
                max_ = np.max(feats, axis=0)[2]
                aggr_feats['emotion'][emotion] = np.array([min_, avg_, max_])
        return aggr_feats

    def get_sentence_features(self, sent, statement):
        sent_doc = self.nlp(self.text_preprocess(sent)[:self.nlp.max_length])
        annots = [self.lexicon[self.lexicon['lemma'] == t.lemma_]
                  for t in sent_doc]
        feats = {}

        if 'embedding' in self.basic_params['included_feats']:
            feats['embedding'] = self.get_embedding(sent_doc)
        if 'similarity' in self.basic_params['included_feats']:
            feats['similarity'] = self.get_similarity(sent_doc, statement)
        if 'subjectivity' in self.basic_params['included_feats']:
            feats['subjectivity'] = self.get_subjectivity(annots)
        if 'subjectivity_counts' in self.basic_params['included_feats']:
            feats['subjectivity_counts'] = self.get_subjectivity_counts(annots)
        if 'sentiment' in self.basic_params['included_feats']:
            feats['sentiment'] = self.get_sentiment(annots)
        if 'sentiment_counts' in self.basic_params['included_feats']:
            feats['sentiment_counts'] = self.get_sentiment_counts(annots)
        if 'emotion' in self.basic_params['included_feats']:
            feats['emotion'] = self.get_emotion(annots)  # TODO decide map value for N/A
        return feats

    def get_resource_features(self, title, body, sim_par, sim_sent, statement):
        feats = {}

        if 'title' in self.basic_params['included_resource_parts']:
            feats['title'] = self.get_sentence_features(title, statement)
        if 'body' in self.basic_params['included_resource_parts']:
            pars_feats = [self.get_sentence_features(par, statement)
                          for par in body.splitlines()]
            feats['body'] = self.aggregate_features(pars_feats)
        if 'sim_par' in self.basic_params['included_resource_parts']:
            feats['sim_par'] = \
                self.get_sentence_features(sim_par, statement)
        if 'sim_sent' in self.basic_params['included_resource_parts']:
            feats['sim_sent'] = \
                self.get_sentence_features(sim_sent, statement)
        return feats

    def get_statement_features(self, d):
        s_text, s_resources = d['s_text'], d['s_resources'].dropna()
        feats = {'s': self.get_sentence_features(s_text, s_text), 'r': None}
        resources_feats = [self.get_resource_features(
            row.title, row.body, row.sim_par, row.sim_sent, s_text)
            for row in s_resources.itertuples()]
        if resources_feats:
            feats['r'] = {resource_part: self.aggregate_features(
                [d[resource_part] for d in resources_feats]) for resource_part
                in self.basic_params['included_resource_parts']}
        # TODO investigate why this is needed. In what case is an aggregated
        #  emotion equal to [np.nan, np.nan, np.nan]? For now just set
        #  nones to 0.0
        result = {k: (np.nan_to_num(v) if np.isnan(v).any() else v) for k, v in
                  flatten_dict(feats).items()}
        return result

    def run(self, statement_dicts):
        return [self.get_statement_features(d) for d in statement_dicts]

    def run_dev(self):
        start_time = time.time()
        if not os.path.exists(DirConf.FEATURES_RESULTS_DIR):
            os.mkdir(DirConf.FEATURES_RESULTS_DIR)
        statement_df = pd.read_csv(DirConf.CSV_FILE)
        for s_id, s_text in zip(statement_df['Fact id'], statement_df['Text']):
            t0 = time.time()
            df = pd.read_csv(
                os.path.join(DirConf.HARVEST_RESULTS_DIR, f'{s_id}.csv'))
            statement_dict = {
                's_id': s_id, 's_text': s_text, 's_resources': df}
            result = self.run([statement_dict])[0]
            t1 = time.time()
            print(f'Statement id {s_id}: Features extracted in '
                  f'{t1-t0:.2f} secs.')
            out = os.path.join(DirConf.FEATURES_RESULTS_DIR, f'{s_id}.json')
            pd.Series(result).to_json(out, indent=4)
        stop_time = time.time()
        print(f'Features extraction done in {stop_time-start_time:.2f} secs.')
