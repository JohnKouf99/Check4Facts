import os
import string
import time
import polyglot
import numpy as np
import pandas as pd
import spacy
from nltk.corpus import stopwords
from snowballstemmer import stemmer
from polyglot.text import Text #from polyglot.text import Text
import openai 
from openai import OpenAI
import tiktoken
from transformers import AutoTokenizer, AutoModel
import ollama
import unicodedata
from sklearn.decomposition import PCA
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
        self.llm_embedding_settings = kwargs.get('llm_embeddings', {})

        self.nlp = spacy.load(self.basic_params['model'])
        self.stemmer = stemmer('greek')
        self.lexicon_ = None
        
        if self.llm_embedding_settings:
            self.initialize_llm()

    @property
    def lexicon(self):
        if self.lexicon_ is None:
            self.lexicon_ = pd.read_csv(self.basic_params['lexicon'], sep='\t')
            self.lexicon_.columns = self.lexicon_.columns.str.lower()
            self.lexicon_ = self.lexicon_.fillna('N/A')
            self.lexicon_['lemma'] = self.lexicon_['term'].apply(
                lambda x: self.nlp(x.lower().split()[0])[0].lemma_)
            self.lexicon_['stem'] = self.lexicon_['term'].apply(
                lambda x: self.stemmer.stemWord(x.lower().split()[0]))

            for feat in ['subj', 'sent', 'emo']:
                params = getattr(self, feat + '_params')
                prefixes, scores = params['prefixes'], params['scores']
                cols = [prefix + str(i) for prefix in prefixes for i in
                        range(1, 5)]
                for col in cols:
                    self.lexicon_[col] = self.lexicon_[col].map(scores)
        return self.lexicon_

    def _initialize_ollama_llm(self):
        llm_embeddings = self.llm_embedding_settings.get('ollama')
        model_name = llm_embeddings.get('model_name', 'ilsp/meltemi-instruct:latest')
        #CHANGED
        #self.embedding_model = ollama.load_model(model_name)
        embeddings = ollama.embeddings(model=model_name, prompt='test', keep_alive=-1) # keep the model into memory 
        #get embeddings size for ollama
        self.embedding_size = len(np.array(embeddings['embedding']))
        
       

        # def embed(x:str, model:str='llama3:8b-instruct-q8_0'):
        #     return np.array(ollama.embeddings(model=model, prompt=x)['embedding'])


    def _initialize_transformers_llm(self):
        llm_embeddings = self.llm_embedding_settings.get('transformers')
        model_name = llm_embeddings.get('model_name', 'nlpaueb/bert-base-greek-uncased-v1')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedding_model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.embedding_model.config.hidden_size
        #5/10
        self.embedding_size = self.embedding_model.config.hidden_size

    #4/10
    def _initialize_openai_llm(self):
        os.environ['OPENAI_API_KEY'] = self.llm_embedding_settings.get('openai').get('api_key')
        self.client = OpenAI()
        model_name = self.llm_embedding_settings.get('openai').get('model_name')
        response = self.client.embeddings.create(input='test prompt', model=model_name)
        embedding = response.data[0].embedding
        embedding_array = np.array(embedding)
        self.embedding_size = len(embedding_array)
        

    def initialize_llm(self):
        self.llm_embedding_method = self.llm_embedding_settings["method"]
        if self.llm_embedding_method.lower() == 'transformers':
            self._initialize_transformers_llm()
        elif self.llm_embedding_method.lower() == 'ollama':
            self._initialize_ollama_llm()
        else:
            self._initialize_openai_llm() #4/10

    def get_llm_embeddings(self, text):
        if text is None:
            #print('text is none')
            return None
        if self.llm_embedding_method == 'transformers':
            #4/10 changed from max_length=self.embedding_model.config.hidden_size
            inputs = self.tokenizer(text=text, return_tensors='pt', truncation=True,
                                     max_length=self.embedding_model.config.max_position_embeddings) 

            outputs = self.embedding_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
            #print(embeddings)
            return embeddings.flatten()
        elif self.llm_embedding_method == 'ollama':
            llm_embeddings = self.llm_embedding_settings.get('ollama')
            model_name = llm_embeddings.get('model_name', 'ilsp/meltemi-instruct:latest')
            response = ollama.embeddings(model=model_name, prompt=text, keep_alive=-1) # keep the model into memory 
            embeddings = response['embedding'] # this is a list
            #print(text)
            return np.array(embeddings)
        
            #4/10, #7/10
        elif self.llm_embedding_method == 'openai':
             model_name =self.llm_embedding_settings.get('openai').get('model_name')
             #7/10 if the text is too long (max limit of tokens is 8192), truncate from the end
             tokenizer = tiktoken.encoding_for_model(self.llm_embedding_settings.get('openai').get('model_name'))
             tokenized_text = tokenizer.encode(text)
             token_count = len(tokenized_text)   
             max_tokens = self.llm_embedding_settings.get('openai').get('max_tokens')
             #print(token_count)
             if token_count > max_tokens:
                 truncated_tokenized_text = tokenized_text[:max_tokens]
                 truncated_text = tokenizer.decode(truncated_tokenized_text)
                 #print(truncated_text)
                 response = self.client.embeddings.create(
                 input=truncated_text,model=model_name)
                 return np.array(response.data[0].embedding, dtype=np.float64)
             else:
                response = self.client.embeddings.create(
                input=text,model=model_name)
                return np.array(response.data[0].embedding, dtype=np.float64)
             pass
            # response = openai.Embedding.create(input=text, model=self.llm_embeddings.get('model', 'text-embedding-ada-002'))
            # return np.array(response['data'][0]['embedding'])
        else:
            raise ValueError("Invalid embedding method selected for LLM embeddings")

    #changed it to match the transformer's model preprocessing method
    @staticmethod
    def text_preprocess(text,transf):
        if(transf):
             text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn').lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.lower()
        text = ' '.join([word for word in text.split() if
                         word not in stopwords.words('greek')])
        return text

    @staticmethod
    def get_embedding(sent_doc):
        return sent_doc.vector

    def get_similarity(self, sent_doc, statement):
        statement_doc = self.nlp(self.text_preprocess(statement, transf=False))
        return sent_doc.similarity(statement_doc)

    def get_subjectivity(self, annots):
        cols = [col for col in self.lexicon if col.startswith('subjectivity')]
        return np.mean([np.mean(a[cols].values) for a in annots]) \
            if annots else 0.5

    def get_sentiment(self, annots):
        cols = [col for col in self.lexicon if col.startswith('polarity')]
        return np.mean([np.mean(a[cols].values) for a in annots]) \
            if annots else 0.5

    def get_sentiment_counts(self, annots):
        cols = [col for col in self.lexicon if col.startswith('polarity')]
        scores = [np.mean(a[cols].values) for a in annots]
        neg_tokens = sum(s < self.sent_params['thr']['NEG'] for s in scores)
        pos_tokens = sum(s > self.sent_params['thr']['POS'] for s in scores)
        return np.array([neg_tokens, pos_tokens])

    def get_emotion(self, annots):
        emotions = {}
        for emotion in self.emo_params['prefixes']:
            cols = [col for col in self.lexicon if col.startswith(emotion)]
            scores = [np.mean(a[cols].values) for a in annots] if annots else [0.0]
            n_tokens = sum(s > self.emo_params['thr'] for s in scores)
            emotions[emotion] = np.array(
                [np.min(scores), np.mean(scores), np.max(scores), n_tokens])
        return emotions

    @staticmethod
    def get_pg_polarity_counts(sent):
        text = Text(sent, hint_language_code='el')
        scores = [w.polarity for w in text.words]
        neg_tokens = sum(s < 0 for s in scores)
        pos_tokens = sum(s > 0 for s in scores)
        return np.array([neg_tokens, pos_tokens])

    def aggregate_sentence_features(self, feats_list):
       
        n_sentences = len(feats_list)
        aggr_feats = {'fertile_terms': np.sum([
            f['fertile_terms'] for f in feats_list])}
        
        
        if 'llm_embeddings' in self.basic_params['included_feats']:
            #4/10 changes
            if self.llm_embedding_method.lower() == 'ollama':
                feats = [d.get('llm_embeddings', 
                np.zeros(self.embedding_size)) for d in feats_list]  # Assuming embedding size is 768
            elif self.llm_embedding_method.lower() == 'transformers':
                 feats = [d.get('llm_embeddings', 
                np.zeros(self.embedding_model.config.hidden_size)) for d in feats_list]
            else: 
                 feats = [d.get('llm_embeddings', 
                 np.zeros(self.embedding_size)) for d in feats_list]
            #TODO
            #insert PCA analysis instead of the mean value of all llm_embeddings
            #prior implementation
            #aggr_feats['llm_embeddings'] = np.mean(feats, axis=0)
            
            #7/10
            
            
            #remove zero valued arrays
            feats = np.array([np.array(arr) for arr in feats if not np.all(arr==0)])
            #print(feats.shape)
            #start the pca transformation
            if feats.size!=0:
                pca = PCA(n_components=1)
                feats = np.transpose(feats)
                feats = pca.fit_transform(feats)
                #aggr_feats['llm_embeddings'] = np.mean(aggr_feats['llm_embeddings'], axis=0)
                aggr_feats['llm_embeddings'] = feats.flatten()
                #print(aggr_feats['llm_embeddings'])



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
            # n_obj_terms = np.sum(feats, axis=0)[0]
            # n_subj_terms = np.sum(feats, axis=0)[1]
            # aggr_feats['subjectivity_counts'] = np.array([
            #     n_obj_terms / aggr_feats['fertile_terms'],
            #     n_subj_terms / aggr_feats['fertile_terms']])
            n_obj_sents = sum([True if f[0] > 0 else False for f in feats])
            n_subj_sents = sum([True if f[1] > 0 else False for f in feats])
            aggr_feats['subjectivity_counts'] = np.array([
                n_obj_sents / n_sentences,
                n_subj_sents / n_sentences])
        if 'sentiment' in self.basic_params['included_feats']:
            feats = [d['sentiment'] for d in feats_list]
            aggr_feats['sentiment'] = np.mean(feats, axis=0)
        if 'sentiment_counts' in self.basic_params['included_feats']:
            feats = [d['sentiment_counts'] for d in feats_list]
            # n_neg_terms = np.sum(feats, axis=0)[0]
            # n_pos_terms = np.sum(feats, axis=0)[1]
            # aggr_feats['sentiment_counts'] = np.array([
            #     n_neg_terms / aggr_feats['fertile_terms'],
            #     n_pos_terms / aggr_feats['fertile_terms']])
            n_neg_sents = sum([True if f[0] > 0 else False for f in feats])
            n_pos_sents = sum([True if f[1] > 0 else False for f in feats])
            aggr_feats['sentiment_counts'] = np.array([
                n_neg_sents / n_sentences,
                n_pos_sents / n_sentences])
        if 'emotion' in self.basic_params['included_feats']:
            aggr_feats['emotion'] = {}
            for emotion in self.emo_params['prefixes']:
                feats = [d['emotion'][emotion] for d in feats_list]
                min_ = np.min(feats, axis=0)[0]
                avg_ = np.mean(feats, axis=0)[1]
                max_ = np.max(feats, axis=0)[2]
                # n_emo_terms = np.sum(feats, axis=0)[3]
                # aggr_feats['emotion'][emotion] = np.array([
                #     min_, avg_, max_,
                #     n_emo_terms / aggr_feats['fertile_terms']])
                n_emo_sents = sum([True if f[3] > 0 else False for f in feats])
                aggr_feats['emotion'][emotion] = np.array([
                    min_, avg_, max_, n_emo_sents / n_sentences])
        if 'pg_polarity_counts' in self.basic_params['included_feats']:
            feats = [d['pg_polarity_counts'] for d in feats_list]
            n_neg_sents = sum([True if f[0] > 0 else False for f in feats])
            n_pos_sents = sum([True if f[1] > 0 else False for f in feats])
            aggr_feats['pg_polarity_counts'] = np.array([
                n_neg_sents / n_sentences,
                n_pos_sents / n_sentences])
        return aggr_feats

    def aggregate_body_features(self, feats_list):
        aggr_feats = {'n_pars': len(feats_list)}

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
            n_obj_pars = sum([True if f[0] > 0 else False for f in feats])
            n_subj_pars = sum([True if f[1] > 0 else False for f in feats])
            aggr_feats['subjectivity_counts'] = np.array([
                n_obj_pars, n_subj_pars])
        if 'sentiment' in self.basic_params['included_feats']:
            feats = [d['sentiment'] for d in feats_list]
            aggr_feats['sentiment'] = np.mean(feats, axis=0)
        if 'sentiment_counts' in self.basic_params['included_feats']:
            feats = [d['sentiment_counts'] for d in feats_list]
            n_neg_pars = sum([True if f[0] > 0 else False for f in feats])
            n_pos_pars = sum([True if f[1] > 0 else False for f in feats])
            aggr_feats['sentiment_counts'] = np.array([n_neg_pars, n_pos_pars])
        if 'emotion' in self.basic_params['included_feats']:
            aggr_feats['emotion'] = {}
            for emotion in self.emo_params['prefixes']:
                feats = [d['emotion'][emotion] for d in feats_list]
                min_ = np.min(feats, axis=0)[0]
                avg_ = np.mean(feats, axis=0)[1]
                max_ = np.max(feats, axis=0)[2]
                n_emo_pars = sum([True if f[3] > 0 else False for f in feats])
                aggr_feats['emotion'][emotion] = np.array([
                    min_, avg_, max_, n_emo_pars])
        if 'pg_polarity_counts' in self.basic_params['included_feats']:
            feats = [d['pg_polarity_counts'] for d in feats_list]
            n_neg_pars = sum([True if f[0] > 0 else False for f in feats])
            n_pos_pars = sum([True if f[1] > 0 else False for f in feats])
            aggr_feats['pg_polarity_counts'] = np.array([n_neg_pars, n_pos_pars])
        return aggr_feats

    def aggregate_bodies_features(self, feats_list):
        aggr_feats = {'n_pars': np.sum([f['n_pars'] for f in feats_list])}

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
            n_obj_pars = np.sum(feats, axis=0)[0]
            n_subj_pars = np.sum(feats, axis=0)[1]
            aggr_feats['subjectivity_counts'] = np.array([
                n_obj_pars / aggr_feats['n_pars'] if aggr_feats['n_pars'] != 0 else 0,
                n_subj_pars / aggr_feats['n_pars'] if aggr_feats['n_pars'] != 0 else 0])
        if 'sentiment' in self.basic_params['included_feats']:
            feats = [d['sentiment'] for d in feats_list]
            aggr_feats['sentiment'] = np.mean(feats, axis=0)
        if 'sentiment_counts' in self.basic_params['included_feats']:
            feats = [d['sentiment_counts'] for d in feats_list]
            n_neg_pars = np.sum(feats, axis=0)[0]
            n_pos_pars = np.sum(feats, axis=0)[1]
            aggr_feats['sentiment_counts'] = np.array([
                n_neg_pars / aggr_feats['n_pars'] if aggr_feats['n_pars'] != 0 else 0,
                n_pos_pars / aggr_feats['n_pars'] if aggr_feats['n_pars'] != 0 else 0])
        if 'emotion' in self.basic_params['included_feats']:
            aggr_feats['emotion'] = {}
            for emotion in self.emo_params['prefixes']:
                feats = [d['emotion'][emotion] for d in feats_list]
                min_ = np.min(feats, axis=0)[0]
                avg_ = np.mean(feats, axis=0)[1]
                max_ = np.max(feats, axis=0)[2]
                n_emo_pars = np.sum(feats, axis=0)[3]
                aggr_feats['emotion'][emotion] = np.array([
                    min_, avg_, max_, n_emo_pars / aggr_feats['n_pars']
                    if aggr_feats['n_pars'] != 0 else 0])
        if 'pg_polarity_counts' in self.basic_params['included_feats']:
            feats = [d['pg_polarity_counts'] for d in feats_list]
            n_neg_pars = np.sum(feats, axis=0)[0]
            n_pos_pars = np.sum(feats, axis=0)[1]
            aggr_feats['pg_polarity_counts'] = np.array(
                [n_neg_pars / aggr_feats['n_pars'] if aggr_feats['n_pars'] != 0 else 0,
                 n_pos_pars / aggr_feats['n_pars'] if aggr_feats['n_pars'] != 0 else 0])
        return aggr_feats

    def get_default_sentence_features(self):
        feats = {'fertile_terms': 0}
        if 'embedding' in self.basic_params['included_feats']:
            feats['embedding'] = np.zeros((300, )).astype('float32')
        if 'similarity' in self.basic_params['included_feats']:
            feats['similarity'] = np.float64(0)
        if 'subjectivity' in self.basic_params['included_feats']:
            feats['subjectivity'] = 0.5
        if 'subjectivity_counts' in self.basic_params['included_feats']:
            feats['subjectivity_counts'] = np.array([0, 0])
        if 'sentiment' in self.basic_params['included_feats']:
            feats['sentiment'] = 0.5
        if 'sentiment_counts' in self.basic_params['included_feats']:
            feats['sentiment_counts'] = np.array([0, 0])
        if 'emotion' in self.basic_params['included_feats']:
            feats['emotion'] = {
                e: np.array([0, 0, 0, 0]) for e in self.emo_params['prefixes']}
        if 'pg_polarity_counts' in self.basic_params['included_feats']:
            feats['pg_polarity_counts'] = np.array([0, 0])
        return feats

    def get_sentence_features(self, sent, statement):
        
        #8/10
        #some sentences are equal to float('nan') so it needs handling
        if pd.isna(sent):
            if 'llm_embeddings' in self.basic_params['included_feats']:
                feats = {'fertile_terms': 0}
                feats['llm_embeddings'] = np.zeros(self.embedding_size)
                return feats
            else: 
                sent = ' '
        #if transformers are initialized, text preprocessing should be different 16/10
        if self.llm_embedding_method.lower() == 'transformers':
            sent_doc = self.nlp(self.text_preprocess(sent,transf=True)[:self.nlp.max_length])
        else:
            sent_doc = self.nlp(self.text_preprocess(sent,transf=False)[:self.nlp.max_length])
        # annots = [
        #     self.lexicon[self.lexicon['lemma'] == t.lemma_]
        #     for t in sent_doc if t.lemma_ in self.lexicon['lemma'].values]
        annots = [
            self.lexicon[self.lexicon['stem'] == self.stemmer.stemWord(t.text)]
            for t in sent_doc if
            self.stemmer.stemWord(t.text) in self.lexicon['stem'].values]
        feats = {'fertile_terms': len(sent_doc)}
        if 'llm_embeddings' in self.basic_params['included_feats']:
            feats['llm_embeddings'] = self.get_llm_embeddings(sent)
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
            feats['emotion'] = self.get_emotion(annots)
        if 'pg_polarity_counts' in self.basic_params['included_feats']:
            feats['pg_polarity_counts'] = self.get_pg_polarity_counts(sent)
        return feats

    def get_resource_features(self, title, body, sim_par, sim_sent, statement):
        feats = {}
        if 'title' in self.basic_params['included_resource_parts']:
            if title:
                feats['title'] = self.get_sentence_features(title, statement)
            else:
                feats['title'] = self.get_default_sentence_features()
        if 'body' in self.basic_params['included_resource_parts']:
            if body and pd.notna(body):
                if 'llm_embeddings' in self.basic_params['included_feats']:
                    #take the whole body as a sentence and dont break it into sub-sentences
                    feats['body'] = self.get_sentence_features(body, statement) #4/10
                else:
                    #TODO if body exceeds the limit range, truncate it from the end [:max_token_limit]
                    pars_feats = [self.get_sentence_features(par, statement)
                                    for par in body.splitlines()]
                    feats['body'] = self.aggregate_body_features(pars_feats)
                
                
            else:
                feats['body'] = self.get_default_sentence_features()
                feats['body']['n_pars'] = 0
        if 'sim_par' in self.basic_params['included_resource_parts']:
            if sim_par:
                feats['sim_par'] = self.get_sentence_features(
                    sim_par, statement)
            else:
                feats['sim_par'] = self.get_default_sentence_features()
        if 'sim_sent' in self.basic_params['included_resource_parts']:
            if sim_sent:
                feats['sim_sent'] = self.get_sentence_features(
                    sim_sent, statement)
            else:
                feats['sim_sent'] = self.get_default_sentence_features()
        return feats

    def get_statement_features(self, d):

        s_text = d['s_text']
        s_resources = d['s_resources'].where(pd.notnull(d['s_resources']), None)
        feats = {'s': self.get_sentence_features(s_text, s_text), 'r': None}
        resources_feats = [self.get_resource_features(
            row.title, row.body, row.sim_par, row.sim_sent, s_text)
            for row in s_resources.itertuples()]
        # If the statement has no resources then add the default features of a
        # dummy resource in order to fill its resources' features part
        if len(resources_feats) == 0:
            resources_feats.append(self.get_resource_features(
                None, None, None, None, s_text))
        
        if 'llm_embeddings' in self.basic_params['included_feats']:
            # TODO: Make sure the dimensions are aligned if any of the title, body etc are not included.
            # eg if title features where missing, body llm embeddings would come on the same position of the title's.
            llm_embeddings_for_all_resources = []
            #flag that prohibits extra addition of the s_statement embedding to the r_llm feature
            flag=False
            #8/10 added pd.isna() condition because of nan values on the harvest dataset
            for row in s_resources.itertuples():
                resource_embeddings = []
                if not pd.isna(row.title):
                    title_embedding = self.get_llm_embeddings(text=row.title)
                else:
                    title_embedding = None
                if not pd.isna(row.body):
                     body_embedding = self.get_llm_embeddings(text=row.body)
                else:
                     body_embedding = None
                if not pd.isna(row.sim_par):
                    sim_par_embedding = self.get_llm_embeddings(text=row.sim_par)
                else:
                    sim_par_embedding = None
                if not pd.isna(row.sim_sent):
                    sim_sent_embedding = self.get_llm_embeddings(text=row.sim_sent)
                else:
                    sim_sent_embedding = None
                if not pd.isna(s_text):
                    s_text_embedding = self.get_llm_embeddings(s_text)
                else:
                    s_text_embedding = None

                # body_embedding = self.get_llm_embeddings(text=row.body)
                # sim_par_embedding = self.get_llm_embeddings(text=row.sim_par)
                # sim_sent_embedding = self.get_llm_embeddings(text=row.sim_sent)
                # s_text_embedding = self.get_llm_embeddings(text=s_text)

                #5/10 if embedding is none i dont want to add it to the array
                if title_embedding is not None:  
                    resource_embeddings.append(title_embedding)
                if body_embedding is not None:
                    resource_embeddings.append(body_embedding)
                if sim_par_embedding is not None:
                    resource_embeddings.append(sim_par_embedding)
                if sim_sent_embedding is not None:
                    resource_embeddings.append(sim_sent_embedding)
                #8/10 - Statement is NOT part of a resource so it will not be added to r_llm
                #TODO Comment it out in the future
                # if s_text_embedding is not None and flag==False:
                #     resource_embeddings.append(s_text_embedding)
                #     flag=True

                #print(np.array(resource_embeddings).shape)
                #resource_embeddings = [title_embedding, body_embedding, sim_par_embedding, sim_sent_embedding,  s_text_embedding]
                resource_embeddings = np.array(resource_embeddings).flatten()
                resource_embeddings_np = [np.array(x, dtype=np.float64) for x in resource_embeddings]
                
                #print(np.array(resource_embeddings_np).shape)
                llm_embeddings_for_all_resources.append(resource_embeddings_np)
    
                # Check shapes
                # print(f"title_embedding shape: {np.shape(title_embedding)}")
                # print(f"body_embedding shape: {np.shape(body_embedding)}")
                # print(f"sim_par_embedding shape: {np.shape(sim_par_embedding)}")
                # print(f"sim_sent_embedding shape: {np.shape(sim_sent_embedding)}")
                # print(f"s_text_embedding shape: {np.shape(s_text_embedding)}")
                # print()
            
            #llm_embeddings_for_all_resources = np.array(llm_embeddings_for_all_resources).flatten()
            if llm_embeddings_for_all_resources: 
                llm_embeddings_for_all_resources = np.concatenate([np.array(sub_array, dtype=np.float64) 
                for sub_array in llm_embeddings_for_all_resources])    
                                                       
            
            
        if resources_feats:
            feats['r'] = {}
            if 'title' in self.basic_params['included_resource_parts']:
                feats['r']['title'] = self.aggregate_sentence_features(
                    [d['title'] for d in resources_feats])
            if 'body' in self.basic_params['included_resource_parts']:
                 if 'llm_embeddings' in self.basic_params['included_feats']:
                     #4/10 (i want the body to be treated as a sentence when it comes to llm embedding)
                    feats['r']['body'] = self.aggregate_sentence_features(
                            [d['body'] for d in resources_feats])
                 else:
                    #if we are not implementing llm embeddings, continue as is
                    feats['r']['body'] = self.aggregate_bodies_features(
                      [d['body'] for d in resources_feats]) 
                 
            if 'sim_par' in self.basic_params['included_resource_parts']:
                feats['r']['sim_par'] = self.aggregate_sentence_features(
                    [d['sim_par'] for d in resources_feats])
            if 'sim_sent' in self.basic_params['included_resource_parts']:
                feats['r']['sim_sent'] = self.aggregate_sentence_features(
                    [d['sim_sent'] for d in resources_feats])
            # TODO: Make sure these are added correctly.
            if 'llm_embeddings' in self.basic_params['included_feats']:
                feats['r']['llm'] = llm_embeddings_for_all_resources

            #10/10 create a final embedding that contains all the information gathered
            if 'llm_embeddings' in self.basic_params['included_feats']:
                 #self.text_preprocess(sent)[:self.nlp.max_length]
                 final_emb = {}
                 #final_feats = list()
                 #final_emb['s_llm'] = feats['s']
                 
                 if 'title' in self.basic_params['included_resource_parts']:
                     final_emb['title'] = feats['r']['title']
                 if 'body' in self.basic_params['included_resource_parts']:
                     final_emb['body'] = feats['r']['body']
                 if 'sim_par' in self.basic_params['included_resource_parts']:
                     final_emb['sim_par'] = feats['r']['sim_par']
                 if 'sim_sent' in self.basic_params['included_resource_parts']:
                     final_emb['sim_sent'] = feats['r']['sim_sent']
                
                 training_emb = self.aggregate_sentence_features([d for d in final_emb.values()])
                 feats['r']['final'] = training_emb


                 
                 









        # result = {k: (np.nan_to_num(v) if np.isnan(v).any() else v) for k, v in
        #           flatten_dict(feats).items()}
        result = flatten_dict(feats)
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
