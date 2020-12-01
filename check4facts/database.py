import pandas
import psycopg2
import numpy as np
from sqlalchemy import create_engine
from psycopg2.extensions import register_adapter, AsIs


def adapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)


def adapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)


def adapt_numpy_float32(numpy_float32):
    return AsIs(numpy_float32)


def adapt_numpy_int32(numpy_int32):
    return AsIs(numpy_int32)


def adapt_numpy_array(numpy_array):
    return AsIs(list(numpy_array))


register_adapter(np.float64, adapt_numpy_float64)
register_adapter(np.int64, adapt_numpy_int64)
register_adapter(np.float32, adapt_numpy_float32)
register_adapter(np.int32, adapt_numpy_int32)
register_adapter(np.ndarray, adapt_numpy_array)


class DBHandler:

    def __init__(self, **kwargs):
        self.conn_params = kwargs

    def insert_claim_articles(self, c_id, article_records):
        conn = None
        sql1 = "SELECT MAX(resource.harvest_iteration) FROM resource" \
               " WHERE resource.statement_id = %s;"
        sql2 = "INSERT INTO resource (url, title, body, sim_paragraph," \
               " sim_sentence, file_format, harvest_date, harvest_iteration," \
               " statement_id) VALUES (%(url)s, %(title)s, %(body)s," \
               " %(sim_paragraph)s, %(sim_sentence)s, %(file_format)s," \
               " %(harvest_date)s, %(harvest_iteration)s, %(statement_id)s);"
        try:
            conn = psycopg2.connect(**self.conn_params)
            cur = conn.cursor()
            cur.execute(sql1, (c_id,))
            res = cur.fetchone()[0]
            h_iter = res + 1 if res else 1
            article_records = [{
                **r, **{'statement_id': c_id, 'harvest_iteration': h_iter}}
                for r in article_records]
            cur.executemany(sql2, article_records)
            conn.commit()
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None: conn.close()

    def insert_claim_features(self, c_id, features_record):
        # engine = create_engine('postgresql://check4facts@localhost:5432/check4facts')
        # features_record.to_sql('feature_statement', engine, if_exists='append', index=False)
        conn = None
        sql = "INSERT INTO feature_statement (s_embedding," \
              " s_subjectivity, s_subjectivity_counts, s_sentiment," \
              " s_sentiment_counts, s_emotion_anger, s_emotion_disgust," \
              " s_emotion_fear, s_emotion_happiness, s_emotion_sadness," \
              " s_emotion_surprise, r_title_embedding, r_title_similarity," \
              " r_title_subjectivity, r_title_subjectivity_counts," \
              " r_title_sentiment, r_title_sentiment_counts," \
              " r_title_emotion_anger, r_title_emotion_disgust," \
              " r_title_emotion_fear, r_title_emotion_happiness," \
              " r_title_emotion_sadness, r_title_emotion_surprise," \
              " r_body_embedding, r_body_similarity, r_body_subjectivity," \
              " r_body_subjectivity_counts, r_body_sentiment, r_body_sentiment_counts," \
              " r_body_emotion_anger, r_body_emotion_disgust," \
              " r_body_emotion_fear, r_body_emotion_happiness," \
              " r_body_emotion_sadness, r_body_emotion_surprise," \
              " r_sim_par_embedding, r_sim_par_similarity," \
              " r_sim_par_subjectivity, r_sim_par_subjectivity_counts," \
              " r_sim_par_sentiment, r_sim_par_sentiment_counts," \
              " r_sim_par_emotion_anger," \
              " r_sim_par_emotion_disgust," \
              " r_sim_par_emotion_fear," \
              " r_sim_par_emotion_happiness," \
              " r_sim_par_emotion_sadness," \
              " r_sim_par_emotion_surprise," \
              " r_sim_sent_embedding, r_sim_sent_similarity," \
              " r_sim_sent_subjectivity, r_sim_sent_subjectivity_counts," \
              " r_sim_sent_sentiment, r_sim_sent_sentiment_counts," \
              " r_sim_sent_emotion_anger," \
              " r_sim_sent_emotion_disgust," \
              " r_sim_sent_emotion_fear," \
              " r_sim_sent_emotion_happiness," \
              " r_sim_sent_emotion_sadness," \
              " r_sim_sent_emotion_surprise, statement_id)" \
              " VALUES (array%(claim.embedding)s," \
              " %(claim.subjectivity)s, %(claim.subjectivity_c)s," \
              " %(claim.sentiment)s, %(claim.sentiment_c)s," \
              " %(claim.emotion.Anger)s, %(claim.emotion.Disgust)s," \
              " %(claim.emotion.Fear)s, %(claim.emotion.Happiness)s," \
              " %(claim.emotion.Sadness)s, %(claim.emotion.Surprise)s," \
              " array%(articles.title.embedding)s," \
              " %(articles.title.similarity)s," \
              " %(articles.title.subjectivity)s," \
              " array%(articles.title.subjectivity_c)s," \
              " %(articles.title.sentiment)s," \
              " array%(articles.title.sentiment_c)s," \
              " %(articles.title.emotion.Anger)s," \
              " %(articles.title.emotion.Disgust)s," \
              " %(articles.title.emotion.Fear)s," \
              " %(articles.title.emotion.Happiness)s," \
              " %(articles.title.emotion.Sadness)s," \
              " %(articles.title.emotion.Surprise)s," \
              " array%(articles.body.embedding)s," \
              " %(articles.body.similarity)s," \
              " %(articles.body.subjectivity)s," \
              " array%(articles.body.subjectivity_c)s," \
              " %(articles.body.sentiment)s," \
              " array%(articles.body.sentiment_c)s," \
              " %(articles.body.emotion.Anger)s," \
              " %(articles.body.emotion.Disgust)s," \
              " %(articles.body.emotion.Fear)s," \
              " %(articles.body.emotion.Happiness)s," \
              " %(articles.body.emotion.Sadness)s," \
              " %(articles.body.emotion.Surprise)s," \
              " array%(articles.sim_paragraph.embedding)s," \
              " %(articles.sim_paragraph.similarity)s," \
              " %(articles.sim_paragraph.subjectivity)s," \
              " array%(articles.sim_paragraph.subjectivity_c)s," \
              " %(articles.sim_paragraph.sentiment)s," \
              " array%(articles.sim_paragraph.sentiment_c)s," \
              " %(articles.sim_paragraph.emotion.Anger)s," \
              " %(articles.sim_paragraph.emotion.Disgust)s," \
              " %(articles.sim_paragraph.emotion.Fear)s," \
              " %(articles.sim_paragraph.emotion.Happiness)s," \
              " %(articles.sim_paragraph.emotion.Sadness)s," \
              " %(articles.sim_paragraph.emotion.Surprise)s," \
              " array%(articles.sim_sentence.embedding)s," \
              " %(articles.sim_sentence.similarity)s," \
              " %(articles.sim_sentence.subjectivity)s," \
              " array%(articles.sim_sentence.subjectivity_c)s," \
              " %(articles.sim_sentence.sentiment)s," \
              " array%(articles.sim_sentence.sentiment_c)s," \
              " %(articles.sim_sentence.emotion.Anger)s," \
              " %(articles.sim_sentence.emotion.Disgust)s," \
              " %(articles.sim_sentence.emotion.Fear)s," \
              " %(articles.sim_sentence.emotion.Happiness)s," \
              " %(articles.sim_sentence.emotion.Sadness)s," \
              " %(articles.sim_sentence.emotion.Surprise)s," \
              " %(statement_id)s);"
        try:
            conn = psycopg2.connect(**self.conn_params)
            cur = conn.cursor()
            features_record['statement_id'] = c_id
            cur.execute(sql, features_record)
            conn.commit()
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None: conn.close()
