import numpy as np
import psycopg2
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

    def insert_statement_resources(self, s_id, resource_records):
        conn = None
        sql1 = "SELECT MAX(resource.harvest_iteration) FROM resource" \
               " WHERE resource.statement_id = %s;"
        sql2 = "INSERT INTO resource (url, title, body, sim_paragraph," \
               " sim_sentence, file_format, harvest_date, harvest_iteration," \
               " statement_id) VALUES (%(url)s, %(title)s, %(body)s," \
               " %(sim_par)s, %(sim_sent)s, %(file_format)s," \
               " %(harvest_date)s, %(harvest_iteration)s, %(statement_id)s);"
        try:
            conn = psycopg2.connect(**self.conn_params)
            cur = conn.cursor()
            cur.execute(sql1, (s_id,))
            res = cur.fetchone()[0]
            h_iter = res + 1 if res else 1
            resource_records = [{
                **r, **{'statement_id': s_id, 'harvest_iteration': h_iter}}
                for r in resource_records]
            cur.executemany(sql2, resource_records)
            conn.commit()
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None: conn.close()

    def insert_statement_features(self, s_id, features_record, s_preds):
        conn = None
        sql = "INSERT INTO feature_statement (" \
              " s_embedding," \
              " s_subjectivity," \
              " s_subjectivity_counts," \
              " s_sentiment," \
              " s_sentiment_counts," \
              " s_emotion_anger," \
              " s_emotion_disgust," \
              " s_emotion_fear," \
              " s_emotion_happiness," \
              " s_emotion_sadness," \
              " s_emotion_surprise," \
              " r_title_embedding," \
              " r_title_similarity," \
              " r_title_subjectivity," \
              " r_title_subjectivity_counts," \
              " r_title_sentiment," \
              " r_title_sentiment_counts," \
              " r_title_emotion_anger," \
              " r_title_emotion_disgust," \
              " r_title_emotion_fear," \
              " r_title_emotion_happiness," \
              " r_title_emotion_sadness," \
              " r_title_emotion_surprise," \
              " r_body_embedding," \
              " r_body_similarity," \
              " r_body_subjectivity," \
              " r_body_subjectivity_counts," \
              " r_body_sentiment," \
              " r_body_sentiment_counts," \
              " r_body_emotion_anger," \
              " r_body_emotion_disgust," \
              " r_body_emotion_fear," \
              " r_body_emotion_happiness," \
              " r_body_emotion_sadness," \
              " r_body_emotion_surprise," \
              " r_sim_par_embedding," \
              " r_sim_par_similarity," \
              " r_sim_par_subjectivity," \
              " r_sim_par_subjectivity_counts," \
              " r_sim_par_sentiment," \
              " r_sim_par_sentiment_counts," \
              " r_sim_par_emotion_anger," \
              " r_sim_par_emotion_disgust," \
              " r_sim_par_emotion_fear," \
              " r_sim_par_emotion_happiness," \
              " r_sim_par_emotion_sadness," \
              " r_sim_par_emotion_surprise," \
              " r_sim_sent_embedding," \
              " r_sim_sent_similarity," \
              " r_sim_sent_subjectivity," \
              " r_sim_sent_subjectivity_counts," \
              " r_sim_sent_sentiment," \
              " r_sim_sent_sentiment_counts," \
              " r_sim_sent_emotion_anger," \
              " r_sim_sent_emotion_disgust," \
              " r_sim_sent_emotion_fear," \
              " r_sim_sent_emotion_happiness," \
              " r_sim_sent_emotion_sadness," \
              " r_sim_sent_emotion_surprise," \
              " predict_label," \
              " statement_id)" \
              " VALUES (" \
              " array%(s_embedding)s," \
              " %(s_subjectivity)s," \
              " %(s_subjectivity_counts)s," \
              " %(s_sentiment)s," \
              " %(s_sentiment_counts)s," \
              " %(s_emotion_anger)s," \
              " %(s_emotion_disgust)s," \
              " %(s_emotion_fear)s," \
              " %(s_emotion_happiness)s," \
              " %(s_emotion_sadness)s," \
              " %(s_emotion_surprise)s," \
              " array%(r_title_embedding)s," \
              " %(r_title_similarity)s," \
              " %(r_title_subjectivity)s," \
              " %(r_title_subjectivity_counts)s," \
              " %(r_title_sentiment)s," \
              " %(r_title_sentiment_counts)s," \
              " %(r_title_emotion_anger)s," \
              " %(r_title_emotion_disgust)s," \
              " %(r_title_emotion_fear)s," \
              " %(r_title_emotion_happiness)s," \
              " %(r_title_emotion_sadness)s," \
              " %(r_title_emotion_surprise)s," \
              " array%(r_body_embedding)s," \
              " %(r_body_similarity)s," \
              " %(r_body_subjectivity)s," \
              " %(r_body_subjectivity_counts)s," \
              " %(r_body_sentiment)s," \
              " %(r_body_sentiment_counts)s," \
              " %(r_body_emotion_anger)s," \
              " %(r_body_emotion_disgust)s," \
              " %(r_body_emotion_fear)s," \
              " %(r_body_emotion_happiness)s," \
              " %(r_body_emotion_sadness)s," \
              " %(r_body_emotion_surprise)s," \
              " array%(r_sim_par_embedding)s," \
              " %(r_sim_par_similarity)s," \
              " %(r_sim_par_subjectivity)s," \
              " %(r_sim_par_subjectivity_counts)s," \
              " %(r_sim_par_sentiment)s," \
              " %(r_sim_par_sentiment_counts)s," \
              " %(r_sim_par_emotion_anger)s," \
              " %(r_sim_par_emotion_disgust)s," \
              " %(r_sim_par_emotion_fear)s," \
              " %(r_sim_par_emotion_happiness)s," \
              " %(r_sim_par_emotion_sadness)s," \
              " %(r_sim_par_emotion_surprise)s," \
              " array%(r_sim_sent_embedding)s," \
              " %(r_sim_sent_similarity)s," \
              " %(r_sim_sent_subjectivity)s," \
              " %(r_sim_sent_subjectivity_counts)s," \
              " %(r_sim_sent_sentiment)s," \
              " %(r_sim_sent_sentiment_counts)s," \
              " %(r_sim_sent_emotion_anger)s," \
              " %(r_sim_sent_emotion_disgust)s," \
              " %(r_sim_sent_emotion_fear)s," \
              " %(r_sim_sent_emotion_happiness)s," \
              " %(r_sim_sent_emotion_sadness)s," \
              " %(r_sim_sent_emotion_surprise)s," \
              " %(predict_label)s," \
              " %(predict_proba)s," \
              " %(statement_id)s);"
        try:
            conn = psycopg2.connect(**self.conn_params)
            cur = conn.cursor()
            features_record['predict_label'] = np.argmax(s_preds)
            features_record['predict_proba'] = np.max(s_preds)
            features_record['statement_id'] = s_id
            cur.execute(sql, features_record)
            conn.commit()
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None: conn.close()

    # TODO
    def fetch_statement_features(self, features):
        conn, res = None, None
        sql = "SELECT {} FROM feature_statement;".format(', '.join(features))
        try:
            conn = psycopg2.connect(**self.conn_params)
            cur = conn.cursor()
            cur.execute(sql)
            res = cur.fetchall()
            conn.commit()
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None: conn.close()
            return res
