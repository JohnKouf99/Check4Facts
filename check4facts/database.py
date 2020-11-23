import psycopg2


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
        conn = None
        sql = "INSERT INTO feature_statement (c_embedding, c_similarity," \
              " c_subjectivity, c_subjectivity_c, c_sentiment," \
              " c_sentiment_c, c_emotion_anger, c_emotion_disgust," \
              " c_emotion_fear, c_emotion_happiness, c_emotion_sadness," \
              " c_emotion_surprise, a_title_embedding, a_title_similarity," \
              " a_title_subjectivity, a_title_subjectivity_c," \
              " a_title_sentiment, a_title_sentiment_c," \
              " a_title_emotion_anger, a_title_emotion_disgust," \
              " a_title_emotion_fear, a_title_emotion_happiness," \
              " a_title_emotion_sadness, a_title_emotion_surprise," \
              " a_body_embedding, a_body_similarity, a_body_subjectivity," \
              " a_body_subjectivity_c, a_body_sentiment, a_body_sentiment_c," \
              " a_body_emotion_anger, a_body_emotion_disgust," \
              " a_body_emotion_fear, a_body_emotion_happiness," \
              " a_body_emotion_sadness, a_body_emotion_surprise," \
              " a_sim_paragraph_embedding, a_sim_paragraph_similarity," \
              " a_sim_paragraph_subjectivity, a_sim_paragraph_subjectivity_c" \
              " a_sim_paragraph_sentiment, a_sim_paragraph_sentiment_c," \
              " a_sim_paragraph_emotion_anger," \
              " a_sim_paragraph_emotion_disgust," \
              " a_sim_paragraph_emotion_fear," \
              " a_sim_paragraph_emotion_happiness," \
              " a_sim_paragraph_emotion_sadness," \
              " a_sim_paragraph_emotion_surprise," \
              " a_sim_sentence_embedding, a_sim_sentence_similarity," \
              " a_sim_sentence_subjectivity, a_sim_sentence_subjectivity_c" \
              " a_sim_sentence_sentiment, a_sim_sentence_sentiment_c," \
              " a_sim_sentence_emotion_anger," \
              " a_sim_sentence_emotion_disgust," \
              " a_sim_sentence_emotion_fear," \
              " a_sim_sentence_emotion_happiness," \
              " a_sim_sentence_emotion_sadness," \
              " a_sim_sentence_emotion_surprise, statement_id" \
              " VALUES (%(claim.embedding)s, %(claim.similarity)s," \
              " %(claim.subjectivity)s, %(claim.subjectivity_c)s," \
              " %(claim.sentiment)s, %(claim.sentiment_c)s," \
              " %(claim.emotion.Anger)s, %(claim.emotion.Disgust)s," \
              " %(claim.emotion.Fear)s, %(claim.emotion.Happiness)s," \
              " %(claim.emotion.Sadness)s, %(claim.emotion.Surprise)s," \
              " %(articles.title.embedding)s," \
              " %(articles.title.similarity)s," \
              " %(articles.title.subjectivity)s," \
              " %(articles.title.subjectivity_c)s," \
              " %(articles.title.sentiment)s," \
              " %(articles.title.sentiment_c)s," \
              " %(articles.title.emotion.Anger)s," \
              " %(articles.title.emotion.Disgust)s," \
              " %(articles.title.emotion.Fear)s," \
              " %(articles.title.emotion.Happiness)s," \
              " %(articles.title.emotion.Sadness)s," \
              " %(articles.title.emotion.Surprise)s," \
              " %(articles.body.embedding)s," \
              " %(articles.body.similarity)s," \
              " %(articles.body.subjectivity)s," \
              " %(articles.body.subjectivity_c)s," \
              " %(articles.body.sentiment)s," \
              " %(articles.body.sentiment_c)s," \
              " %(articles.body.emotion.Anger)s," \
              " %(articles.body.emotion.Disgust)s," \
              " %(articles.body.emotion.Fear)s," \
              " %(articles.body.emotion.Happiness)s," \
              " %(articles.body.emotion.Sadness)s," \
              " %(articles.body.emotion.Surprise)s," \
              " %(articles.sim_paragraph.embedding)s," \
              " %(articles.sim_paragraph.similarity)s," \
              " %(articles.sim_paragraph.subjectivity)s," \
              " %(articles.sim_paragraph.subjectivity_c)s," \
              " %(articles.sim_paragraph.sentiment)s," \
              " %(articles.sim_paragraph.sentiment_c)s," \
              " %(articles.sim_paragraph.emotion.Anger)s," \
              " %(articles.sim_paragraph.emotion.Disgust)s," \
              " %(articles.sim_paragraph.emotion.Fear)s," \
              " %(articles.sim_paragraph.emotion.Happiness)s," \
              " %(articles.sim_paragraph.emotion.Sadness)s," \
              " %(articles.sim_paragraph.emotion.Surprise)s," \
              " %(articles.sim_sentence.embedding)s," \
              " %(articles.sim_sentence.similarity)s," \
              " %(articles.sim_sentence.subjectivity)s," \
              " %(articles.sim_sentence.subjectivity_c)s," \
              " %(articles.sim_sentence.sentiment)s," \
              " %(articles.sim_sentence.sentiment_c)s," \
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
