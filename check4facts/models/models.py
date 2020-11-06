import flask_sqlalchemy

db = flask_sqlalchemy.SQLAlchemy()


class SubTopic(db.Model):
    __tablename__ = 'sub_topic'

    id = db.Column(db.BigInteger, primary_key=True)
    name = db.Column(db.String(255), nullable=False, unique=True)


class Topic(db.Model):
    __tablename__ = 'topic'

    id = db.Column(db.BigInteger, primary_key=True)
    name = db.Column(db.String(255), nullable=False, unique=True)


class Statement(db.Model):
    __tablename__ = 'statement'

    id = db.Column(db.BigInteger, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    author = db.Column(db.String(255))
    statement_date = db.Column(db.DateTime)
    registration_date = db.Column(db.DateTime)
    main_article_text = db.Column(db.Text)
    main_article_url = db.Column(db.Text)
    topic_id = db.Column(db.ForeignKey('topic.id'))

    topic = db.relationship('Topic')
    sub_topicss = db.relationship('SubTopic', secondary='statement_sub_topics')


class Resource(db.Model):
    __tablename__ = 'resource'

    id = db.Column(db.BigInteger, primary_key=True)
    url = db.Column(db.Text, nullable=False)
    harvest_iteration = db.Column(db.BigInteger, nullable=False)
    title = db.Column(db.Text, nullable=False)
    sim_sentence = db.Column(db.Text)
    sim_paragraph = db.Column(db.Text)
    file_format = db.Column(db.String(255))
    body = db.Column(db.Text)
    harvest_date = db.Column(db.DateTime, nullable=False)
    statement_id = db.Column(db.ForeignKey('statement.id'))

    statement = db.relationship('Statement')


class StatementSource(db.Model):
    __tablename__ = 'statement_source'

    id = db.Column(db.BigInteger, primary_key=True)
    url = db.Column(db.Text, nullable=False)
    title = db.Column(db.String(255))
    snippet = db.Column(db.Text)
    statement_id = db.Column(db.ForeignKey('statement.id'))

    statement = db.relationship('Statement')


t_statement_sub_topics = db.Table(
    'statement_sub_topics', db.Model.metadata,
    db.Column('sub_topics_id', db.ForeignKey('sub_topic.id'), primary_key=True, nullable=False),
    db.Column('statement_id', db.ForeignKey('statement.id'), primary_key=True, nullable=False)
)
