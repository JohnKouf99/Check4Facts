import os
import yaml
import datetime

from flask import request, jsonify
from flask_cors import CORS
from flask_marshmallow import Marshmallow
from threading import Thread


from check4facts.api.init import create_app
from check4facts.api.tasks import search_harvest_task
from check4facts.models.models import db, Statement, Resource
from check4facts.config import DirConf
from check4facts.scripts.search import SearchEngine
from check4facts.scripts.harvest import Harvester

app = create_app()
app.app_context().push()
ma = Marshmallow(app)
CORS(app)


class ResourceSchema(ma.Schema):
    class Meta:
        fields = ("body", "fileFormat", "harvestDate", "harvestIteration", "id", "simParagraph", "simSentence", "title",
                  "url")


class StatementSourcesSchema(ma.Schema):
    class Meta:
        fields = ("id", "snippet", "title", "url")


class TopicSchema(ma.Schema):
    class Meta:
        fields = ("id", "name")


class StatementSchema(ma.Schema):
    class Meta:
        fields = ("author", "id", "main_article_text", "main_article_url", "registration_date", "statement_date", "text"
                  , "topic")

    topic = ma.Nested(TopicSchema)


statement_schema = StatementSchema(many=True)
resources_schema = ResourceSchema(many=True)


@app.route('/search_harvest', methods=['POST'])
def search_harvest():
    statement = request.json

    search_harvest_task(app, db, statement)

    return jsonify({'started': True})


# Successfully returns the list of all statements in database.
@app.route('/statement')
def fetch_statements():
    statements = Statement.query.all()
    result = statement_schema.dump(statements)
    return {"statements": result}


if __name__ == '__main__':
    app.run(debug=True)
