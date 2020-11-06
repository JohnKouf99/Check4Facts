from flask_cors import CORS
from flask_marshmallow import Marshmallow

from check4facts.api.init import create_app
from check4facts.models.models import Statement

app = create_app()
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
    # TODO add search.run() and harvest.run() here and send back results.
    # statement = statement_schema.dump(request.json)
    return resources_schema.jsonify([
        {
            "body": "string",
            "fileFormat": "PDF",
            "harvestDate": "2020-11-06T10:58:30.672Z",
            "harvestIteration": 0,
            "id": 0,
            "simParagraph": "string",
            "simSentence": "string",
            "statement": {
                "author": "string",
                "id": 0,
                "mainArticleText": "string",
                "mainArticleUrl": "string",
                "registrationDate": "2020-11-06T10:58:30.672Z",
                "statementDate": "2020-11-06T10:58:30.672Z",
                "statementSources": [
                    {
                        "id": 0,
                        "snippet": "string",
                        "title": "string",
                        "url": "string"
                    }
                ],
                "subTopics": [
                    {
                        "id": 0,
                        "name": "string"
                    }
                ],
                "text": "string",
                "topic": {
                    "id": 0,
                    "name": "string"
                }
            },
            "title": "string",
            "url": "string"
        }
    ])

# Successfully returns the list of all statements in database.
@app.route('/statement')
def fetch_statements():
    statements = Statement.query.all()
    result = statement_schema.dump(statements)
    return {"statements": result}


if __name__ == '__main__':
    app.run(debug=True)
