from flask import Flask

"""
This is responsible for creating the API layer app for our python module Check4Facts
"""


def create_app():
    flask_app = Flask(__name__)
    # flask_app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql+psycopg2://check4facts@localhost:5432/check4facts'
    # flask_app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    # db.init_app(flask_app)

    return flask_app
