from os import system
from flask import Flask
from flask_restful import Resource, Api, reqparse

c4fCLI = 'python -m check4facts.cli'

app = Flask(__name__)
api = Api(app)

# Required for parsing the request arguments/parameters.
parser = reqparse.RequestParser()


# noinspection PyMethodMayBeStatic
class ApiHarvest(Resource):
    def get(self):
        system(c4fCLI + ' harvest')
        # Send a pretty simple success response after execution.
        return 'Harvest Complete', 201


# noinspection PyMethodMayBeStatic
class ApiSearch(Resource):
    def get(self):
        system(c4fCLI + ' search')
        # Send a pretty simple success response after execution.
        return 'Search Complete', 201


# Setup the API routing.
api.add_resource(ApiHarvest, '/harvest')
api.add_resource(ApiSearch, '/search')

if __name__ == '__main__':
    app.run(debug=True)
