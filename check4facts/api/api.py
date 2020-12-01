import os

import yaml
from flask import request, jsonify
from flask_cors import CORS


from check4facts.api.init import create_app
from check4facts.api.tasks import analyze_task
from check4facts.config import DirConf
from check4facts.database import DBHandler

app = create_app()
CORS(app)

path = os.path.join(DirConf.CONFIG_DIR, 'db_config.yml')  # while using uwsgi
with open(path, 'r') as f:
    db_params = yaml.safe_load(f)
    dbh = DBHandler(**db_params)


@app.route('/search_harvest', methods=['POST'])
def analyze():
    statement = request.json

    analyze_task(statement, dbh)

    return jsonify({'started': True})


if __name__ == '__main__':
    app.run(debug=True)
