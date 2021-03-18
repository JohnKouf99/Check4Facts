import os
import time
import yaml
from celery import Celery
from flask_cors import CORS
from flask import request, jsonify
from check4facts.api.init import create_app
from check4facts.api.tasks import status_task, analyze_task, train_task, intial_train_task
from check4facts.config import DirConf
from check4facts.database import DBHandler

app = create_app()
app.config['CELERY_BROKER_URL'] = 'sqla+postgresql://check4facts@localhost:5432/check4facts'
app.config['result_backend'] = 'db+postgresql://check4facts@localhost:5432/check4facts'

client = Celery(app.name, backend=app.config['result_backend'], broker=app.config['CELERY_BROKER_URL'])
client.conf.update(app.config)
CORS(app)


@app.route('/analyze', methods=['POST'])
def analyze():
    statement = request.json

    task = analyze_task.apply_async(kwargs={'statement': statement})

    return jsonify({'status': 'PROGRESS',
                    'taskId': task.task_id,
                    'taskInfo': {'current': 1, 'total': 4,
                                 'type': f'{statement.get("id")}'}
                    })


@app.route('/train', methods=['POST'])
def train():
    task = train_task.apply_async(task_id=f"train_task_on_{time.strftime('%Y-%m-%d-%H:%M')}")

    return jsonify({'status': 'PROGRESS',
                    'taskId': task.task_id,
                    'taskInfo': {'current': 1, 'total': 2, 'type': 'TRAIN'}
                    })


@app.route('/intial-train', methods=['GET'])
def initial_train():
    db_path = os.path.join(DirConf.CONFIG_DIR, 'db_config.yml')  # while using uwsgi
    with open(db_path, 'r') as db_f:
        db_params = yaml.safe_load(db_f)
    dbh = DBHandler(**db_params)
    total = dbh.count_statements()

    task = intial_train_task.apply_async()

    return jsonify({
        'status': 'PROGRESS',
        'taskId': task.task_id,
        'taskInfo': {
            'current': 1,
            'total': (4 * total) + 1,
            'type': 'INITIAL_TRAIN'
        }
    })


@app.route('/task-status/<task_id>', methods=['GET'])
def task_status(task_id):
    result = status_task(task_id)

    return jsonify({'taskId': task_id, 'status': result.status, 'taskInfo': result.info})


@app.route('/batch-task-status', methods=['POST'])
def batch_task_status():
    json = request.json

    response = []
    for j in json:
        result = status_task(j['id'])
        response.append({'taskId': j['id'], 'status': result.status, 'taskInfo': result.info})

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
