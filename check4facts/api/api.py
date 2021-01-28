import time
from celery import Celery
from flask_cors import CORS
from flask import request, jsonify
from check4facts.api.init import create_app
from check4facts.api.tasks import status_task, analyze_task, train_task

app = create_app()
app.config['CELERY_BROKER_URL'] = 'sqla+postgresql://check4facts@localhost:5432/check4facts'
app.config['result_backend'] = 'db+postgresql://check4facts@localhost:5432/check4facts'

client = Celery(app.name, backend=app.config['result_backend'], broker=app.config['CELERY_BROKER_URL'])
client.conf.update(app.config)
CORS(app)


def inspect(method):
    inspect_app = Celery(app.name, backend=app.config['result_backend'], broker=app.config['CELERY_BROKER_URL'])
    return getattr(inspect_app.control.inspect(), method)()


@app.route('/analyze', methods=['POST'])
def analyze():
    statement = request.json

    task = analyze_task.apply_async(kwargs={'statement': statement})

    return jsonify({'started': True, 'taskId': task.task_id})


@app.route('/train', methods=['POST'])
def train():

    task = train_task.apply_async(task_id=f"train_task_on_{time.strftime('%Y-%m-%d-%H:%M')}")

    return jsonify({'started': True, 'taskId': task.task_id})


@app.route('/task-status/<task_id>', methods=['GET'])
def task_status(task_id):

    result = status_task(task_id)

    return jsonify({'taskId': task_id, 'status': result.status, 'info': result.info})


@app.route('/active-tasks', methods=['GET'])
def active_tasks():

    return jsonify({'tasks': inspect('active')})


if __name__ == '__main__':
    app.run(debug=True)
