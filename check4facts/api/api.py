from celery import Celery
from flask_cors import CORS
from flask import request, jsonify
from check4facts.api.init import create_app
from check4facts.api.tasks import analyze_task

app = create_app()
app.config['CELERY_BROKER_URL'] = 'sqla+postgresql://check4facts@localhost:5432/check4facts'

client = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
client.conf.update(app.config)
CORS(app)


@app.route('/analyze', methods=['POST'])
def analyze():
    statement = request.json

    task = analyze_task.apply_async(kwargs={'statement': statement})

    return jsonify({'started': True, 'taskId': task.task_id})


if __name__ == '__main__':
    app.run(debug=True)
