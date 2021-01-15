import os
import yaml
from celery import shared_task
from check4facts.config import DirConf
from check4facts.predict import Predictor
from check4facts.database import DBHandler
from check4facts.scripts.harvest import Harvester
from check4facts.scripts.search import SearchEngine
from check4facts.scripts.features import FeaturesExtractor


db_path = os.path.join(DirConf.CONFIG_DIR, 'db_config.yml')  # while using uwsgi
with open(db_path, 'r') as db_f:
    db_params = yaml.safe_load(db_f)
dbh = DBHandler(**db_params)


@shared_task(bind=True)
def analyze_task(self, statement):
    statement_id = statement.get('id')
    statement_text = statement.get('text')

    print(f'[Worker: {os.getpid()}] Started analyze procedure for statement id: "{statement_id}"')

    path = os.path.join(DirConf.CONFIG_DIR, 'search_config.yml')  # when using uwsgi.
    with open(path, 'r') as f:
        search_params = yaml.safe_load(f)
    se = SearchEngine(**search_params)
    statements = [statement_text]
    # Currently state of task is not queried.
    self.update_state(state='PROGRESS',
                      meta={'current': 1, 'total': 4,
                            'status': f'Initiated Search for statementd id: "{statement_id}"'})
    # Using first element only for the result cause only one statement is being checked.
    search_results = se.run(statements)[0]

    path = os.path.join(DirConf.CONFIG_DIR, 'harvest_config.yml')  # while using uwsgi.
    with open(path, 'r') as f:
        harvest_params = yaml.safe_load(f)
    h = Harvester(**harvest_params)
    articles = [{
        's_id': statement_id,
        's_text': statement_text,
        's_resources': search_results}]
    # Currently state of task is not queried.
    self.update_state(state='PROGRESS',
                      meta={'current': 2, 'total': 4,
                            'status': f'Initiated Harvest for statementd id: "{statement_id}"'})
    # Using first element only for the result cause only one statement is being checked.
    harvest_results = h.run(articles)[0]

    if harvest_results.empty:
        print(f'[Worker: {os.getpid()}] No resources found for statement id: "{statement_id}"')
        return

    path = os.path.join(DirConf.CONFIG_DIR, 'features_config.yml')  # while using uwsgi
    with open(path, 'r') as f:
        features_params = yaml.safe_load(f)
    fe = FeaturesExtractor(**features_params)
    statement_dicts = [{'s_id': statement_id, 's_text': statement_text,
                        's_resources': harvest_results}]
    # Currently state of task is not queried.
    self.update_state(state='PROGRESS',
                      meta={'current': 3, 'total': 4,
                            'status': f'Initiated Feature Extraction for statementd id: "{statement_id}"'})
    features_results = fe.run(statement_dicts)[0]

    path = os.path.join(DirConf.CONFIG_DIR, 'predict_config.yml')
    with open(path, 'r') as f:
        predict_params = yaml.safe_load(f)
    p = Predictor(**predict_params)
    # Currently state of task is not queried.
    self.update_state(state='PROGRESS',
                      meta={'current': 4, 'total': 4,
                            'status': f'Initiated Predict for statementd id: "{statement_id}"'})
    predict_result = p.run([features_results])[0]

    resource_records = harvest_results.to_dict('records')
    dbh.insert_statement_resources(statement_id, resource_records)
    print(f'[Worker: {os.getpid()}] Finished storing harvest results for statement id: "{statement_id}"')
    dbh.insert_statement_features(statement_id, features_results, predict_result)
    print(f'[Worker: {os.getpid()}] Finished storing features results for statement id: "{statement_id}"')
