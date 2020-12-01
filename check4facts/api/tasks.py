import os
import yaml
from uwsgidecorators import thread
from check4facts.config import DirConf
from check4facts.scripts.features import FeaturesExtractor
from check4facts.scripts.harvest import Harvester
from check4facts.scripts.search import SearchEngine


@thread
def analyze_task(statement, dbh):
    statement_id = statement.get('id')
    statement_text = statement.get('text')

    print(f'Started analyze procedure for statement id: "{statement_id}"')

    path = os.path.join(DirConf.CONFIG_DIR, 'search_config.yml')  # when using uwsgi.
    with open(path, 'r') as f:
        search_params = yaml.safe_load(f)
    se = SearchEngine(**search_params)
    claims = [statement_text]
    # Using first element only for the result cause only one statement is being checked.
    search_results = se.run(claims)[0]

    path = os.path.join(DirConf.CONFIG_DIR, 'harvest_config.yml')  # while using uwsgi.
    with open(path, 'r') as f:
        harvest_params = yaml.safe_load(f)
    h = Harvester(**harvest_params)
    articles = [{
        'c_id': statement_id,
        'c_text': statement_text,
        'c_articles': search_results}]
    # Using first element only for the result cause only one statement is being checked.
    harvest_results = h.run(articles)[0]

    path = os.path.join(DirConf.CONFIG_DIR, 'features_config.yml')  # while using uwsgi
    with open(path, 'r') as f:
        features_params = yaml.safe_load(f)
    fe = FeaturesExtractor(**features_params)
    claim_dicts = [{'c_id': statement_id, 'c_text': statement_text,
                    'c_articles': harvest_results}]
    features_results = fe.run(claim_dicts)[0]

    article_records = harvest_results.to_dict('records')
    dbh.insert_claim_articles(statement_id, article_records)
    print(f'Finished storing harvest results for statement id: "{statement_id}"')
    features_record = features_results.to_dict('records')[0]
    dbh.insert_claim_features(statement_id, features_record)
    print(f'Finished storing features results for statement id: "{statement_id}"')
