import os
import yaml
import datetime
from uwsgidecorators import thread
from check4facts.config import DirConf
from check4facts.scripts.harvest import Harvester
from check4facts.scripts.search import SearchEngine
from check4facts.models.models import Resource


@thread
def search_harvest_task(app, db, statement):
    # path = os.path.join('../../', DirConf.CONFIG_DIR, 'search_config.yml')
    path = os.path.join(DirConf.CONFIG_DIR, 'search_config.yml')  # when using uwsgi.
    with open(path, 'r') as f:
        search_params = yaml.safe_load(f)
    se = SearchEngine(**search_params)
    claims = [statement.get('text')]
    print(f'Mpika me claims {claims[0]}')
    # Using first element only for the result cause only one statement is being checked.
    search_results = se.run(claims)[0]

    # Clean results from pdf/doc links.
    if 'fileFormat' not in search_results:
        search_results['fileFormat'] = 'html'
    search_results['fileFormat'] = search_results['fileFormat'].fillna('html')
    search_results = search_results[search_results['fileFormat'] == 'html']

    # path = os.path.join('../../', DirConf.CONFIG_DIR, 'harvest_config.yml')
    path = os.path.join(DirConf.CONFIG_DIR, 'harvest_config.yml')  # while using uwsgi.
    with open(path, 'r') as f:
        harvest_params = yaml.safe_load(f)
    h = Harvester(**harvest_params)
    articles = [{
        'c_id': statement.get('id'),
        'c_text': statement.get('text'),
        'c_articles': search_results}]
    # Using first element only for the result cause only one statement is being checked.
    harvest_results = h.run(articles)[0]
    # Maybe remove this array.
    # response_result = []

    with app.app_context():
        for row in harvest_results.itertuples():
            if row.title is not None:
                # response_result.append({
                #     'url': row.url,
                #     'title': row.title,
                #     'body': row.body,
                #     'simParagraph': row.sim_par,
                #     'simSentence': row.sim_sent,
                # })
                # Create a resource according to database model and insert it
                resource = Resource(url=row.url,
                                    harvest_iteration=1, title=row.title, sim_paragraph=row.sim_par,
                                    sim_sentence=row.sim_sent, file_format='NONE', statement_id=statement.get('id'),
                                    body=row.body, harvest_date=statement.get('registrationDate'))
                db.session.add(resource)
        # After all resources are inserted commit to database (Check for potential overflow of memory)
        db.session.commit()
