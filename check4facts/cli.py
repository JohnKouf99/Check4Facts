import os
import argparse

import pandas as pd
import yaml

from check4facts.scripts.search import SearchEngine
from check4facts.scripts.harvest import Harvester
from check4facts.scripts.features import FeaturesExtractor
from check4facts.train import Trainer
from check4facts.predict import Predictor
from check4facts.database import DBHandler
from check4facts.config import DirConf


class Interface:
    """
    This is the CLI of the C4F project. It is responsible
    for handling different types of actions based on given arguments.
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='Run various ops of the C4F project.')

        self.subparsers = self.parser.add_subparsers(
            help='sub-command help', dest='action')

        # create parser for "search_dev" command
        self.search_dev_parser = self.subparsers.add_parser(
            'search_dev', help='triggers relevant resource search (dev)')

        # create parser for "search" command
        self.search_parser = self.subparsers.add_parser(
            'search', help='triggers relevant resource search')

        # create parser for "harvest_dev" command
        self.harvest_dev_parser = self.subparsers.add_parser(
            'harvest_dev', help='triggers search results harvest (dev)')

        # create parser for "harvest" command
        self.harvest_parser = self.subparsers.add_parser(
            'harvest', help='triggers search results harvest')

        # create parser for "features_dev" command
        self.features_dev_parser = self.subparsers.add_parser(
            'features_dev', help='triggers features extraction (dev)')

        # create parser for "features" command
        self.features_parser = self.subparsers.add_parser(
            'features', help='triggers features extraction')

        # create parser for "predict_dev" command
        self.predict_dev_parser = self.subparsers.add_parser(
            'predict_dev', help='triggers model predictions (dev)')

        # create parser for "train_dev" command
        self.train_dev_parser = self.subparsers.add_parser(
            'train_dev', help='triggers model training (dev)')

        # create parser for "analyze_task_demo" command
        self.analyze_task_demo_parser = self.subparsers.add_parser(
            'analyze_task_demo', help='triggers a full analysis workflow')

        # create parser for "train_task_demo" command
        self.train_task_demo_parser = self.subparsers.add_parser(
            'train_task_demo', help='triggers model training workflow')

    def run(self):
        # arguments for "search_dev" command
        self.search_dev_parser.add_argument(
            '--settings', type=str, default='search_config.yml',
            help='name of YAML configuration file containing search params')

        # arguments for "search" command
        self.search_parser.add_argument(
            '--settings', type=str, default='search_config.yml',
            help='name of YAML configuration file containing search params')

        # arguments for "harvest_dev" command
        self.harvest_dev_parser.add_argument(
            '--settings', type=str, default='harvest_config.yml',
            help='name of YAML configuration file containing harvest params')

        # arguments for "harvest" command
        self.harvest_parser.add_argument(
            '--settings', type=str, default='harvest_config.yml',
            help='name of YAML configuration file containing harvest params')

        # arguments for "features_dev" command
        self.features_dev_parser.add_argument(
            '--settings', type=str, default='features_config.yml',
            help='name of YAML configuration file containing features params')

        # arguments for "features" command
        self.features_parser.add_argument(
            '--settings', type=str, default='features_config.yml',
            help='name of YAML configuration file containing features params')

        # arguments for "predict_dev" command
        self.predict_dev_parser.add_argument(
            '--settings', type=str, default='predict_config.yml',
            help='name of YAML configuration file containing predict params')

        # arguments for "train_dev" command
        self.train_dev_parser.add_argument(
            '--settings', type=str, default='train_config.yml',
            help='name of YAML configuration file containing training params')

        # arguments for "analyze_task_demo" command
        self.analyze_task_demo_parser.add_argument(
            '--search_settings', type=str, default='search_config.yml',
            help='name of YAML configuration file containing search params')
        self.analyze_task_demo_parser.add_argument(
            '--harvest_settings', type=str, default='harvest_config.yml',
            help='name of YAML configuration file containing harvest params')
        self.analyze_task_demo_parser.add_argument(
            '--features_settings', type=str, default='features_config.yml',
            help='name of YAML configuration file containing features params')
        self.analyze_task_demo_parser.add_argument(
            '--predict_settings', type=str, default='predict_config.yml',
            help='name of YAML configuration file containing predict params')
        self.analyze_task_demo_parser.add_argument(
            '--db_settings', type=str, default='db_config.yml',
            help='name of YAML configuration file containing database params')

        # arguments for "train_task_demo" command
        self.train_task_demo_parser.add_argument(
            '--train_settings', type=str, default='train_config.yml',
            help='name of YAML configuration file containing training params')
        self.train_task_demo_parser.add_argument(
            '--db_settings', type=str, default='db_config.yml',
            help='name of YAML configuration file containing database params')

        cmd_args = self.parser.parse_args()

        if cmd_args.action == 'search_dev' and cmd_args.settings:
            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.settings)
            with open(path, 'r') as f:
                params = yaml.safe_load(f)
            se = SearchEngine(**params)
            se.run_dev()

        elif cmd_args.action == 'search' and cmd_args.settings:
            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.settings)
            with open(path, 'r') as f:
                params = yaml.safe_load(f)
            se = SearchEngine(**params)
            statements_texts = ['Τι χρήματα παίρνουν οι αιτούντες άσυλο']
            results = se.run(statements_texts)

        elif cmd_args.action == 'harvest_dev' and cmd_args.settings:
            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.settings)
            with open(path, 'r') as f:
                params = yaml.safe_load(f)
            h = Harvester(**params)
            h.run_dev()

        elif cmd_args.action == 'harvest' and cmd_args.settings:
            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.settings)
            with open(path, 'r') as f:
                params = yaml.safe_load(f)
            h = Harvester(**params)
            data = {'index': [0], 'link': ['https://www.liberal.gr/eidiseis/']}
            statements_dicts = [{
                's_id': 1, 's_text': 'Τι χρήματα παίρνουν οι αιτούντες άσυλο',
                's_resources': pd.DataFrame(data)}]
            results = h.run(statements_dicts)

        elif cmd_args.action == 'features_dev' and cmd_args.settings:
            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.settings)
            with open(path, 'r') as f:
                params = yaml.safe_load(f)
            fe = FeaturesExtractor(**params)
            fe.run_dev()

        elif cmd_args.action == 'features' and cmd_args.settings:
            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.settings)
            with open(path, 'r') as f:
                params = yaml.safe_load(f)
            fe = FeaturesExtractor(**params)
            data = {
                'title': ['title'],
                'body': ['This a the body. It contains paragraphs.'],
                'sim_par': ['A quite similar paragraph.'],
                'sim_sent': ['A very similar sentence!']}
            statement_dicts = [{
                's_id': 1, 's_text': 'Τι χρήματα παίρνουν οι αιτούντες άσυλο',
                's_resources': pd.DataFrame(data)}]
            results = fe.run(statement_dicts)

        elif cmd_args.action == 'predict_dev' and cmd_args.settings:
            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.settings)
            with open(path, 'r') as f:
                params = yaml.safe_load(f)
            p = Predictor(**params)
            p.run_dev()

        elif cmd_args.action == 'train_dev' and cmd_args.settings:
            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.settings)
            with open(path, 'r') as f:
                params = yaml.safe_load(f)
            t = Trainer(**params)
            t.run_dev()

        elif cmd_args.action == 'analyze_task_demo' \
                and cmd_args.search_settings and cmd_args.harvest_settings \
                and cmd_args.features_settings and cmd_args.db_settings:

            statement_ids = [1, 2]
            statement_texts = [
                'Τι χρήματα παίρνουν οι αιτούντες άσυλο',
                'Σάλος για ΜΚΟ που «διχοτομεί» τη Λέσβο']

            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.search_settings)
            with open(path, 'r') as f:
                search_params = yaml.safe_load(f)
            se = SearchEngine(**search_params)
            search_results = se.run(statement_texts)

            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.harvest_settings)
            with open(path, 'r') as f:
                harvest_params = yaml.safe_load(f)
            h = Harvester(**harvest_params)
            statement_dicts = [{
                's_id': statement_ids[i],
                's_text': statement_texts[i],
                's_resources': search_results[i]}
                for i in range(len(statement_texts))]
            harvest_results = h.run(statement_dicts)

            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.features_settings)
            with open(path, 'r') as f:
                features_params = yaml.safe_load(f)
            fe = FeaturesExtractor(**features_params)
            statement_dicts = [{
                's_id': statement_ids[i],
                's_text': statement_texts[i],
                's_resources': harvest_results[i]}
                for i in range(len(statement_texts))]
            features_results = fe.run(statement_dicts)

            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.predict_settings)
            with open(path, 'r') as f:
                predict_params = yaml.safe_load(f)
            p = Predictor(**predict_params)
            predict_results = p.run(features_results)

            # TODO review necessary queries
            # path = os.path.join(DirConf.CONFIG_DIR, cmd_args.db_settings)
            # with open(path, 'r') as f:
            #     db_params = yaml.safe_load(f)
            # dbh = DBHandler(**db_params)
            # for s_id, s_resources, s_features in \
            #         zip(statement_ids, harvest_results, features_results):
            #     resource_records = s_resources.to_dict('records')
            #     dbh.insert_statement_resources(s_id, resource_records)
            #     features_record = s_features
            #     dbh.insert_statement_features(s_id, features_record)

        # TODO complete train_task_demo
        elif cmd_args.action == 'train_task_demo' \
                and cmd_args.train_settings and cmd_args.db_settings:
            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.train_settings)
            with open(path, 'r') as f:
                train_params = yaml.safe_load(f)
            t = Trainer(**train_params)

            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.db_settings)
            with open(path, 'r') as f:
                db_params = yaml.safe_load(f)
            dbh = DBHandler(**db_params)
            included_feats = [
                f for k, v in t.data_params['features'].items() for f in v]
            # TODO fetch only included features. convert them to a dataframe
            #  and pass it to t.run(). Also, query statement table for the
            #  labels and pass them too in t.run()

            if not os.path.exists(DirConf.MODELS_DIR):
                os.mkdir(DirConf.MODELS_DIR)
            path = os.path.join(
                DirConf.MODELS_DIR, t.save_params['name'])
            t.save_model(path)


if __name__ == "__main__":
    interface = Interface()
    interface.run()
