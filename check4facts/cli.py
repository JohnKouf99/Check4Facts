import os
import argparse

import pandas as pd
import yaml

from check4facts.scripts.search import SearchEngine
from check4facts.scripts.harvest import Harvester
from check4facts.scripts.features import FeaturesExtractor
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
            'search_dev', help='triggers relevant article search (dev)')

        # create parser for "search" command
        self.search_parser = self.subparsers.add_parser(
            'search', help='triggers relevant article search')

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

        # create parser for "demo" command
        self.demo_parser = self.subparsers.add_parser(
            'demo', help='triggers a full actions workflow')

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

        # arguments for "demo" command
        self.demo_parser.add_argument(
            '--search_settings', type=str, default='search_config.yml',
            help='name of YAML configuration file containing search params')
        self.demo_parser.add_argument(
            '--harvest_settings', type=str, default='harvest_config.yml',
            help='name of YAML configuration file containing harvest params')
        self.demo_parser.add_argument(
            '--features_settings', type=str, default='features_config.yml',
            help='name of YAML configuration file containing features params')
        self.demo_parser.add_argument(
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
            claim_texts = ['Τι χρήματα παίρνουν οι αιτούντες άσυλο']
            results = se.run(claim_texts)

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
            claim_dicts = [{
                'c_id': 1, 'c_text': 'Τι χρήματα παίρνουν οι αιτούντες άσυλο',
                'c_articles': pd.DataFrame(data)}]
            results = h.run(claim_dicts)

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
                'sim_paragraph': ['A quite similar paragraph.'],
                'sim_sentence': ['A very similar sentence!']}
            claim_dicts = [{
                'c_id': 1, 'c_text': 'Τι χρήματα παίρνουν οι αιτούντες άσυλο',
                'c_articles': pd.DataFrame(data)}]
            results = fe.run(claim_dicts)

        elif cmd_args.action == 'demo' \
                and cmd_args.search_settings and cmd_args.harvest_settings \
                and cmd_args.features_settings and cmd_args.db_settings:

            claim_ids = [1, 2]
            claim_texts = [
                'Τι χρήματα παίρνουν οι αιτούντες άσυλο',
                'Σάλος για ΜΚΟ που «διχοτομεί» τη Λέσβο']

            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.search_settings)
            with open(path, 'r') as f:
                search_params = yaml.safe_load(f)
            se = SearchEngine(**search_params)
            search_results = se.run(claim_texts)

            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.harvest_settings)
            with open(path, 'r') as f:
                harvest_params = yaml.safe_load(f)
            h = Harvester(**harvest_params)
            claim_dicts = [{'c_id': claim_ids[i], 'c_text': claim_texts[i],
                            'c_articles': search_results[i]}
                           for i in range(len(claim_texts))]
            harvest_results = h.run(claim_dicts)

            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.features_settings)
            with open(path, 'r') as f:
                features_params = yaml.safe_load(f)
            fe = FeaturesExtractor(**features_params)
            claim_dicts = [{'c_id': claim_ids[i], 'c_text': claim_texts[i],
                            'c_articles': harvest_results[i]}
                           for i in range(len(claim_texts))]
            features_results = fe.run(claim_dicts)

            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.db_settings)
            with open(path, 'r') as f:
                db_params = yaml.safe_load(f)
            dbh = DBHandler(**db_params)
            for i, c_id in enumerate(claim_ids):
                article_records = harvest_results[i].to_dict('records')
                dbh.insert_claim_articles(c_id, article_records)
                features_record = features_results[i].to_dict('records')
                dbh.insert_claim_features(c_id, features_record)


if __name__ == "__main__":
    interface = Interface()
    interface.run()
