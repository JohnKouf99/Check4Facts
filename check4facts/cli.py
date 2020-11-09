import os
import argparse

import pandas as pd
import yaml

from check4facts.scripts.search import SearchEngine
from check4facts.scripts.harvest import Harvester
from check4facts.scripts.features import FeaturesExtractor
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

        # create parser for "search_harvest" command
        self.search_harvest_parser = self.subparsers.add_parser(
            'search_harvest', help='triggers search and harvest actions')

        # create parser for "features" command
        self.features_parser = self.subparsers.add_parser(
            'features', help='triggers features extraction')

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

        # arguments for "search_harvest" command
        self.search_harvest_parser.add_argument(
            '--search_settings', type=str, default='search_config.yml',
            help='name of YAML configuration file containing search params')
        self.search_harvest_parser.add_argument(
            '--harvest_settings', type=str, default='harvest_config.yml',
            help='name of YAML configuration file containing harvest params')

        # arguments for "features" command
        self.features_parser.add_argument(
            '--settings', type=str, default='features_config.yml',
            help='name of YAML configuration file containing features params')

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
            claims = ['Τι χρήματα παίρνουν οι αιτούντες άσυλο']
            results = se.run(claims)

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
            # NOTE: Filter out non-html articles before passing them to h.run()
            data = {'index': [0], 'link': ['https://www.liberal.gr/eidiseis/']}
            articles = [{
                'c_id': '98j34r',
                'c_text': 'Τι χρήματα παίρνουν οι αιτούντες άσυλο',
                'articles': pd.DataFrame(data)}]
            results = h.run(articles)

        elif cmd_args.action == 'search_harvest' \
                and cmd_args.search_settings and cmd_args.harvest_settings:
            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.search_settings)
            with open(path, 'r') as f:
                search_params = yaml.safe_load(f)
            se = SearchEngine(**search_params)
            claims = ['Τι χρήματα παίρνουν οι αιτούντες άσυλο']
            search_results = se.run(claims)
            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.harvest_settings)
            with open(path, 'r') as f:
                harvest_params = yaml.safe_load(f)
            h = Harvester(**harvest_params)
            articles = [{
                'c_id': '98j34r',
                'c_text': claims[0],
                'articles': search_results[0]}]
            harvest_results = h.run(articles)

        elif cmd_args.action == 'features' and cmd_args.settings:
            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.settings)
            with open(path, 'r') as f:
                params = yaml.safe_load(f)
            fe = FeaturesExtractor(**params)
            fe.run()


if __name__ == "__main__":
    interface = Interface()
    interface.run()
