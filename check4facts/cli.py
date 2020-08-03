import os
import argparse

import yaml

from check4facts.config import DirConf
from check4facts.search import CustomSearchEngine


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

        # create parser for "search" command
        self.search_parser = self.subparsers.add_parser(
            'search', help='triggers relevant article search')

    def run(self):
        # arguments for "search" command
        self.search_parser.add_argument(
            '--settings', type=str, default='search_config.yml',
            help='name of YAML configuration file containing search params')

        cmd_args = self.parser.parse_args()

        if cmd_args.action == 'search' and cmd_args.settings:
            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.settings)
            with open(path, 'r') as f:
                params = yaml.safe_load(f)

            cse = CustomSearchEngine(**params)
            cse.run()


if __name__ == "__main__":
    interface = Interface()
    interface.run()
