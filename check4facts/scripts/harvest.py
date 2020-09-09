import os
import glob
import csv
import urllib.error

import requests
from bs4 import BeautifulSoup
from tika import parser

from check4facts.config import DirConf


class Harvester:

    def __init__(self, **kwargs):
        self.basic_params = kwargs['basic']
        self.html_params = kwargs['html']

    def harvest(self, row):
        # Link to html
        if 'fileFormat' not in row or not row['fileFormat']:
            # Replace with requests.utils.unquote(row['link'])
            # to encode link with greek characters
            content = self.harvest_html(
                row['statement_id'], row['index'], row['link'])
        # Link to PDF or Word
        else:
            content = self.harvest_file(row['link'])
        return content

    def harvest_html(self, statement_id, link_id, link):
        try:
            page = requests.get(link)
            soup = BeautifulSoup(page.content, self.html_params['parser'])

            # Save raw page content as .xml file
            output_file = os.path.join(
                DirConf.HARVEST_XML_DIR,
                '{}_{}.xml'.format(statement_id, link_id))
            with open(output_file, 'w') as output_xml:
                output_xml.write(soup.prettify())

            # Remove blacklisted tags
            for tag in self.html_params['blacklist']:
                for match in soup.select(tag):
                    match.decompose()

            return soup.get_text(' ', strip=True)
        except requests.exceptions.RequestException as e:
            print(type(e), '::', e)
            return None

    @staticmethod
    def harvest_file(link):
        try:
            parsed_file = parser.from_file(link)
            content = parsed_file['content']
            return ' '.join(content.split()) if content else None
        except urllib.error.URLError as e:
            print(type(e), '::', e)
            return None

    def process_file(self, input_file):
        with open(input_file, newline='') as input_csv:
            reader = csv.DictReader(input_csv)

            output_file = os.path.join(
                DirConf.HARVEST_RESULTS_DIR,
                '{}.csv'.format(os.path.basename(input_file).split('.')[0]))

            with open(output_file, 'w', newline='') as output_csv:
                fieldnames = ['id', 'link', 'content']
                writer = csv.DictWriter(output_csv, fieldnames=fieldnames)
                writer.writeheader()

                for row in reader:
                    print('Harvesting link:', row['index'], row['link'])
                    writer.writerow({
                        'id': row['index'], 'link': row['link'],
                        'content': self.harvest(row)})
        return

    def run(self):
        # Folder to store harvest results
        if not os.path.exists(DirConf.HARVEST_RESULTS_DIR):
            os.mkdir(DirConf.HARVEST_RESULTS_DIR)

        # Sub-folder to store the original html files
        if not os.path.exists(DirConf.HARVEST_XML_DIR):
            os.mkdir(DirConf.HARVEST_XML_DIR)

        # All .csv files in given path
        path = os.path.join(DirConf.DATA_DIR, self.basic_params['dir_name'])
        files = glob.glob(os.path.join(path, '*.csv'))

        for file in files:
            print('Processing file:', file)
            self.process_file(file)
        return
