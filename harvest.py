import logging
import http.client
import urllib.error
from os import listdir
from tika import parser
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
import csv, sys, requests, re, json
from os.path import abspath, isfile, isdir, join

USAGE = """
<harvest.py> takes exactly one argument. This argument must be a directory containing .csv files or a single .csv file

    directory: Traverses the directory (one level only) and searches for .csv files and havrvests each one.
    file: Harvests the given .csv file.

Unexpected behaviour if file(s) are not .csv of format
"""

# Colors for prettier printing to console.
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def harvest_from_csv(filename: str) :
    # HTML tags to ignore.
    blacklist = [
        'meta',
        'head', 
        'input',
        'noscript',
        'script',
        'style',
        'form',
        'select',
        'header',
        'footer'
    ]

    with open(filename, 'r', newline='') as csvfile:
        input_filename = filename.split('/')[-1]
        output_filename = f"harvest_{input_filename}"
        dirName = output_filename.split('.')[0]
        Path(f'{dirName}').mkdir(exist_ok=True)

        reader = csv.DictReader(csvfile)

        print(f'{bcolors.HEADER}Started harvesting links from file: {input_filename}{bcolors.ENDC}')
        with open(f'{dirName}/{output_filename}', 'w', newline='') as output_file :

            output_fieldNames = ['link', 'content']
            writer = csv.DictWriter(output_file, fieldnames=output_fieldNames)
            writer.writeheader()

            row_counter = 1;
            for idx, row in enumerate(reader):
                row_counter+=1 # Starting row count from 2 to match with the csv rows.

                # Link is HTML page
                if ('fileFormat' not in row or not row['fileFormat']):
                    try:
                        res = requests.get(row['link'])
                        html_page = res.content
                        soup = BeautifulSoup(html_page, 'lxml')
                        
                        # Discard every tag that is in our blacklist. s.decompose() can be replaced with s.extract() that does not delete the extracted tags.
                        [s.decompose() for bl in blacklist for s in soup.select(bl)]

                        allText = soup.findAll(text=True)

                        # Store the HTML page after blacklisting to a .xml file.
                        with open(f"{dirName}/{dirName}_{row_counter}.xml", 'w') as xml_file:
                            xml_file.write(soup.prettify())

                        output_text = ''
                        for t in allText:
                            # Further blacklisting for nested unwanted tags.
                            if t.parent.name not in ['[document]', 'style']:
                                output_text += f'{t} '

                        # Uncomment line below to remove newlines and tabs
                        # output_text = " ".join(output_text.split())
                        writer.writerow({'link': row['link'], 'content': output_text})
                        print(f'\tText from link: {row["link"]} persisted on {output_filename}.')
                    except requests.exceptions.HTTPError as errh:
                        logging.error(f'Http Error: {errh} on link {row["link"]}')
                    except requests.exceptions.ConnectionError as errc:
                        logging.error(f'Error Connecting: {errc} on link {row["link"]}')
                    except requests.exceptions.Timeout as errt:
                        logging.error(f'Timeout Error: {errt} on link {row["link"]}')
                    except requests.exceptions.RequestException as err:
                        logging.error(f'Ooops: Something Else: {err} on link {row["link"]}')

                # Link is a file (.pdf, .doc/docx, .txt, etc.)
                # The file crawling implementation has some issues that are listed on Trello.
                else:
                    try:
                        raw = parser.from_file(row['link'])
                        output_text = " ".join(raw['content'].split()) if raw['content'] != None else 'No text from parse'
                        writer.writerow({'link': row['link'], 'content': output_text})
                        print(f'\tText from file {row["link"]} persisted on {output_filename}.')
                    except urllib.error.HTTPError as errh:
                        logging.error(f'Tika Http Error: {errh} on link {row["link"]}')
                    except urllib.error.URLError as erru:
                        logging.error(f'Tika URL Error: {erru} on link {row["link"]}')
                    except urllib.error.ContentTooShortError as errc:
                        logging.error(f'Tika Content Too Short Error: {errc} on link {row["link"]}')
                    except http.client.RemoteDisconnected as errd:
                        logging.error(f'Tika Remote Disconnected Error: {errd} on link {row["link"]}')
            print(f'{bcolors.OKGREEN}Finished harvesting links from file: {input_filename}')

if __name__ == "__main__":
    if len(sys.argv) == 2:
        arg = sys.argv[1]
        Path('logs').mkdir(exist_ok=True)

        now = datetime.now()
        logging.basicConfig(filename=f'logs/{now.strftime("%d_%m_%Y_%H_%M_%S")}.log', level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

        if isdir(arg):
            mypath = abspath(arg)
            input_files = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
            for input_file in input_files :
                harvest_from_csv(input_file)
        elif isfile(arg):
            input_file = abspath(arg)
            harvest_from_csv(input_file)
        else:
            print(USAGE)
    else:
        print(USAGE)