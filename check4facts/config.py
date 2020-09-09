import os


class DirConf:
    DATA_DIR = 'data'
    SEARCH_RESULTS_DIR = os.path.join(DATA_DIR, 'search_results')
    HARVEST_RESULTS_DIR = os.path.join(DATA_DIR, 'harvest_results')
    HARVEST_XML_DIR = os.path.join(HARVEST_RESULTS_DIR, 'xml_files')
    CONFIG_DIR = 'config'
    MODELS_DIR = 'models'
