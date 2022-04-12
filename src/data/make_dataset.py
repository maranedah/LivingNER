# -*- coding: utf-8 -*-
import os
import logging
import requests
import shutil
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = os.path.join(PROJECT_DIR, "data")
EXTERNAL_DIR = os.path.join(PROJECT_DIR, "data", "external")
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
INTERIM_DIR = os.path.join(PROJECT_DIR, "data", "interim")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")

def download_data():

    data_url = "https://zenodo.org/record/6421410/files/training_valid.zip?download=1"
    r = requests.get(data_url)

    OUTPUT_FILE = os.path.join(RAW_DIR, "data.zip")
    OUTPUT_DIR = RAW_DIR
    open(OUTPUT_FILE, 'wb').write(r.content)
    if not os.path.exists(OUTPUT_DIR): os.mkdir(OUTPUT_DIR) 
    
    shutil.unpack_archive(OUTPUT_FILE, extract_dir=OUTPUT_DIR)
    os.remove(OUTPUT_FILE)

def download_vocab():
    vocab_url = "https://zenodo.org/record/6451758/files/ncbi-taxo-names-spanish_v2.dmp?download=1"
    
    OUTPUT_FILE = os.path.join(RAW_DIR, "ncbi-taxo-names-spanish_v2.dmp")
    
    with requests.get(vocab_url, stream=True) as r:
        r.raise_for_status()
        with open(OUTPUT_FILE, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


    
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Downloading data')
    download_data()
    logger.info('Data downloaded')
    
    logger.info('Downloading vocabulary')
    download_vocab()
    logger.info('Vocab downloaded')
    
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
