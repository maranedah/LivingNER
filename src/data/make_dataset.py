# -*- coding: utf-8 -*-
import os
import logging
import shutil
import wget
from pathlib import Path

from src.utils import DATA_PATH


def download_data():
    data_url = "https://zenodo.org/record/6421410/files/training_valid.zip?download=1"
    wget.download(data_url, str(DATA_PATH))
    output_file = f"{DATA_PATH}/training_valid.zip"
    shutil.unpack_archive(output_file, extract_dir=DATA_PATH)
    os.remove(output_file)


def download_vocab():
    vocab_url = "https://zenodo.org/record/6451758/files/ncbi-taxo-names-spanish_v2.dmp?download=1"
    wget.download(vocab_url, str(DATA_PATH))


def download_dataset():
    print("Downloading data")
    download_data()
    print("Downloading vocabulary")
    download_vocab()
