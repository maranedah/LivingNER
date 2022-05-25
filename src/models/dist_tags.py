from pathlib import Path
import os 
import pandas as pd
import numpy as np
from tqdm import tqdm
from levenshtein import lev_dist

pd.set_option('display.max_columns', 500)

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = os.path.join(PROJECT_DIR, "data")
EXTERNAL_DIR = os.path.join(PROJECT_DIR, "data", "external")
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
INTERIM_DIR = os.path.join(PROJECT_DIR, "data", "interim")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")

def get_dictionary_definitions():
	
	dictionary = pd.read_csv(os.path.join(RAW_DIR, "ncbi-taxo-names-spanish_v2.dmp"), sep="\t", on_bad_lines="skip", header=None)
	df = dictionary.groupby(0)[4].apply(list).reset_index(name='new')
	definitions, codes = df["new"].values, df[0].values 
	definitions = [list(x) for x in set(tuple(x) for x in definitions)]
	codes = [str(x) for x in codes]
	return definitions, codes

def get_train_definitions():
	entities = pd.read_csv(os.path.join(RAW_DIR, "training", "subtask2-Norm", "training_entities_subtask2.tsv"), sep="\t").sample(100)
	return entities["span"].values, entities["code"].values


def get_entities_definitions():
	entities = pd.read_csv(os.path.join(RAW_DIR, "valid", "subtask2-Norm", "validation_entities_subtask2.tsv"), sep="\t").sample(100)
	return entities["span"].values, entities["code"].values

def calc_score():
	definitions, definition_codes = get_dictionary_definitions()
	entities, entities_codes = get_entities_definitions()
	acc = 0
	for entity, entity_code in zip(entities, entities_codes):
		min_dist = 10000
		min_tag = None
		for definition_list, definition_code in zip(definitions, definition_codes):
			for definition in definition_list: 
				dist = lev_dist(str(entity), str(definition))
				print(len(entities), len(definitions), len(definition_list))
				if dist < min_dist:
					min_dist = lev_dist(definition, entity)
					min_tag = definition_code

		print(min_tag, entity_code)
		if str(min_tag) == str(entity_code):
			acc+=1

	acc = acc/len(entities) 
	print(acc)

calc_score()