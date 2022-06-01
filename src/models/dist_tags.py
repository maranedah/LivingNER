from pathlib import Path
import os 
import pandas as pd
import numpy as np
from tqdm import tqdm
import Levenshtein 
from rapidfuzz.distance import Levenshtein
from time import time
import itertools

pd.set_option('display.max_columns', 500)

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = os.path.join(PROJECT_DIR, "data")
EXTERNAL_DIR = os.path.join(PROJECT_DIR, "data", "external")
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
INTERIM_DIR = os.path.join(PROJECT_DIR, "data", "interim")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")

def get_dictionary_definitions():
	
	dictionary = pd.read_csv(os.path.join(RAW_DIR, "ncbi-taxo-names-spanish_v2.dmp"), sep="\t", on_bad_lines="skip", header=None)
	dictionary = dictionary.rename(columns={0:"code", 4:"span"})
	print(dictionary.shape)
	dictionary = dictionary.drop_duplicates(subset=["code", "span"])
	print(dictionary.shape)
	df = dictionary.groupby("code")["span"].apply(list).reset_index()
	definitions, codes = df["span"].values, df["code"].astype(str).values 
	return definitions, codes

def get_train_definitions():
	entities = pd.read_csv(os.path.join(RAW_DIR, "training", "subtask2-Norm", "training_entities_subtask2.tsv"), sep="\t")
	print(entities.shape)
	entities = entities.drop_duplicates(subset=["code", "span"])
	print(entities.shape)
	entities = entities.groupby("code")["span"].apply(list).reset_index()
	return entities["span"].values, entities["code"].values


def get_entities_definitions():
	entities = pd.read_csv(os.path.join(RAW_DIR, "valid", "subtask2-Norm", "validation_entities_subtask2.tsv"), sep="\t")
	return entities["span"].values, entities["code"].values

def calc_score():
	train_definitions, train_definition_codes = get_train_definitions()
	print("se cargo el diccionario de train")
	entities, entities_codes = get_entities_definitions()
	print("se cargaron los spans en validacion")

	
	definitions, definition_codes = get_dictionary_definitions()
	print("se cargo el diccionario")
	
	definitions = np.concatenate((train_definitions, definitions), axis=0)
	definition_codes = np.concatenate((train_definition_codes,definition_codes), axis=0)
	
	acc = 0
	for entity, entity_code in zip(entities, entities_codes): #entity = span en validation
		min_dist = 10000
		min_tag = None
		for definition_list, definition_code in zip(definitions, definition_codes): #definition = span en dictionario
			#print(definition_code, definition_list)
			for definition in definition_list: 
				dist = Levenshtein.distance(str(entity), str(definition), score_cutoff=min_dist+1)
				if dist < min_dist and definition_code:
					min_dist = dist
					min_tag = definition_code

		print(min_tag, entity_code)
		if str(min_tag) == str(entity_code):
			acc+=1
		
	acc = acc/len(entities) 
	print("Acc", acc)

start_time = time()
calc_score()
end_time = time()
print(end_time - start_time)