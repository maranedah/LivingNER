import os
from typing import List
import numpy as np
import pandas as pd
from src.utils import MODELS_PATH, DATA_PATH, DOCUMENTS_PATH, RESULTS_PATH, Split, read_sentences
from src.models.string_distance_matcher import StringMatcher


def train_subtask2_codes(splits_to_use_for_training=[Split.training, Split.valid]):
    dictionary_definitions = get_dictionary_definitions()
    dataset_definitions = get_dataset_definitions(splits_to_use_for_training)
    sentences = np.concatenate((dictionary_definitions[0], dataset_definitions[0]))
    labels = np.concatenate((dictionary_definitions[1], dataset_definitions[1]))
    dictionary = {sentence: label for sentence, label in zip(sentences, labels)}

    species_predictions = pd.read_csv(RESULTS_PATH / "species_predictions.tsv", sep="\t")
    only_species_df = species_predictions[species_predictions["label"] == "SPECIES"]
    sentences_to_predict = [s for s in only_species_df["span"].values]

    code_model = StringMatcher()
    code_model.fit(dictionary, sentences_to_predict)
    model_path = MODELS_PATH / "codes_string_matcher.pickle"
    code_model.save(model_path)


def get_dictionary_definitions():
    dictionary = pd.read_csv(
        os.path.join(DATA_PATH, "ncbi-taxo-names-spanish_v2.dmp"), sep="\t", on_bad_lines="skip", header=None
    )
    dictionary = dictionary.rename(columns={0: "code", 4: "span"})
    dictionary = dictionary.drop_duplicates(subset=["code", "span"])
    return dictionary["span"].values, dictionary["code"].astype(str).values


def get_dataset_definitions(split_types: List[Split]):
    df = pd.DataFrame()
    for split_type in split_types:
        mapx = {Split.training: "training", Split.valid: "validation"}
        entities = pd.read_csv(
            os.path.join(
                DOCUMENTS_PATH, split_type.value, "subtask2-Norm", f"{mapx[split_type]}_entities_subtask2.tsv"
            ),
            sep="\t",
        )
        entities = entities.drop_duplicates(subset=["code", "span"])
        df = pd.concat([df, entities])
    return df["span"].values, df["code"].values
