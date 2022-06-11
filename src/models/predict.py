from html import entities
from pathlib import Path
from typing import Dict, List, Tuple
from flair.models.sequence_tagger_model import SequenceTagger
from flair.data import Sentence, Token, Span
import numpy as np
import pandas as pd

from src.utils import DOCUMENTS_PATH, Split, DATA_PATH, MODELS_PATH, RESULTS_PATH, Task, read_sentences
from src.models.string_distance_matcher import StringMatcher


def predict_subtask1(split_to_predict=Split.test_background):
    model_filepath = MODELS_PATH / "species.pt"
    prediction_filepath = RESULTS_PATH / "species_predictions.tsv"
    predict_flair_model(model_filepath, prediction_filepath, split_to_predict)


def predict_subtask2():
    code_matcher_filepath = MODELS_PATH / "codes_string_matcher.pickle"
    sm = StringMatcher.load(code_matcher_filepath)
    species_predictions = pd.read_csv(RESULTS_PATH / "species_predictions.tsv", sep="\t")
    only_humans_df = species_predictions[species_predictions["label"] == "HUMAN"]
    only_species_df = species_predictions[species_predictions["label"] == "SPECIES"]
    print("Predicting string matcher")
    predictions = sm.predict([s for s in only_species_df["span"].values])

    only_species_df["code"] = predictions
    only_humans_df["code"] = ["9606"] * only_humans_df.shape[0]
    predictions_df = pd.concat([only_humans_df, only_species_df])

    predictions_df["NCBITax"] = predictions_df["code"]

    predictions_columns = ["filename", "mark", "label", "off0", "off1", "span", "NCBITax"]
    predictions_df = predictions_df[predictions_columns]
    predictions_df.to_csv(RESULTS_PATH / "subtask2_predictions.tsv", sep="\t", index=False)


def predict_subtask3(read_entities=False, split_to_predict=Split.test_background):
    if not read_entities:
        predict_all_entities(split_to_predict=split_to_predict)
    subtask2_predictions = pd.read_csv(RESULTS_PATH / "subtask2_predictions.tsv", sep="\t")
    entities_dataframes = []
    map_entity2 = {"pet": "Pet", "animal": "AnimalInjury", "food": "Food", "nosocomial": "Nosocomial"}
    for entity_rc in ["pet", "animal", "food", "nosocomial"]:
        entity_df = pd.read_csv(RESULTS_PATH / f"{entity_rc}_predictions.tsv", sep="\t")
        entity_df = entity_df[["filename", "off0", "off1", "label"]]

        entities_df = subtask2_predictions.merge(
            entity_df, on=["filename", "off0", "off1"], suffixes=(None, f"_{entity_rc}"), how="inner"
        )
        entities_df = entities_df.groupby("filename", as_index=False).agg({f"NCBITax": "+".join})
        entities_df[f"is{map_entity2[entity_rc]}"] = [True] * entities_df.shape[0]
        entities_df[f"{map_entity2[entity_rc]}IDs"] = entities_df["NCBITax"]
        entities_df = entities_df[["filename", f"is{map_entity2[entity_rc]}", f"{map_entity2[entity_rc]}IDs"]]
        entities_dataframes.append(entities_df)

    filenames = read_filenames(split_to_predict, Task.subtask3)
    predictions_df = pd.DataFrame(data={"filename": filenames})
    for df in entities_dataframes:
        predictions_df = predictions_df.merge(df, on=["filename"], how="left")

    for entity in map_entity2.values():
        predictions_df[f"is{entity}"] = predictions_df[f"is{entity}"].replace(True, "Yes")
        predictions_df[f"is{entity}"] = predictions_df[f"is{entity}"].replace(np.nan, "No")
        predictions_df[f"{entity}IDs"] = predictions_df[f"{entity}IDs"].replace(np.nan, "NA")

    predictions_df.to_csv(RESULTS_PATH / "subtask3_predictions.tsv", sep="\t", index=False)


def read_filenames(split_to_predict: Split, subtask: Task):
    if split_to_predict == Split.valid and subtask == Task.subtask3:
        with open(
            DOCUMENTS_PATH / split_to_predict.value / Task.subtask3.value / "validation_files_task3.txt", "r"
        ) as f:
            return f.read().splitlines()
    else:
        return [file for file in (DOCUMENTS_PATH / split_to_predict.value / "text-files").iterdir()]


def predict_all_entities(split_to_predict=Split.test_background):
    for entity in ["pet", "animal", "food", "nosocomial"]:
        model_filepath = MODELS_PATH / f"{entity}.pt"
        prediction_filepath = RESULTS_PATH / f"{entity}_predictions.tsv"
        predict_flair_model(model_filepath, prediction_filepath, split_to_predict)


def predict_flair_model(model_filepath: Path, prediction_filepath: Path, split_to_predict: Split):
    print("Loading model")
    print(model_filepath)
    trained_model: SequenceTagger = SequenceTagger.load(model_filepath)
    print("Loaded model")
    sentences = read_sentences(split_to_predict)
    print("Read sentences")
    trained_model.predict([s for s, o in sentences], label_name="predicted_ner", verbose=True)
    rows = []
    for sentence, original_text in sentences:
        labels_predicted = transform_to_format(sentence, original_text)
        rows += labels_predicted
    predicted_df: pd.DataFrame = pd.DataFrame(rows, columns=["filename", "mark", "label", "off0", "off1", "span"])
    predicted_df.to_csv(prediction_filepath, sep="\t", index=False)


def transform_to_format(sentence: Sentence, original_text: str):
    spans = sentence.get_spans("predicted_ner")
    spans_predicted_transformed = get_start_and_end_pos(sentence.get_spans("predicted_ner"), original_text)
    rows = []
    for i, (span, (start_pos, end_pos)) in enumerate(zip(spans, spans_predicted_transformed)):
        row = (
            sentence.get_label("filename").value,
            f"T{i}",
            span.get_label("predicted_ner").value,
            start_pos,
            end_pos,
            original_text[start_pos:end_pos],
        )
        rows.append(row)
    return rows


def get_start_and_end_pos(spans: List[Span], original_text: str) -> Dict[Token, Tuple[int, int]]:
    start_text = 0
    start_end_pos = []
    for span in spans:
        text_to_find = span.text
        start_pos = original_text.find(text_to_find, start_text, len(original_text))
        end_pos = start_pos + len(text_to_find)
        start_text = end_pos
        if span.start_position == 0:
            start_pos = 0
        start_end_pos.append((start_pos, end_pos))
    return start_end_pos
