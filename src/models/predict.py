from pathlib import Path
from typing import Dict, List, Tuple
from flair.models.sequence_tagger_model import SequenceTagger
from flair.data import Sentence, Token, Span
import pandas as pd

from src.utils import Split, DATA_PATH, MODELS_PATH, RESULTS_PATH, read_sentences
from src.models.string_distance_matcher import StringMatcher


def predict_subtask1():
    model_filepath = MODELS_PATH / "species.pt"
    prediction_filepath = RESULTS_PATH / "species_predictions.tsv"
    split_to_predict = Split.valid
    predict_flair_model(model_filepath, prediction_filepath, split_to_predict)


def predict_subtask2(split_to_predict=Split.valid, read_nosocomial_predictions=False):
    if not read_nosocomial_predictions:
        predict_flair_model(
            MODELS_PATH / "nosocomial.pt", RESULTS_PATH / "nosocomial_predictions.tsv", split_to_predict
        )

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

    nosocomial_df = pd.read_csv(RESULTS_PATH / "nosocomial_predictions.tsv", sep="\t")
    nosocomial_df = nosocomial_df[["filename", "off0", "off1", "label"]]

    predictions_df = predictions_df.merge(
        nosocomial_df, on=["filename", "off0", "off1"], how="left", suffixes=(None, "_nosocomial")
    )

    predictions_df["isN"] = predictions_df["label_nosocomial"] == "NOSOCOMIAL"
    predictions_df["isH"] = predictions_df["code"].str.contains("H")
    predictions_df["iscomplex"] = predictions_df["code"].str.contains("\|")
    predictions_df["NCBITax"] = predictions_df["code"]

    predictions_columns = ["filename", "mark", "label", "off0", "off1", "span", "NCBITax"]
    predictions_df = predictions_df[predictions_columns]
    predictions_df.to_csv(RESULTS_PATH / "subtask2_predictions.tsv", sep="\t", index=False)


def predict_flair_model(model_filepath: Path, prediction_filepath: Path, split_to_predict: Split):
    print("Loading model")
    trained_model: SequenceTagger = SequenceTagger.load(model_filepath)
    print("Loaded model")
    sentences = read_sentences(split_to_predict)
    print("Read sentences")
    trained_model.predict([s for s, o in sentences], label_name="predicted_ner")
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
