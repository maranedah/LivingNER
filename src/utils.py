import os
from enum import Enum
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv
from flair.data import Sentence

load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" if not os.getenv("DATA_PATH") else Path(os.getenv("DATA_PATH"))
DOCUMENTS_PATH = (
    DATA_PATH / "training_valid_test_background_multilingual"
    if not os.getenv("DOCUMENTS_PATH")
    else Path(os.getenv("DOCUMENTS_PATH"))
)
MODELS_PATH = PROJECT_DIR / "models" if not os.getenv("MODELS_PATH") else Path(os.getenv("MODELS_PATH"))
RESULTS_PATH = PROJECT_DIR / "results" if not os.getenv("RESULTS_PATH") else Path(os.getenv("RESULTS_PATH"))
MINI_BATCH_SIZE = 32 if not os.getenv("MINI_BATCH_SIZE") else int(os.getenv("MINI_BATCH_SIZE"))

subtask1_evaluation_command = "python main.py -g ../../LivingNER/data/valid/subtask1-NER/validation_entities_subtask1.tsv -p ../../LivingNER/results/species_predictions.tsv -s ner"
subtask2_evaluation_command = "python main.py -g ../../LivingNER/data/valid/subtask2-Norm/evaluation.tsv -p ../../LivingNER/results/subtask2_predictions.tsv -s norm"
subtask3_evaluation_command = "python main.py -g ../../LivingNER/data/valid/subtask3-Clinical-Impact/validation_subtask3.tsv -p ../../LivingNER/results/subtask3_predictions.tsv -s app"


class Split(Enum):
    valid = "valid"
    training = "training"
    test_background = "test_background"


class Task(Enum):
    subtask1 = "subtask1-NER"
    subtask2 = "subtask2-Norm"
    subtask3 = "subtask3-Clinical-Impact"


def read_sentences(split: Split) -> List[Tuple[Sentence, str]]:
    text_files_path: Path = DOCUMENTS_PATH / split.value / "text-files"
    sentences = []
    for file in text_files_path.iterdir():
        sentence = read_document_sentence(file)
        sentences.append(sentence)
    return sentences


def read_document_sentence(file: Path) -> Tuple[Sentence, str]:
    with open(file, "r", encoding="utf-8") as txt:
        original_txt = txt.read()
        sentence = Sentence(original_txt)
        sentence.add_label("filename", file.name.split(".")[0])
        return sentence, original_txt
