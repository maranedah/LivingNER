from enum import Enum
from pathlib import Path
from typing import List, Tuple
from flair.data import Sentence


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data"
DOCUMENTS_PATH = DATA_PATH
MODELS_PATH = PROJECT_DIR / "models"
RESULTS_PATH = PROJECT_DIR / "results"

subtask1_evaluation_command = "python main.py -g ../../LivingNER/data/valid/subtask1-NER/validation_entities_subtask1.tsv -p ../../LivingNER/results/species_predictions.tsv -s ner"
subtask2_evaluation_command = "python main.py -g ../../LivingNER/data/valid/subtask2-Norm/evaluation.tsv -p ../../LivingNER/results/subtask2_predictions.tsv -s norm"


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
