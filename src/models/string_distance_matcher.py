import pickle
from pathlib import Path
from typing import Dict, List

from rapidfuzz.distance import Levenshtein


class StringMatcher:
    def __init__(self, sentence_label_map: Dict[str, str] = {}):
        self.sentence_label_map: Dict[str, str] = sentence_label_map

    @classmethod
    def load(cls, model_path: Path):
        print("Loading model ...")
        with open(model_path, "rb") as f:
            sentence_label_map = pickle.load(f)
        return cls(sentence_label_map)

    def fit(self, dictionary_definitions: Dict[str, str], sentences_to_predict: List[str] = []):
        print("Fitting definitions")
        for definition, label in dictionary_definitions.items():
            self.sentence_label_map[definition] = label
        if sentences_to_predict:
            self.predict(sentences_to_predict)

    def predict(self, sentences: List[str]):
        print("Predicting labels of sentences")
        predicted_labels = []
        for i, sentence in enumerate(sentences):
            if i % 100 == 0:
                print(f"{i}/{len(sentences)}")
            if not self.sentence_label_map.get(sentence):
                min_dist = 10000
                min_label = None
                for definition, label in self.sentence_label_map.items():
                    dist = Levenshtein.distance(str(definition), str(sentence), score_cutoff=min_dist + 1)
                    if dist < min_dist:
                        min_dist = dist
                        min_label = label
                self.sentence_label_map[sentence] = min_label
            predicted_labels.append(self.sentence_label_map[sentence])
        return predicted_labels

    def save(self, model_path: Path):
        print("Saving model")
        with open(model_path, "wb") as f:
            pickle.dump(self.sentence_label_map, f)
