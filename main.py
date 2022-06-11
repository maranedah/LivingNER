from src.models.predict import predict_all_entities, predict_subtask1, predict_subtask2, predict_subtask3
from src.data.make_dataset import download_dataset
from src.models.train import train_subtask2_codes

if __name__ == "__main__":
    predict_subtask1()
    predict_subtask2()
    predict_subtask3()
