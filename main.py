from src.models.predict import predict_subtask1, predict_subtask2
from src.data.make_dataset import download_dataset
from src.models.train import train_subtask2_codes

if __name__ == "__main__":
    predict_subtask2(read_nosocomial_predictions=True)
