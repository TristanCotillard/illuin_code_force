import argparse
import pandas as pd

from src.data_processing import extract_fields_from_jsons, clean_dataset, save
from src.model_training_and_evaluation import (
    prepare_codeforce_train_test_datasets,
    train_all_models,
)
from src.predict import PredictionTools

KEPT_TAGS = [
    "math",
    "graphs",
    "strings",
    "number theory",
    "trees",
    "geometry",
    "games",
    "probabilities",
]
PREDICTIVE_COLS = [
    "prob_desc_description",
    "filename",
]  # filename used only to look at erros after prediction


def preprocess_data(data_path: str, save_path: str = "data/codeforce_dataset.csv"):
    codeforce_dataset = pd.DataFrame.from_dict(
        extract_fields_from_jsons(data_path), orient="columns"
    )
    codeforce_dataset = clean_dataset(codeforce_dataset)
    save(codeforce_dataset, save_path)


def train_and_eval_models(codeforce_dataset_path: str):
    codeforce_dataset = pd.read_csv(codeforce_dataset_path)

    for tag in KEPT_TAGS:
        if tag not in codeforce_dataset.columns:
            codeforce_dataset[tag] = 0.0
            print(
                f"tag {tag} has been detected in the input dataset, please provide it for better prediction"
            )

    codeforce_dataset = codeforce_dataset[[*PREDICTIVE_COLS, *KEPT_TAGS]]
    labels = codeforce_dataset[KEPT_TAGS].to_numpy()

    codeforce_training_dataset, codeforce_testing_dataset = (
        prepare_codeforce_train_test_datasets(codeforce_dataset, labels)
    )
    train_all_models(codeforce_training_dataset)

    prediction_tools = PredictionTools()
    prediction_tools.eval(
        codeforce_testing_dataset[["prob_desc_description"]],
        codeforce_testing_dataset[KEPT_TAGS],
    )


def predict(dataset_path: str):
    codeforce_dataset = pd.read_csv(dataset_path)

    prediction_tools = PredictionTools()
    predicted_tags = prediction_tools.predict(
        codeforce_dataset[["prob_desc_description"]],
    )
    return predicted_tags


def evaluate(codeforce_dataset_path: str = None):
    codeforce_dataset = pd.read_csv(codeforce_dataset_path)

    prediction_tools = PredictionTools()
    prediction_tools.eval(
        codeforce_dataset[["prob_desc_description"]],
        codeforce_dataset[KEPT_TAGS],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ML Model Prediction and Evaluation for tags attached to programming problems"
    )
    parser.add_argument(
        "--preprocess_data",
        nargs=2,
        metavar=("dataset_path", "save_path"),
        help="Preprocess samples in dataset_path and create working csv with extracted code description and tags. Default value fixed to data/raw_data/",
    )
    parser.add_argument(
        "--train",
        nargs=1,
        metavar="dataset_path",
        help="Train the tag all prediction models with default parameters on a preprocessed dataset",
    )
    parser.add_argument(
        "--eval",
        nargs=1,
        metavar="dataset_path",
        help="Eval tag prediction models on a specific preprocessed dataset",
    )
    parser.add_argument(
        "--predict",
        nargs=1,
        metavar="dataset_path",
        help="Predict all tags on a specific preprocessed dataset",
    )

    args = parser.parse_args()

    if args.preprocess_data:
        dataset_path = (
            args.preprocess_data[0]
            if isinstance(args.preprocess_data[0], str)
            else "data/raw_data/"
        )
        save_path = (
            args.preprocess_data[1]
            if isinstance(args.preprocess_data[1], str)
            else "data/codeforce_dataset.csv"
        )
        print(args.preprocess_data, dataset_path, save_path)
        preprocess_data(dataset_path, save_path)

    if args.train:
        dataset_path = (
            args.train
            if isinstance(args.train, str)
            else "data/cleaned_codeforce_dataset.csv"
        )
        train_and_eval_models(dataset_path)
        print("Tag prediction models trained and saved")

    if args.predict:
        dataset_path = (
            args.predict
            if isinstance(args.predict, str)
            else "data/codeforce_dataset.csv"
        )
        result = predict(dataset_path)
        print("Tag predicted")
        print(result)

    if args.eval:
        dataset_path = (
            args.eval if isinstance(args.eval, str) else "data/codeforce_dataset.csv"
        )
        evaluate(dataset_path)
