import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score
import warnings

from src.predict import PredictionTools

warnings.filterwarnings("ignore")


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


def build_test_set_equal_prop_per_tag(Y):
    extracted_indices = set()
    for rank_tag in range(len(KEPT_TAGS)):
        nb_sample_with_tag = sum(Y[:, rank_tag])
        idx_tag = np.where(Y[:, rank_tag] == 1)[0]
        extracted_indices.update(
            np.random.choice(
                idx_tag, size=int(nb_sample_with_tag * 0.2), replace=False
            ).tolist()
        )
    return np.array(list(extracted_indices))


def prepare_codeforce_train_test_datasets(codeforce_dataset, labels):
    test_index = build_test_set_equal_prop_per_tag(labels)
    print(
        f"proportion of {len(test_index) / len(codeforce_dataset)} kept for testing models"
    )
    codeforce_training_dataset = codeforce_dataset[
        ~codeforce_dataset.index.isin(test_index)
    ]
    codeforce_training_dataset = codeforce_training_dataset.reset_index(drop=True)
    codeforce_testing_dataset = codeforce_dataset[
        codeforce_dataset.index.isin(test_index)
    ]
    codeforce_testing_dataset = codeforce_testing_dataset.reset_index(drop=True)
    print(
        f"{len(codeforce_training_dataset)} remaining samples in the training dataset"
    )
    print(f"{len(codeforce_testing_dataset)} samples in the testing dataset")
    return codeforce_training_dataset, codeforce_testing_dataset


def evaluate_model_and_get_best_hyperparameters(
    dataset,
    tag_to_evaluate,
    max_features_to_test,
    model=RandomForestClassifier(n_estimators=200, max_depth=5),
    max_df=1.0,
    min_df=5,
):
    all_corpus = dataset["prob_desc_description"]
    Y_binarized_tag = dataset[tag_to_evaluate]
    best_max_features = None
    best_score = (0, 0, 0)

    for max_feature in max_features_to_test:
        tdf_string_vectorizer = TfidfVectorizer(
            max_df=max_df,
            min_df=min_df,
            max_features=max_feature,
            ngram_range=(1, 3),
            stop_words="english",
        )

        skfold = StratifiedKFold(n_splits=5, shuffle=True)

        precisions = []
        recalls = []
        f1_scores = []

        for train_index, test_index in skfold.split(
            X=np.zeros(len(Y_binarized_tag)), y=Y_binarized_tag
        ):
            training_corpus = all_corpus[
                all_corpus.index.isin(train_index) & (dataset[tag_to_evaluate] == 1)
            ]
            tdf_string_vectorizer.fit(training_corpus)
            X_tdf_string = tdf_string_vectorizer.transform(all_corpus)

            X_test = X_tdf_string[test_index]
            Y_test = Y_binarized_tag[test_index]
            X_train = X_tdf_string[train_index]
            Y_train = Y_binarized_tag[train_index]

            model.fit(X_train, Y_train)
            Y_prediction = model.predict(X_test)
            precision = precision_score(Y_test, Y_prediction)
            recall = recall_score(Y_test, Y_prediction)
            if precision + recall > 0:
                f1_score = 2 * precision * recall / (recall + precision)
            else:
                f1_score = 0
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)

        if np.mean(f1_scores) > best_score[2]:
            best_max_features = max_feature
            best_score = np.mean(precisions), np.mean(recalls), np.mean(f1_scores)

    print(
        f"\nPerformance for tag {tag_to_evaluate}:\n precision: {best_score[0]}, recall: {best_score[1]}, f1_score: {best_score[2]}"
    )

    return best_max_features


def train_one_model_on_tf_idf_features(
    dataset,
    tag_to_train,
    best_max_feature=50,
    max_df=1.0,
    min_df=5,
):
    # Train the final full model of string prediction
    all_corpus = dataset["prob_desc_description"]
    training_corpus = dataset.loc[dataset[tag_to_train] == 1, "prob_desc_description"]
    Y_binarized_tag = dataset[tag_to_train]

    tdf_tag_vectorizer = TfidfVectorizer(
        max_df=max_df,
        min_df=min_df,
        max_features=best_max_feature,
        ngram_range=(1, 3),
        stop_words="english",
    )
    tdf_tag_vectorizer.fit(training_corpus)
    X_tdf_tag = tdf_tag_vectorizer.transform(all_corpus)

    model = RandomForestClassifier(n_estimators=200, max_depth=5)
    model.fit(X_tdf_tag, Y_binarized_tag)
    return make_pipeline(tdf_tag_vectorizer, model)


def train_all_models(codeforce_training_dataset):
    # Train string model
    string_best_max_features = evaluate_model_and_get_best_hyperparameters(
        codeforce_training_dataset, "strings", [5, 20, 50, 100, 150, 200, 250, 300]
    )
    string_model = train_one_model_on_tf_idf_features(
        codeforce_training_dataset, "strings", best_max_feature=string_best_max_features
    )

    # Train number theory model without strings problems
    codeforce_training_dataset_no_string = codeforce_training_dataset[
        codeforce_training_dataset["strings"] == 0
    ].reset_index(drop=True)
    nb_theory_best_max_features = evaluate_model_and_get_best_hyperparameters(
        codeforce_training_dataset_no_string,
        "number theory",
        [5, 20, 50, 100, 150, 200, 250, 300],
    )
    nb_theory_model = train_one_model_on_tf_idf_features(
        codeforce_training_dataset_no_string,
        "number theory",
        best_max_feature=nb_theory_best_max_features,
    )

    # Train geometry model without strings nor number theory problems
    codeforce_training_dataset_no_string_no_nb_theory = codeforce_training_dataset[
        (codeforce_training_dataset["strings"] == 0)
        & (codeforce_training_dataset["number theory"] == 0)
    ].reset_index(drop=True)
    geometry_best_max_features = evaluate_model_and_get_best_hyperparameters(
        codeforce_training_dataset_no_string_no_nb_theory,
        "geometry",
        [5, 20, 50, 100, 150, 200, 250, 300],
    )
    geometry_model = train_one_model_on_tf_idf_features(
        codeforce_training_dataset_no_string_no_nb_theory,
        "geometry",
        best_max_feature=geometry_best_max_features,
    )

    # Train graph or tree model
    codeforce_training_dataset_no_string_with_graph_or_tree_label = (
        codeforce_training_dataset_no_string.copy()
    )
    codeforce_training_dataset_no_string_with_graph_or_tree_label["graph_or_tree"] = (
        codeforce_training_dataset_no_string_with_graph_or_tree_label["graphs"]
        + codeforce_training_dataset_no_string_with_graph_or_tree_label["trees"]
    )
    codeforce_training_dataset_no_string_with_graph_or_tree_label["graph_or_tree"] = (
        np.minimum(
            codeforce_training_dataset_no_string_with_graph_or_tree_label[
                "graph_or_tree"
            ],
            1,
        )
    )
    graph_tree_best_max_features = evaluate_model_and_get_best_hyperparameters(
        codeforce_training_dataset_no_string_with_graph_or_tree_label,
        "graph_or_tree",
        [5, 20, 50, 100, 150, 200, 250, 300],
    )
    graph_or_tree_model = train_one_model_on_tf_idf_features(
        codeforce_training_dataset_no_string_with_graph_or_tree_label,
        "graph_or_tree",
        best_max_feature=graph_tree_best_max_features,
    )

    # Train games model without strings, number theory and geometry problems
    codeforce_training_dataset_no_string_no_nb_theory_no_geometry = (
        codeforce_training_dataset[
            (codeforce_training_dataset["strings"] == 0)
            & (codeforce_training_dataset["number theory"] == 0)
            & (codeforce_training_dataset["geometry"] == 0)
        ].reset_index(drop=True)
    )
    games_best_max_features = evaluate_model_and_get_best_hyperparameters(
        codeforce_training_dataset_no_string_no_nb_theory_no_geometry,
        "games",
        [5, 20, 50, 100, 150, 200, 250, 300],
    )
    games_model = train_one_model_on_tf_idf_features(
        codeforce_training_dataset_no_string_no_nb_theory_no_geometry,
        "games",
        best_max_feature=games_best_max_features,
    )

    # Train probabilities model without strings, number theory and geometry problems
    proba_best_max_features = evaluate_model_and_get_best_hyperparameters(
        codeforce_training_dataset_no_string_no_nb_theory_no_geometry,
        "probabilities",
        [5, 20, 50, 100, 150, 200, 250, 300],
    )
    proba_model = train_one_model_on_tf_idf_features(
        codeforce_training_dataset_no_string_no_nb_theory_no_geometry,
        "probabilities",
        best_max_feature=proba_best_max_features,
    )

    # Train math model on all problems
    math_best_max_features = evaluate_model_and_get_best_hyperparameters(
        codeforce_training_dataset, "math", [100, 200, 300, 400], max_df=0.8, min_df=10
    )
    math_model = train_one_model_on_tf_idf_features(
        codeforce_training_dataset, "math", best_max_feature=math_best_max_features
    )

    joblib.dump(string_model, "src/trained_models/strings_model.sav")
    joblib.dump(nb_theory_model, "src/trained_models/nb_theory_model.sav")
    joblib.dump(geometry_model, "src/trained_models/geometry_model.sav")
    joblib.dump(graph_or_tree_model, "src/trained_models/graph_or_tree_model.sav")
    joblib.dump(games_model, "src/trained_models/games_model.sav")
    joblib.dump(proba_model, "src/trained_models/probabilities_model.sav")
    joblib.dump(math_model, "src/trained_models/math_model.sav")


if __name__ == "__main__":
    codeforce_dataset = pd.read_csv("data/cleaned_codeforce_dataset.csv")
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
