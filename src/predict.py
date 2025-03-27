import joblib
from sklearn.metrics import precision_score, recall_score


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


def make_prediction_on_zeros(
    dataset, tag_to_consider_zero, model, feature_column="prob_desc_description"
):
    final_prediction = dataset[tag_to_consider_zero].copy()

    dataset_to_predict_on = dataset[dataset[tag_to_consider_zero] == 0]
    prediction_on_zero = model.predict(dataset_to_predict_on[feature_column])
    final_prediction.loc[dataset[tag_to_consider_zero] == 0] = prediction_on_zero
    final_prediction.loc[dataset[tag_to_consider_zero] == 1] = 0
    return final_prediction


class PredictionTools:
    def __init__(self):
        self.string_model = joblib.load("src/trained_models/strings_model.sav")
        self.nb_theory_model = joblib.load("src/trained_models/nb_theory_model.sav")
        self.graph_or_tree_model = joblib.load(
            "src/trained_models/graph_or_tree_model.sav"
        )
        self.geometry_model = joblib.load("src/trained_models/geometry_model.sav")
        self.games_model = joblib.load("src/trained_models/games_model.sav")
        self.proba_model = joblib.load("src/trained_models/probabilities_model.sav")
        self.math_model = joblib.load("src/trained_models/math_model.sav")

    def predict(self, dataset, feature_column="prob_desc_description"):
        dataset["strings"] = self.string_model.predict(dataset[feature_column])
        dataset["number theory"] = make_prediction_on_zeros(
            dataset, tag_to_consider_zero="strings", model=self.nb_theory_model
        )
        dataset["graphs"] = make_prediction_on_zeros(
            dataset, tag_to_consider_zero="strings", model=self.graph_or_tree_model
        )
        dataset["trees"] = make_prediction_on_zeros(
            dataset, tag_to_consider_zero="strings", model=self.graph_or_tree_model
        )
        dataset["geometry"] = make_prediction_on_zeros(
            dataset, tag_to_consider_zero="number theory", model=self.geometry_model
        )
        dataset["games"] = make_prediction_on_zeros(
            dataset, tag_to_consider_zero="geometry", model=self.games_model
        )
        dataset["probabilities"] = make_prediction_on_zeros(
            dataset, tag_to_consider_zero="geometry", model=self.proba_model
        )
        dataset["math"] = self.math_model.predict(dataset[feature_column])
        return dataset

    def predict_without_simplification_logic(
        self, dataset, feature_column="prob_desc_description"
    ):
        dataset["strings"] = self.string_model.predict(dataset[feature_column])
        dataset["number theory"] = self.nb_theory_model.predict(dataset[feature_column])
        dataset["graphs"] = self.graph_or_tree_model.predict(dataset[feature_column])
        dataset["trees"] = self.graph_or_tree_model.predict(dataset[feature_column])
        dataset["geometry"] = self.geometry_model.predict(dataset[feature_column])
        dataset["games"] = self.games_model.predict(dataset[feature_column])
        dataset["probabilities"] = self.proba_model.predict(dataset[feature_column])
        dataset["math"] = self.math_model.predict(dataset[feature_column])
        return dataset

    def eval(self, dataset, labels, with_logic=True):
        if with_logic:
            prediction = self.predict(dataset)
        else:
            prediction = self.predict_without_simplification_logic(dataset)
        for tag in KEPT_TAGS:
            precision = precision_score(prediction[tag], labels[tag])
            recall = recall_score(prediction[tag], labels[tag])
            f1_score = 0
            if precision + recall > 0:
                f1_score = 2 * recall * precision / (recall + precision)
            print(
                f"{tag}: recall={recall:.2f}, precision={precision:.2f}, f1={f1_score:.2f}"
            )
