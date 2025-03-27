import os
import pandas as pd

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

KEPT_COLS = [
    "prob_desc_time_limit",
    "prob_desc_notes",
    "prob_desc_description",
    "prob_desc_output_spec",
    "prob_desc_input_spec",
    "lang",
    "source_code",
]


def extract_fields_from_jsons(data_path) -> list:
    result_list = []
    all_samples_names = os.listdir(data_path)

    for sample_path in all_samples_names:
        sample = pd.read_json(data_path + sample_path)
        sample_tags = list(sample["tags"])
        if len(sample) > 0:
            sample_dict_to_append = {col: sample.loc[0, col] for col in KEPT_COLS}
            sample_dict_to_append["filename"] = sample_path

        for sample_tag in sample_tags:
            if sample_tag in KEPT_TAGS:
                sample_dict_to_append[sample_tag] = 1
        result_list.append(sample_dict_to_append)
    return result_list


def clean_dataset(codeforce_dataset: pd.DataFrame) -> pd.DataFrame:
    tags_in_dataset = []
    for tag in KEPT_TAGS:
        if tag in codeforce_dataset.columns:
            tags_in_dataset.append(tag)
            codeforce_dataset.loc[codeforce_dataset[tag].isna(), tag] = 0

    codeforce_dataset = codeforce_dataset.loc[
        codeforce_dataset[tags_in_dataset].sum(axis=1) > 0
    ].reset_index(drop=True)  # keep problems with enough examples
    codeforce_dataset["prob_desc_description"] = codeforce_dataset[
        "prob_desc_description"
    ].str.lower()
    codeforce_dataset["prob_desc_notes"] = codeforce_dataset[
        "prob_desc_description"
    ].str.lower()
    codeforce_dataset["prob_desc_output_spec"] = codeforce_dataset[
        "prob_desc_description"
    ].str.lower()
    codeforce_dataset["prob_desc_input_spec"] = codeforce_dataset[
        "prob_desc_description"
    ].str.lower()
    codeforce_dataset = codeforce_dataset.drop_duplicates().reset_index(drop=True)
    return codeforce_dataset


def save(
    codeforce_dataset: pd.DataFrame, save_path="data/codeforce_dataset.csv"
) -> None:
    codeforce_dataset.to_csv(save_path)


if __name__ == "__main__":
    DATA_PATH = "data/raw_data/"
    codeforce_dataset = pd.DataFrame.from_dict(
        extract_fields_from_jsons(), orient="columns"
    )
    codeforce_dataset = clean_dataset(codeforce_dataset)
    save(codeforce_dataset)
