This folder contains all needed scripts to run the project.
It is composed of the following files:

- **data_processing.py**: details how the dataset *data/cleaned_codeforce_dataset.csv* was obtained from the raw_problems. Illustrated with a subsample of problems that gives the dataset *subsample_cleaned_codeforce_dataset.csv*
- **model_training_and_evaluation.py**: details the training, train/test split, features extraction functions
- **predict.py**: details how the prediction pipeline (in particular as the order is very important)