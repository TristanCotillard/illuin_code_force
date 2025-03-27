# illuin_code_force

## Problem Statement

TBD

## Commands
```python
usage: main.py [-h] [--preprocess_data dataset_path save_path] [--train dataset_path] [--eval dataset_path] [--predict dataset_path]

ML Model Prediction and Evaluation for tags attached to programming problems

options:
  -h, --help            show this help message and exit
  --preprocess_data dataset_path save_path
                        Preprocess samples in dataset_path and create working csv with extracted code description and tags. Default value fixed to data/raw_data/
  --train dataset_path  Train the tag all prediction models with default parameters on a preprocessed dataset
  --eval dataset_path   Eval tag prediction models on a specific preprocessed dataset
  --predict dataset_path
                        Predict all tags on a specific preprocessed dataset
```

```python
python main.py --preprocess_data data/raw_data/  data/codeforce_dataset_bis.csv
python main.py --train data/codeforce_dataset.csv
python main.py --eval data/codeforce_dataset.csv  
python main.py --predict data/codeforce_dataset.csv
```

Note that training may not work without enough data furnished(at least at few examples per tag).
