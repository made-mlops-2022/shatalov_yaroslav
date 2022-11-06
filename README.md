## Dataset taken from:
https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci<br>
Dataset csv file is in the "data" folder

## 1. Create train/test datasets from original csv file
To create datasets from raw csv the command `make-datasets` is used.<br>
At the command prompt type:<br>
`python main_pipeline.py make-datasets`<br>

You can specify raw file path and/or output folder with flags `--df-file-path` and `--out-folder`. For example:<br>
`python main_pipeline.py make-datasets --df-file-path data\\heart_cleveland_upload.csv --out-folder data`<br>

As a result you will get four files: train dataframe, train target, validation dataframe and validation target<br>

## 2. Fit the model
In this iteration of module only Logistic Regression is used as a model.<br>
To fit the model use `fit-model command` as such:<br>
`python main_pipeline.py fit-model`<br>

You can provide specific dataframe and target paths as well as path to save fitted model.<br>
To do so you can use flags `--train-df-file-path`, `--train-y-file-path` and `--cls-file-path`:<br>
`python main_pipeline.py fit-model --train-df-file-path data\\train_df.csv --train-y-file-path data\\train_y.csv --cls-file-path models\\cls.sav`<br>

## 3. Make predictions
To make predictions from validation file use `make-predictions` command:<br>
`python main_pipeline.py make-predictions`<br>

You can specify validation csv file path, fitted model file path and output file path:<br>
`python main_pipeline.py make-predictions --val-df-file-path data\\val_df.csv --cls-file-path models\\cls.sav --res-predictions-file data\\predictions.csv`<br>

## Folder organization
`├── README.md                      <- README file with instructions`<br>
`├── data                            <- Folder for datasets and predictions`<br>
`│   ├── heart_cleveland_upload.csv  <- Original used dataset`<br>
`│`<br>
`├── logs                            <- Folder to save logs by default`<br>
`│`<br>
`├── models                          <- Folder to save models by default`<br>
`|`<br>
`|modules                            <- Modules folder contains used in piplene .py files`<br>
`│   ├── __init__.py`<br>
`│   ├── data_preparing.py           <- Scripts to process dataset`<br>
`│   ├── models.py                   <- Scripts to process models`<br>
`|`<br>
`├── notebooks                       <- Jupyter notebooks`<br>
`│   ├── EDA.ipynb                   <- Analysis of data and model tryouts`<br>
`│`<br>
`├── requirements.txt                <- The requirements file for reproducing the environment`<br>
`│`<br>
`├── main_pipeline.py                <- Main script for working through terminal`<br>
