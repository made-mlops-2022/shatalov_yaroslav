import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

TARGET_COLUMN = 'condition'

def _make_model() -> None:
    # TODO: implement selection of models 
    cls = LogisticRegression()
    return cls

def fit_cls(train_df_file_path: str, train_y_file_path: str, cls_file_path: str):
    cls = _make_model()
    train_df = pd.read_csv(train_df_file_path)
    train_y = pd.read_csv(train_y_file_path)
    cls.fit(train_df, train_y[TARGET_COLUMN].values)
    pickle.dump(cls, open(cls_file_path, 'wb')) 

def make_predictions_with_cls(val_df_file_path: str, cls_file_path: str, res_predictions_file_path: str) -> list:
    val_df = pd.read_csv(val_df_file_path)
    cls = pickle.load(open(cls_file_path, 'rb'))
    predictions = pd.DataFrame(cls.predict(val_df))
    predictions.to_csv(res_predictions_file_path, index=False)


