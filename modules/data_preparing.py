import pandas as pd
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL_COLUMNS = [
        'thal', 
        'ca',
        'slope',
        'restecg',
        'cp',
    ]

REAL_COLUMNS = [
        'oldpeak',
        'thalach',
        'chol',
        'trestbps',
        'age',
    ]

def _make_dataframe_from_file(df_file_path: str) -> pd.DataFrame:
    return pd.read_csv(df_file_path)

    
def make_train_val_dataset(df_file_path: str) -> tuple:
    df = _make_dataframe_from_file(df_file_path)

    encoded_part = _make_encoded_part(df)
    
    df.drop(columns=CATEGORICAL_COLUMNS, inplace=True)
    df = _scale_df(df.join(encoded_part))
    df = _drop_outliers(df)
    
    df.columns = list(str(i) for i in range(df.shape[1] - 1)) + ['condition']
    df = df.sample(frac=1)

    train_ind = int(df.shape[0] * 0.90)
    train_df = df.iloc[:train_ind, :].copy()
    val_df = df.iloc[train_ind:, :].copy()

    train_y = train_df['condition'].copy()
    train_df.drop(columns=['condition'], inplace=True)
    val_y = val_df['condition'].copy()
    val_df.drop(columns=['condition'], inplace=True)

    return train_df, train_y, val_df, val_y


def _make_encoded_part(df: pd.DataFrame) -> pd.DataFrame:
    enc = OneHotEncoder()
    encoded_part = df.loc[:, CATEGORICAL_COLUMNS].copy()
    enc.fit(encoded_part)
    return pd.DataFrame(enc.transform(encoded_part).toarray())


def _scale_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in REAL_COLUMNS:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df


def _drop_outliers(df: pd.DataFrame) -> pd.DataFrame:
    return df
    # TODO: make dropping for bigger datasets
    # upper_lim = train_df[col].quantile(0.95)
    # lower_lim = train_df[col].quantile(0.05)
    # train_df = train_df[(train_df[col] < upper_lim) & (train_df[col] > lower_lim)]