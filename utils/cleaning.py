import pandas as pd

def get_null_columns(df):
    return df[df.columns[df.isnull().any()]]

def fill_missing_values(df, method="mean"):
    df_filled = df.copy()
    for col in df_filled.columns:
        if df_filled[col].isnull().sum() > 0:
            if df_filled[col].dtype in ['float64', 'int64']:
                if method == "mean":
                    df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
                elif method == "median":
                    df_filled[col] = df_filled[col].fillna(df_filled[col].median())
            else:
                if method == "mode":
                    df_filled[col] = df_filled[col].fillna(df_filled[col].mode()[0])
    return df_filled

def drop_missing(df):
    return df.dropna()

def drop_duplicates(df):
    return df.drop_duplicates()

def change_dtype(df, col, new_type):
    try:
        return df.astype({col: new_type})
    except Exception as e:
        return str(e)
