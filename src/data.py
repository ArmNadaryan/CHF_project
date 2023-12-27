import pandas as pd


def import_data(file_path):
    """Loading the data and dropping some columns"""
    df = pd.read_csv(file_path)
    df = df.drop(columns=['id', 'author'])
    return df
