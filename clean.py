import pandas as pd

def get_clean_data():
    df = pd.read_csv('credit_train.csv', usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    df = df.dropna()
    df.to_csv('clean_credit_train.csv', encoding='utf-8')
    return df

get_clean_data()
