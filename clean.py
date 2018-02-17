import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def clean_training_data():
    df = pd.read_csv('Loan payments data.csv', usecols=[1, 2, 3,  7, 8, 9, 10])
    df['loan_status'] = df['loan_status'].map({'PAIDOFF': 0, 'COLLECTION': 1, 'COLLECTION_PAIDOFF': 2})
    df['past_due_days'] = df['past_due_days'].apply(lambda x: x if x > 0 else 0)
    df = df.dropna()
    df.to_csv('clean_data.csv', encoding='utf-8')


def pd_to_np():
    dfc = pd.read_csv('clean_data.csv')
    # df_asarray = dfc.values
    # fig = plt.scatter(df_asarray[10], df_asarray[4])
    # plt.set_xlabel('Credit History(yrs)')
    # plt.set_ylabel('Credit Score')
    # fig = plt.scatter(dfc["Bankruptcies"], dfc["Credit Score"], s=1)
    # plt.xlabel("Bankruptcies")
    # plt.ylabel("Credit Score")
    # # fig.set_xlabel('Credit History(yrs)')
    # # fig.set_ylabel('Credit Score')
    # plt.show(fig)


clean_training_data()
print(pd_to_np())