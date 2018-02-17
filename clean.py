import matplotlib.pyplot as plt
import pandas as pd


def clean_training_data():
    df = pd.read_csv('loanfull.csv', usecols=[
        'loan_amnt',
        'term',
        'int_rate',
        'installment',
        'grade',
        'home_ownership',
        'annual_inc',
        'verification_status',
        'loan_status',
        'pymnt_plan',
        'dti',  # Debt to income ratio
        'delinq_2yrs',
        'inq_last_6mths',
        'open_acc',
        'revol_bal',
        'revol_util',
        'total_acc',
        'out_prncp',
        'out_prncp_inv',
        'total_pymnt',
        'total_pymnt_inv',
        'total_rec_prncp',
        'total_rec_int',
        'total_rec_late_fee',
        'recoveries',
        'collection_recovery_fee'
    ])
    df.fillna(0)
    # df['verification'] = df['verification'].map({'Verified': 0, 'Source Verified': 1, 'Not Verified': 2})
    # df['grade'] = df['grade'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})
    # df['home_ownership'] = df['home_ownership'].map({'RENT': 0, 'OWN': 1, 'MORTGAGE': 2})
    df['loan_status'] = df['loan_status'].map({'Charged Off': 0, 'Fully Paid': 1, 'Current': 2, 'In Grace Period': 3,
                                               'Does not meet the credit policy. Status:Fully Paid': 1,
                                               'Does not meet the credit policy. Status:Charged Off': 0,
                                               'Late (31-120 days)': 4,
                                               'Late (16-30 days)': 4,
                                               'Issued': 5,
                                               'Default': 0
                                               })
    df['term'] = df['term'].map(lambda s: int(s.split()[0]))
    df.to_csv('clean_data.csv', encoding='utf-8')


def pd_to_matplot():
    dfc = pd.read_csv('clean_data.csv')
    # plt.xlabel('Education')
    # plt.ylabel('Days past due')
    # plt.show(fig)
    fig = plt.scatter(dfc["terms"], dfc["past_due_days"])
    plt.show(fig)


clean_training_data()
# print(pd_to_matplot())
