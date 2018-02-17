import tempfile

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def gen_data():
    df = pd.read_csv('credit_train.csv', usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], header=0)
    df = df.dropna()

    df.columns = [c.replace(' ', '_') for c in df.columns]
    df['Loan_Status'] = df['Loan_Status'].map({'Fully Paid': 1, 'Charged Off': 0})

    train_data, test_data = train_test_split(df, test_size=0.4)

    min = len(train_data.query('Loan_Status == 0'))

    train_data_sub_sampled = pd.concat(
        [train_data.query('Loan_Status == 1').sample(min), train_data.query('Loan_Status == 0')])

    train_label = train_data_sub_sampled['Loan_Status']
    train_data = train_data_sub_sampled.drop('Loan_Status', 1)

    test_label = test_data['Loan_Status']
    test_data = test_data.drop('Loan_Status', 1)

    return train_data, train_label, test_data, test_label


def get_input_fn(x_in, y_in):
    input_fn = tf.estimator.inputs.pandas_input_fn(
        x=x_in,
        y=y_in,
        shuffle=False
    )
    return input_fn


def main(argv):
    x_train, y_train, x_test, y_test = gen_data()
    my_feature_columns = [tf.feature_column.numeric_column('Current_Loan_Amount'),
                          tf.feature_column.indicator_column(
                              tf.feature_column.categorical_column_with_hash_bucket('Term', hash_bucket_size=10)),
                          tf.feature_column.numeric_column('Credit_Score'),
                          tf.feature_column.numeric_column('Annual_Income'),
                          tf.feature_column.indicator_column(
                              tf.feature_column.categorical_column_with_hash_bucket('Years_in_current_job',
                                                                                    hash_bucket_size=10)),
                          tf.feature_column.indicator_column(
                              tf.feature_column.categorical_column_with_hash_bucket('Home_Ownership',
                                                                                    hash_bucket_size=10)),
                          tf.feature_column.indicator_column(
                              tf.feature_column.categorical_column_with_hash_bucket('Purpose', hash_bucket_size=10)),
                          tf.feature_column.numeric_column('Monthly_Debt'),
                          tf.feature_column.numeric_column('Years_of_Credit_History'),
                          tf.feature_column.numeric_column('Months_since_last_delinquent'),
                          tf.feature_column.numeric_column('Number_of_Open_Accounts'),
                          tf.feature_column.numeric_column('Number_of_Credit_Problems'),
                          tf.feature_column.numeric_column('Current_Credit_Balance'),
                          tf.feature_column.numeric_column('Maximum_Open_Credit'),
                          tf.feature_column.numeric_column('Bankruptcies'),
                          tf.feature_column.numeric_column('Tax_Liens')]

    model_dir = tempfile.mkdtemp()

    model = tf.estimator.DNNClassifier(model_dir=model_dir, feature_columns=my_feature_columns, hidden_units=[128, 64],
                                       n_classes=2)

    model.train(input_fn=get_input_fn(x_train, y_train), steps=1000)

    results = model.evaluate(input_fn=get_input_fn(x_test, y_test), steps=None)

    print(results)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main, argv=None)
