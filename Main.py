import pandas as pd
import tensorflow as tf


def input_fn():


def main():
    df = pd.read_csv('credit_train.csv', usecols=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    l = pd.read_csv('credit_train.csv', usecols=[2])

    my_feature_columns = [tf.feature_column.numeric_column('loan_amt'),
                          tf.feature_column.categorical_column_with_hash_bucket('term'),
                          tf.feature_column.numeric_column('credit_score'),
                          tf.feature_column.numeric_column('income'),
                          tf.feature_column.categorical_column_with_hash_bucket('job_years'),
                          tf.feature_column.categorical_column_with_hash_bucket('home_ownership'),
                          tf.feature_column.categorical_column_with_hash_bucket('purpose'),
                          tf.feature_column.numeric_column('monthly_debt'),
                          tf.feature_column.numeric_column('credit_history'),
                          tf.feature_column.numeric_column('delinquent_months'),
                          tf.feature_column.numeric_column('num_accounts'),
                          tf.feature_column.numeric_column('credit_probs'),
                          tf.feature_column.numeric_column('credit_bal'),
                          tf.feature_column.numeric_column('max_credit'),
                          tf.feature_column.numeric_column('bankruptcies'),
                          tf.feature_column.numeric_column('tax_liens')]

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 3 classes.
        n_classes=3)

    classifier.train(input_fn=lambda input_fn(df.values, 100, True, 100))


if __name__ == '__main__':
    tf.app.run(main=main, argv=None)
