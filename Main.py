import tempfile

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def gen_data():
    df = pd.read_csv('clean_data.csv', usecols=[1, 2, 3, 4, 5, 6, 7], header=0)

    train_data, test_data = train_test_split(df, test_size=0.4)

    train_label = train_data['loan_status']
    train_data = train_data.drop('loan_status', 1)

    test_label = test_data['loan_status']
    test_data = test_data.drop('loan_status', 1)

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
    my_feature_columns = [tf.feature_column.numeric_column('Principal'),
                          tf.feature_column.numeric_column('terms'),
                          tf.feature_column.numeric_column('past_due_days'),
                          tf.feature_column.numeric_column('age'),
                          tf.feature_column.categorical_column_with_hash_bucket('education', hash_bucket_size=5),
                          tf.feature_column.categorical_column_with_hash_bucket('Gender', hash_bucket_size=2)]

    model_dir = tempfile.mkdtemp()

    model = tf.estimator.LinearClassifier(model_dir=model_dir, feature_columns=my_feature_columns,
                                          n_classes=3)

    model.train(input_fn=get_input_fn(x_train, y_train), steps=1)

    results = model.evaluate(input_fn=get_input_fn(x_test, y_test), steps=None)

    print(results)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main, argv=None)
