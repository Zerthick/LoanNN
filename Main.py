import tempfile

import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def gen_data():
    df = pd.read_csv('clean_data.csv', header=0)
    df = df.dropna()

    train_data, test_data = train_test_split(df, test_size=.4)

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
    my_feature_columns = [tf.feature_column.numeric_column('loan_amnt'),
                          tf.feature_column.numeric_column('term'),
                          tf.feature_column.numeric_column('int_rate'),
                          tf.feature_column.numeric_column('installment'),
                          tf.feature_column.categorical_column_with_hash_bucket('grade', hash_bucket_size=26),
                          tf.feature_column.categorical_column_with_hash_bucket('home_ownership', hash_bucket_size=10),
                          tf.feature_column.numeric_column('annual_inc'),
                          tf.feature_column.categorical_column_with_hash_bucket('verification_status',
                                                                                hash_bucket_size=10),
                          tf.feature_column.categorical_column_with_hash_bucket('pymnt_plan', hash_bucket_size=10),
                          tf.feature_column.numeric_column('dti'),
                          tf.feature_column.numeric_column('delinq_2yrs'),
                          tf.feature_column.numeric_column('inq_last_6mths'),
                          tf.feature_column.numeric_column('open_acc'),
                          tf.feature_column.numeric_column('revol_bal'),
                          tf.feature_column.numeric_column('revol_util'),
                          tf.feature_column.numeric_column('total_acc'),
                          tf.feature_column.numeric_column('out_prncp'),
                          tf.feature_column.numeric_column('out_prncp_inv'),
                          tf.feature_column.numeric_column('total_pymnt'),
                          tf.feature_column.numeric_column('total_pymnt_inv'),
                          tf.feature_column.numeric_column('total_rec_prncp'),
                          tf.feature_column.numeric_column('total_rec_int'),
                          tf.feature_column.numeric_column('total_rec_late_fee'),
                          tf.feature_column.numeric_column('recoveries'),
                          tf.feature_column.numeric_column('collection_recovery_fee')]

    model_dir = tempfile.mkdtemp()

    model = tf.estimator.LinearClassifier(model_dir=model_dir, feature_columns=my_feature_columns, n_classes=6)

    model.train(input_fn=get_input_fn(x_train, y_train), steps=5000)

    results = model.evaluate(input_fn=get_input_fn(x_test, y_test), steps=None)

    print(results)

    predictions = model.predict(input_fn=get_input_fn(x_test, y_test))

    predicted_labels = []

    for prediction in predictions:
        predicted_labels.append(prediction['class_ids'][0])

    print(confusion_matrix(list(y_test), predicted_labels))

    feature_spec = tf.feature_column.make_parse_example_spec(my_feature_columns)
    export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

    model.export_savedmodel(export_dir_base='ouput', serving_input_receiver_fn=export_input_fn)


if __name__ == '__main__':
    tf.app.run(main=main, argv=None)
