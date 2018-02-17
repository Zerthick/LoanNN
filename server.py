import Main
import argparse
import json
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request

##################################################
# API part
##################################################
app = Flask(__name__)


@app.route("/api/predict", methods=['POST'])
def predict():
    start = time.time()

    data = request.data.decode("utf-8")

    if data == "":
        params = request.form
        x_in = json.loads(params['x'])
    else:
        params = json.loads(data)
        x_in = params['x']

    ##################################################
    # Tensorflow part
    ##################################################
    predictions = model.predict(input_fn=eval_input_fn(x_in))

    y_out = []
    for prediction in predictions:
        y_out = prediction['class_ids'][0]

    ##################################################
    # END Tensorflow part
    ##################################################

    json_data = json.dumps({'y': np.asscalar(y_out)})
    print("Time spent handling the request: %f" % (time.time() - start))

    return json_data


##################################################
# END API part
##################################################

def mutate_dict(d):
    for k, v in d.iteritems():
        d[k] = np.array(v)


def gen_dict(x_in):
    dic = {
        "loan_amnt": x_in[0],
        "term": x_in[1],
        "int_rate": x_in[2],
        "installment": x_in[3],
        "grade": x_in[4],
        "home_ownership": x_in[5],
        "annual_inc": x_in[6],
        "verification_status": x_in[7],
        "pymnt_plan": x_in[8],
        "dti": x_in[9],
        "delinq_2yrs": x_in[10],
        "inq_last_6mths": x_in[11],
        "open_acc": x_in[12],
        "revol_bal": x_in[13],
        "revol_util": x_in[14],
        "total_acc": x_in[15],
        "out_prncp": x_in[16],
        "out_prncp_inv": x_in[17],
        "total_pymnt": x_in[18],
        "total_pymnt_inv": x_in[19],
        "total_rec_prncp": x_in[20],
        "total_rec_int": x_in[21],
        "total_rec_late_fee": x_in[22],
        "recoveries": x_in[23],
        "collection_recovery_fee": x_in[24]
    }
    return dic


def eval_input_fn(inputs):
    df = pd.DataFrame(gen_dict(inputs), index=[0])
    print(df.dtypes)
    input_fn = tf.estimator.inputs.pandas_input_fn(
        x=df,
        y=None,
        shuffle=False
    )
    return input_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        help="Frozen model file to import")
    args = parser.parse_args()

    ##################################################
    # Tensorflow part
    ##################################################
    print('Loading the model')
    model = tf.estimator.LinearClassifier(model_dir=Main.model_dir, feature_columns=Main.my_feature_columns,
                                          n_classes=Main.num_classes)
    ##################################################
    # END Tensorflow part
    ##################################################

    print('Starting the API')
app.run()
