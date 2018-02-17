import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import Main


def main(argv):
    model = tf.estimator.LinearClassifier(model_dir=Main.model_dir, feature_columns=Main.my_feature_columns,
                                          n_classes=Main.num_classes)
    print(model.get_variable_names())
    weights = {
        'annual_inc': np.asscalar(model.get_variable_value('linear/linear_model/annual_inc/weights')[0]),
        'collection_recovery_fee': np.asscalar(
            model.get_variable_value('linear/linear_model/collection_recovery_fee/weights')[0]),
        'delinq_2yrs': np.asscalar(model.get_variable_value('linear/linear_model/delinq_2yrs/weights')[0]),
        'dti': np.asscalar(model.get_variable_value('linear/linear_model/dti/weights')[0]),
        'grade': np.asscalar(model.get_variable_value('linear/linear_model/grade/weights')[0]),
        'home_ownership': np.asscalar(model.get_variable_value('linear/linear_model/home_ownership/weights')[0]),
        'inq_last_6mths': np.asscalar(model.get_variable_value('linear/linear_model/inq_last_6mths/weights')[0]),
        'installment': np.asscalar(model.get_variable_value('linear/linear_model/installment/weights')[0]),
        'int_rate': np.asscalar(model.get_variable_value('linear/linear_model/int_rate/weights')[0]),
        'loan_amnt': np.asscalar(model.get_variable_value('linear/linear_model/loan_amnt/weights')[0]),
        'open_acc': np.asscalar(model.get_variable_value('linear/linear_model/open_acc/weights')[0]),
        'out_prncp': np.asscalar(model.get_variable_value('linear/linear_model/out_prncp/weights')[0]),
        'out_prncp_inv': np.asscalar(model.get_variable_value('linear/linear_model/out_prncp_inv/weights')[0]),
        'pymnt_plan': np.asscalar(model.get_variable_value('linear/linear_model/pymnt_plan/weights')[0]),
        'recoveries': np.asscalar(model.get_variable_value('linear/linear_model/recoveries/weights')[0]),
        'revol_bal': np.asscalar(model.get_variable_value('linear/linear_model/revol_bal/weights')[0]),
        'revol_util': np.asscalar(model.get_variable_value('linear/linear_model/revol_util/weights')[0]),
        'term': np.asscalar(model.get_variable_value('linear/linear_model/term/weights')[0]),
        'total_acc': np.asscalar(model.get_variable_value('linear/linear_model/total_acc/weights')[0]),
        'total_pymnt': np.asscalar(model.get_variable_value('linear/linear_model/total_pymnt/weights')[0]),
        'total_pymnt_inv': np.asscalar(model.get_variable_value('linear/linear_model/total_pymnt_inv/weights')[0]),
        'total_rec_int': np.asscalar(model.get_variable_value('linear/linear_model/total_rec_int/weights')[0]),
        'total_rec_late_fee': np.asscalar(
            model.get_variable_value('linear/linear_model/total_rec_late_fee/weights')[0]),
        'total_rec_prncp': np.asscalar(model.get_variable_value('linear/linear_model/total_rec_prncp/weights')[0]),
        'verification_status': np.asscalar(
            model.get_variable_value('linear/linear_model/verification_status/weights')[0]),
    }
    barlist = plt.bar(weights.keys(), weights.values(), width=0.75, color='r')
    plt.xticks(rotation=90)

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.2)
    barlist[2].set_color('b')
    barlist[17].set_color('b')
    barlist[19].set_color('b')
    barlist[23].set_color('b')
    plt.show()


if __name__ == '__main__':
    tf.app.run(main=main, argv=None)
