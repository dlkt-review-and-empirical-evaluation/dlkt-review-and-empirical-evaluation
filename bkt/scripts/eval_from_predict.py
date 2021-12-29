import argparse
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score as auc, matthews_corrcoef as mcc, accuracy_score as acc, f1_score as f1, \
    mean_squared_error as mse, precision_score as prec, recall_score as recall



def evaluate(corrects, predictions):
    print('acc:', acc(corrects, np.round(predictions)))
    print('auc:', auc(corrects, predictions))
    print('prec:', prec(corrects, np.round(predictions)))
    print('recall:', recall(corrects, np.round(predictions)))
    print('f1:', f1(corrects, np.round(predictions)))
    print('mcc:', mcc(corrects, np.round(predictions)))
    print('rmse', np.sqrt(mse(corrects, predictions)))
    return acc(corrects, np.round(predictions)), \
           auc(corrects, predictions), \
           prec(corrects, np.round(predictions)), \
           recall(corrects, np.round(predictions)), \
           f1(corrects, np.round(predictions)), \
           mcc(corrects, np.round(predictions)), \
           np.sqrt(mse(corrects, predictions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('predictions')
    parser.add_argument('test_file')
    parser.add_argument('res_file')

    args = parser.parse_args()
    predictions = pd.read_csv(args.predictions, sep='\t', header=None)
    predictions.columns = ['correct', 'incorrect']

    test_data = pd.read_csv(args.test_file, sep='\t', header=None)
    corrects = -(test_data.iloc[:, 0] - 2)
    predictions = predictions.correct

    results = evaluate(corrects, predictions)

    if args.res_file is not None:
        if not os.path.isfile(args.res_file):
            with open(args.res_file, 'w') as f:
                f.write('acc,auc,prec,recall,f1,mcc,rmse\n')
        with open(args.res_file, 'a') as f:
            f.write(','.join([str(x) for x in results])+'\n')
