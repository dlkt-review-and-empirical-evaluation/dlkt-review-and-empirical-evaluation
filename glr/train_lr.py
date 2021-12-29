import argparse
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as acc, roc_auc_score as auc, f1_score as f1, matthews_corrcoef as mc, \
    mean_squared_error as mse, precision_score as prec, recall_score as recall
from sklearn.model_selection import KFold
from pathlib import Path

def compute_metrics(y_pred, y):
    bin_pred = np.round(y_pred)
    results = {}
    results['acc'] = acc(y, bin_pred)
    results['auc'] = auc(y, y_pred)
    results['prec'] = prec(y, bin_pred)
    results['recall'] = recall(y, bin_pred)
    results['f1'] = f1(y, bin_pred)
    results['mcc'] = mc(y, bin_pred)
    results['rmse'] = np.sqrt(mse(y, y_pred))
    # nll = log_loss(y, y_pred)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train logistic regression on sparse feature matrix.')
    parser.add_argument('--X_file', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--iter', type=int, default=10000)
    args = parser.parse_args()

    features_suffix = (args.X_file.split("-")[-1]).split(".")[0]

    # Load sparse dataset
    X = csr_matrix(load_npz(args.X_file))

    data = pd.read_csv(f'../data/preprocessed/{args.dataset}/preprocessed_data.csv', sep="\t")
    kfold = KFold(n_splits=5)
    results = []
    users = data["user_id"].unique()


    for i, (train_i, test_i) in enumerate(kfold.split(users)):
        print(f"fold {i + 1}")
        # Train-test split
        train_df = data[data["user_id"].isin(users[train_i])]
        test_df = data[data["user_id"].isin(users[test_i])]

        # Student-wise train-test split
        user_ids = X[:, 0].toarray().flatten()
        users_train = train_df["user_id"].unique()
        users_test = test_df["user_id"].unique()
        train = X[np.where(np.isin(user_ids, users_train))]
        test = X[np.where(np.isin(user_ids, users_test))]

        # First 5 columns are the original dataset, including label in column 3
        X_train, y_train = train[:, 5:], train[:, 3].toarray().flatten()
        X_test, y_test = test[:, 5:], test[:, 3].toarray().flatten()

        # Train
        model = LogisticRegression(solver="lbfgs", max_iter=args.iter)
        model.fit(X_train, y_train)

        y_pred_train = model.predict_proba(X_train)[:, 1]
        y_pred_test = model.predict_proba(X_test)[:, 1]

        # Write predictions to csv
        # test_df[f"LR_{features_suffix}"] = y_pred_test
        # print('write')
        # test_df.to_csv(f'data/{args.dataset}/preprocessed_data_test.csv', sep="\t", index=False)

        train_results = compute_metrics(y_pred_train, y_train)
        test_results = compute_metrics(y_pred_test, y_test)
        results.append(test_results)

        print(f"kfold iteration {i + 1}: {args.dataset}, features = {features_suffix}, test results = {test_results}")

    Path('results').mkdir(exist_ok=True)
    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv(f'results/5-fold-lrbest-{args.dataset}.csv', index=False)
