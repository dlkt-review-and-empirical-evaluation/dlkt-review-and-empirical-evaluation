import os
import pandas as pd


def print_best(result_dir):
    res = pd.DataFrame()
    for v in os.listdir(result_dir):
        v_dir = os.path.join(result_dir, v)
        for dataset in os.listdir(v_dir):
            res_file_path = os.path.join(v_dir, dataset, '5fold_results.csv')
            if os.path.isfile(res_file_path):
                tmp_res = pd.read_csv(res_file_path)
                tmp_res["solver"] = v
                tmp_res["dataset"] = dataset
                res = pd.concat([res, tmp_res], ignore_index=True)

        if len(res) == 0:
            print(f'no kfold results for version {v}')
            continue

    avg_res = res.groupby(["dataset", "solver"]).mean().reset_index(level=["solver", "dataset"])
    best_res = avg_res.loc[avg_res.groupby(['dataset']).auc.idxmax()]
    print("best by auc")
    print(best_res.to_string())
    best_res = avg_res.loc[avg_res.groupby(['dataset']).rmse.idxmin()]
    print("best by rmse")
    print(best_res.to_string())


result_dir = 'kfold-results-hmm-scalable'
print(result_dir)
print_best(result_dir)
result_dir = 'kfold-results-standard-bkt'
print(result_dir)
print_best(result_dir)
