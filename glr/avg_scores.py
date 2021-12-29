import pandas as pd
import numpy as np
import os
from pathlib import Path



data_dir = '../data/preprocessed'

result_entries = os.listdir(data_dir)

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def remove_suffix(text, suffix):
    if text.endswith(suffix):
        return text[:len(text) - len(suffix)]
    return text

res_dir = 'results'
for res_entry in result_entries:
    if not os.path.isdir(os.path.join(data_dir, res_entry)): continue

    dataset = res_entry

    res_df = pd.read_csv(os.path.join(res_dir, f'5-fold-lrbest-{dataset}.csv'))

    means = res_df.mean().apply(lambda x: np.round(x, 3)).apply(str)
    sds = res_df.std().apply(lambda x: np.round(x, 3)).apply(str)

    avgs_and_sds = means.str.cat(sds, sep='/')

    save_dir = os.path.join(res_dir, f'{dataset}/lrbest/')
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(save_dir, '5fold-avg-results.csv')
    avgs_and_sds.to_csv(save_path, header=False)
    print(f'wrote {save_path}')
