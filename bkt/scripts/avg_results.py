import os
import argparse
import pandas as pd
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('results_file')

    args = parser.parse_args()
    results = pd.read_csv(args.results_file)

    avgs = results.mean().apply(lambda x: np.round(x, 3)).apply(str)
    sds = results.std().apply(lambda x: np.round(x, 3)).apply(str)

    avgs_and_sds = avgs.str.cat(sds, sep='±')

    avg_file = os.path.splitext(args.results_file)[0] + '.avgs.csv'
    print(avgs_and_sds)
    avgs_and_sds.to_csv(avg_file, index=True, header=False)
    print('Wrote {}'.format(avg_file))
