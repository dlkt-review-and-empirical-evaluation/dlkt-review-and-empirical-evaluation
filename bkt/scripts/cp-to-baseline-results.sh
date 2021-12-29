#!/bin/sh

cd "$(dirname $0)/.." || exit 1

mkdir_n_cp() {
  mkdir -p $(dirname $2)
  cp -r $1 $2
}

baseline_results_dir='../baseline-results'

mkdir_n_cp kfold-results-standard-bkt/1.1/assist09-updated/5fold_results.avgs.csv $baseline_results_dir/assist09-updated/BKT/5fold-avg-results.csv
mkdir_n_cp kfold-results-standard-bkt/1.1/assist15/5fold_results.avgs.csv $baseline_results_dir/assist15/BKT/5fold-avg-results.csv
mkdir_n_cp kfold-results-standard-bkt/1.1/assist17-challenge/5fold_results.avgs.csv $baseline_results_dir/assist17-challenge/BKT/5fold-avg-results.csv
mkdir_n_cp kfold-results-standard-bkt/1.3.1/statics/5fold_results.avgs.csv $baseline_results_dir/statics/BKT/5fold-avg-results.csv
mkdir_n_cp kfold-results-standard-bkt/1.3.1/synthetic-5-k2/5fold_results.avgs.csv $baseline_results_dir/synthetic-5-k2/BKT/5fold-avg-results.csv
mkdir_n_cp kfold-results-standard-bkt/1.3.1/synthetic-5-k5/5fold_results.avgs.csv $baseline_results_dir/synthetic-5-k5/BKT/5fold-avg-results.csv
mkdir_n_cp kfold-results-standard-bkt/1.3.1/intro-prog/5fold_results.avgs.csv $baseline_results_dir/intro-prog/BKT/5fold-avg-results.csv
