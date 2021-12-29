#!/bin/sh

cd "$(dirname $0)/.." || exit 1

baseline_res_file="kfold-baseline-results.csv"
glr_res_file="kfold-glr-results.csv"

# Combine BKT result with simple baseline results in directory
bkt/scripts/cp-to-baseline-results.sh

# Crawl baseline dirs to generate result csv files
./kfold-result-crawler baseline-results -o $baseline_res_file -m baseline-metrics -e results.csv
./kfold-result-crawler glr/results/ -o $glr_res_file -m baseline-metrics -e results.csv

# Combine result csv files
tail +2 $glr_res_file >> $baseline_res_file
scripts/rename-datasets-and-models.sh $baseline_res_file

# Sort
header=$(head -n1 $baseline_res_file)
tail +2 $baseline_res_file | sort -o $baseline_res_file
sed -i -e "1s/^/$header\n/" $baseline_res_file

echo "Combined results to $baseline_res_file"

# Remove extra result file
rm $glr_res_file
echo "Removed temporary file $glr_res_file"