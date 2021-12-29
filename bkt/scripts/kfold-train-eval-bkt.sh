#!/bin/bash

usage() {
  echo "TODO"
}

script_dir=$(dirname "$0")

file="$1"
solver=${2:-1.3.1}
bkt=${3:-hmm-scalable}
k=${4:-5}
train_file="$file".train
test_file="$file".test
bkt_dir="$script_dir/../$bkt"
res_dir=kfold-results-$bkt/$solver/$(basename "${file%.*}")
mkdir -p $res_dir
res_file=$res_dir/${k}fold_results.csv
train_predict_file=$res_dir/predict.train.txt
test_predict_file=$res_dir/predict.test.txt
model_file=$res_dir/model.txt

# bkt_dir="$script_dir/../standard-bkt"


rm -f "$res_file"
for (( i=0; i<$k; i++ )); do
  echo "Training BKT..."
  if "$bkt_dir/trainhmm" -s "$solver" -m 1 "$train_file.$i" "$model_file" "$train_predict_file" > /dev/null; then
    echo "Predicting results..."
    "$bkt_dir/predicthmm" -p 1 "$test_file.$i" "$model_file" "$test_predict_file" >/dev/null

    # Appends prediction results to res_file
    python "$script_dir"/eval_from_predict.py "$test_predict_file" "$test_file".$i "$res_file"
  fi
done

# Compute and save avgs and stds from res_file
python "$script_dir"/avg_results.py "$res_file"
