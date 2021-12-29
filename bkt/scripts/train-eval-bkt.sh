#!/bin/sh

cd "$(dirname $0)/.." || exit 1

usage() {
  echo "not done yet"
}

train_file="$1"
test_file="$2"
train_predict_file=predict.train.txt
test_predict_file=predict.test.txt
model_file=model.txt

script_dir=$(dirname "$0")
bkt_dir="$script_dir/../hmm-scalable"

echo "Training BKT..."
"$bkt_dir/trainhmm" -s 1.1 -m 1 -p 1 "$train_file" $model_file $train_predict_file
echo "Predicting results..."
"$bkt_dir/predicthmm" -p 1 "$test_file" "$model_file" $test_predict_file

python $script_dir/eval_from_predict.py $test_predict_file $test_file