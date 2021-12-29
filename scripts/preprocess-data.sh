#!/bin/sh

cd "$(dirname "$0")/.." || exit 1

data_dir="data"
preprocessed_dir="$data_dir/preprocessed"

for dataset in $(find "$data_dir" -maxdepth 1 -type f -name '*.csv' | xargs -n1 basename | sed 's/.csv//'); do
  mkdir -p "$preprocessed_dir/$dataset"
  case $dataset in
    assist09-updated)     cols="" ;;
    assist15)             cols="--skill-col sequence_id" ;;
    assist17-challenge)   cols="--user-col studentId --skill-col skill" ;;
    intro-prog)           cols="--skill-col assignment_id" ;;
    synthetic-5-k2)       cols="" ;;
    synthetic-5-k5)       cols="" ;;
    statics)              cols="" ;;
    *)                    echo -e "encountered unexpected dataset name: $dataset"; continue
  esac

  preprocessed_file="$preprocessed_dir/$dataset/data.csv"
  python preprocess.py "$data_dir/$dataset.csv" "$preprocessed_file" $cols --max-attempt-count 200 --min-attempt-count 2
done
