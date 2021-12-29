#!/bin/sh

cd "$(dirname "$0")/.." || exit 1

k=${3:-5}

data_dir="../data/preprocessed"

for dataset in $(find "$data_dir" -type f -name data.csv | sort | xargs dirname | sed 's,.*/,,'); do
  in_file="$data_dir/$dataset/data.csv"
  mkdir -p "data/5fold/$dataset"
  out_file="data/5fold/$dataset/$dataset"

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

  python ../../kt-data-fiddler/converter.py "$in_file" "$out_file" $cols --in-format csv --out-format yudelson-bkt --kfold "$k"

done
