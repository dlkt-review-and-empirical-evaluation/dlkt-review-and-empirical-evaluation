#!/bin/bash

cd "$(dirname $0)/.."

data_dir="data"

for dataset in $(find "$data_dir" -maxdepth 1 -type f -name '*.csv' | xargs -n1 basename | sed 's/.csv//'); do
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

  python baselines.py "$data_dir/$dataset.csv" $cols --max-attempt-count 200 --min-attempt-count 2

done
