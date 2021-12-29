#!/bin/sh

data_dir="../data/preprocessed"

for dataset in $(find $data_dir -name data.csv | sort | xargs dirname | sed 's,.*/,,'); do
    echo "preparing $dataset data..."
    python prepare_data.py --dataset "$dataset" --min_interactions 2 --remove_nan_skills
    echo "encoding $dataset data for logreg..."

    # GLR
    python encode.py --dataset "$dataset" -i -s -ic -sc -tc -w -a

    #GLR with time windows
    #python encode.py --dataset $dataset -i -s -ic -sc -tc -w -a -tw

    # #DAS3H
    # python encode.py --dataset $dataset -i -s -sc -w -a -tw
done
echo "all done"
