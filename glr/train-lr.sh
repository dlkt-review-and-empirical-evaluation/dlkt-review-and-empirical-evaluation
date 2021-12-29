#!/bin/sh

data_dir="../data/preprocessed"

for dataset in $(find $data_dir -name data.csv | sort | xargs dirname | sed 's,.*/,,'); do
    echo "training best-lr on $dataset..."
    python train_lr.py --X_file $data_dir/$dataset/X-isicsctcwa.npz --dataset $dataset $@ || continue # lr-best
    echo "done training best-lr on $dataset"
done
echo "all done"