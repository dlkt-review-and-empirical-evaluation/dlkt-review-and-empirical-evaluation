#!/bin/sh

filename="$1"

sed -i 's/rmse/RMSE/g' "$filename"
sed -i 's/acc/Acc/g' "$filename"
sed -i 's/mcc/MCC/g' "$filename"
sed -i 's/f1/F1/g' "$filename"
sed -i 's/auc/AUC/g' "$filename"