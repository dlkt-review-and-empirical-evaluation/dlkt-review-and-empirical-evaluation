#!/bin/sh

cd "$(dirname $0)/.." || exit 1

for bkt in hmm-scalable standard-bkt; do
  for a in 7 6 5 4 3 2 1; do
    for b in 1 2 3; do
  #    for c in 1 2 3 4; do
      for c in 1; do  # These didn't affect performance on first run
         solver=$a.$b
         [ $b = 3 ] && solver=$solver.$c
         for x in data/5fold/*; do

           x=${x##*/}
           scripts/kfold-train-eval-bkt.sh "data/5fold/$x/$x" $solver $bkt;
         done

#         [ $b != 3 ] && break
       done
     done
   done
done