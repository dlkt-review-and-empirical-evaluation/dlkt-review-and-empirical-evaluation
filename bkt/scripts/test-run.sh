#!/bin/sh

cd "$(dirname $0)/.." || exit 1

[ -z "$1" ] && echo "data required as first argument" && exit 1

hmm-scalable/trainhmm -s 1.1 -m 1 -p 1 "$1" model.txt predict.txt


