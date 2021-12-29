# Yudelson BKT for DLKT comparison

## Requirements

https://github.com/taikamurmeli/kt-data-fiddler
for converting csv data for bkt in this repository's parent directory.
```sh
git clone https://github.com/taikamurmeli/kt-data-fiddler ../../kt-data-fiddler
```

For alternate locations, modify path in scripts/prepare-kfold-data.sh

## Setup

### BKT models

Clone https://github.com/IEDMS/standard-bkt
and https://github.com/myudelson/hmm-scalable.git to this directory.

To ensure the repo versions are same as in this study reset the repositories:

```sh
cd standard-bkt && git reset --hard 80baafd43224b59e147a66803e08f6a4cb3021c5
cd ../hmm-scalable && git reset --hard 0292322a5e28fe688b70ccae00d14f32c452b8bc
```

Then compile the hmm-scalable and standard-bkt according to their readmes

### Data

Prepare 5-fold train and test data

- configure data directory and datasets in script

```sh
scripts/prepare-kfold-data.sh
```

## Training and evaluation

5-fold cross-validation for each solver

```sh
scripts/hypertune-bkt.sh
```

### Inspect results

Run

```sh
python scripts/find_best_ver.py
```
