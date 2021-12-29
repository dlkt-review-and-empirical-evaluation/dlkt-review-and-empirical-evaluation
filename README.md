# Deep Learning Models for Knowledge Tracing: Review and Empirical Evaluation

Results and code for evaluating and comparing (deep) knowledge-tracing models and simple baselines, implemented in python and tensorflow v2.

## Model implementations

### DLKT models

- Vanilla/LSTM-DKT (original Deep Knowledge Tracing with Vanilla/LSTM RNN kernel)
- LSTM-DKT-S+
  - LSTM-DKT with key embeddings of next attempt concatenated to LSTM output
- DKVMN
  - based on ([MXNext implementation](https://github.com/jennyzhang0215/DKVMN))
- DKVMN-Paper ([Dynamic Key Value Memory Network](https://arxiv.org/abs/1611.08108))
  - based on the article (in the link)
- SAKT ([Self-Attentive Knowledge Tracing](https://arxiv.org/abs/1907.06837))

### Non-DLKT Baselines

- GLR
- BKT

### Naive baselines

- Majority vote
- Mean
- Predict next as mean of N previous attempts

## Setup

### Data
Used data in experiments is found in data directory apart from
[ASSISTments 2017 challenge](https://sites.google.com/view/assistmentsdatamining),
which can be accessed from the linked site.
The data should be exported as csv and placed in data/assist17-challenge.csv

Namely, the data directory includes the datasets
- [ASSISTments 2009-2010](https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data/skill-builder-data-2009-2010) (assist09-updated)
- [ASSISTments 2015](https://sites.google.com/site/assistmentsdata/home/2015-assistments-skill-builder-data) (assist15)
- IntroProg -- new dataset
- [Statics](https://pslcdatashop.web.cmu.edu) (statics)
- [Synthetic 5 K2/5](https://github.com/chrispiech/DeepKnowledgeTracing/tree/master/data/synthetic) (synthetic-k5-k2/5)

The assist09-updated and statics datasets are retrieved from
https://github.com/jennyzhang0215/DKVMN/tree/master/data/


### Code
Create a new python environment, e.g. with conda

```sh
conda create python==3.8 -n dlkt-review-and-empirical-evaluation
```

Activate it and install requirements

```sh
conda activate dlkt-review-and-empirical-evaluation
pip install -r requirements.txt
```

## Model training and evaluation

### DLKT models

Train a model with default options (default model: lstm-dkt) `python main.py my-data.csv`

Kfold cross-validation is done by passing argument `--kfold $k`, e.g `python main.py data/statics.csv --model sakt --kfold 5`

To see training options and defaults run `python main.py -h` or inspect conf.py.

Alternatively you can generate a script via `scripts/generate-slurm-script.sh $data_abbreviation $model $other_options`,
where data abbreviations are: assist09up, assist15, assist17, stat, intro-prog, synth-k2 and synth-k5.
Script options are available via `scripts/generate-local-run-script.sh -h`

Examples to run all or any of the best hyperparameter combinations according to our grid search are in
scripts/grid-search-top-runs.sh

### Naive baselines

scripts/run-baselines.sh

### GLR

see glr/README.md

### BKT

see bkt/README.md

## Results

The study results are provided in files grid-search-kfold-results.csv, 
baseline-kfold-results.csv, max-attempt-kfold-results.csv

Top results can be extracted from the grid search results for a given metric via
scripts/extract_best_kfold_results.py, e.g.
```sh
python scripts/extract_best_kfold_results.py grid-search-kfold-results.csv --metric auc 
python scripts/extract_best_kfold_results.py grid-search-kfold-results.csv --metric rmse 
```
