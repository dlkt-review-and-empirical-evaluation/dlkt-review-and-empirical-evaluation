# Gervet et al. best logistic regression for comparison

Modified from logistic regression code in [original repo](`https://github.com/theophilee/learner-performance-prediction.git`)

## Setup

### GLR

Setup conda environment (same as in the original repos readme):

Create a new conda environment, install [PyTorch](https://pytorch.org) and the remaining requirements:

```sh
conda create python==3.7 -n learner-performance-prediction
conda activate learner-performance-prediction
pip install -r requirements.txt
conda install pytorch==1.2.0 torchvision==0.4.0 -c pytorch
```

### Data

Preprocess data (needed for maximum attempt count split), other steps are already in the lr code

```sh
../scripts/preprocess-data.sh
./prepare-data.sh
```

## Training

5-fold cross-validation

```sh
train-lr.sh
```

## Average fold scores

Run

```sh
python avg_scores.py
```

The results are found in the results directory
