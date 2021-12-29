#!/bin/bash

# Import util functions
. "$(dirname "$0")/script-utils.sh"

help() {
  echo "Script for generating model training scripts"
  usage "$0"
}

# Required arguments
[ -z "$1" ] && usage 1; dataset=$1; shift
[ -z "$1" ] && usage 1; model=$1; shift

case $dataset in
  assist09up)  cols=""; data="data/assist09-updated.csv" ;;
  assist15)    cols="--skill-col sequence_id"; data="data/assist15.csv";;
  assist17)    cols="--user-col studentId --skill-col skill"; data="data/assist17-challenge.csv" ;;
  stat)     cols=""; data="data/statics.csv" ;;
  intro-prog) cols="--skill-col assignment_id"; data="data/intro-prog.csv" ;;
  synth-k2) cols=""; data="data/synthetic-5-k2.csv" ;;
  synth-k5) cols=""; data="data/synthetic-5-k5.csv" ;;
  *)        echo -e "invalid dataset name $dataset provided, possible model_options are:
                     ass09up, ass15, ass17, stat, intro-prog,
                     synth-k2 and synth-k5"  && exit 1 ;;
esac

# Optional arguments
k=5
seed=42
dropout=0.2
batch_size=32
init_lr=1e-2
verbosity=1
script_dir="model-run-scripts"

while [ -n "$1" ]; do
  case $1 in
    -v | --verbosity)     verbosity="$2"; [ -z "$2" ] && usage "$0" 1 "$2"; shift ;;
    -k)                   k="$2"; [ -z "$2" ] && usage "$0" 1 "$2"; shift ;;
    -s | --seed)          seed="$2"; [ -z "$2" ] && usage "$0" 1 "$2"; shift ;;
    -d | --dropout)       dropout="$2"; [ -z "$2" ] && usage "$0" 1 "$2"; shift ;;
    -l | --layer-sizes)   while [ -n "$2" ] && [ "${2:0:1}" != "-" ]; do layer_sizes="$layer_sizes $2"; [ -z "$2" ] && usage "$0" 1 "$2"; shift; done ;;
    --max-attempts)       max_attempts="$2"; [ -z "$2" ] && usage "$0" 1 "$2"; shift ;;
    --attempt-filter)     attempt_filter="$2"; [ -z "$2" ] && usage "$0" 1 "$2"; shift ;;
    --init-lr)            init_lr="$2"; [ -z "$2" ] && usage "$0" 1 "$2"; shift ;;
    --attention-heads)    attention_heads="$2"; [ -z "$2" ] && usage "$0" 1 "$2"; shift ;;
    --n-blocks)           n_blocks="$2"; [ -z "$2" ] && usage "$0" 1 "$2"; shift ;;
    --batch-size)         batch_size="$2"; [ -z "$2" ] && usage "$0" 1 "$2"; shift ;;
    --onehot-input)       onehot_input=true ;;
    --output-per-skill)   output_per_skill=true ;;
    --skip-validation)    skip_validation=true ;;
    --script-dir)         script_dir="$2"; [ -z "$2" ] && usage "$0" 1 "$2"; shift ;;
    -h | --help)          help ;;
    *)                    usage "$0" 1 "$2" ;;
  esac ; shift
done

if [ -z "$skip_validation" ]; then
  [ "$verbosity" -lt 1 ] || echo "validating model... you may skip validationv via --skip-validation to speed up script generation"
  available_models=$(python -W ignore -c 'import conf; print(*conf.model_choices, sep=" ")' 2>/dev/null) || {
    echo "python >=3.6 and project requirements (see requirements.txt) are required to validate model options from code"
    exit 1
  }
  contains "$available_models" "$model" || {
    echo "error: $model not found in available models: $available_models"
    exit 1
  }
fi

[ -z "$layer_sizes" ] && layer_sizes="50,20,40,50"
# Trim leading whitespace
layer_sizes=$(echo $layer_sizes | sed 's/^\s*//g')

model_options="--seed=$seed --dropout=$dropout --layer-sizes=$(join_space_separated "$layer_sizes" ',') --init-lr=$init_lr --batch-size=$batch_size"

if [ -n "$max_attempts" ]; then
  [ -z "$attempt_filter" ] && echo "attempt filter is required with max attempt count" && exit 1
  model_options="$model_options --max-attempt-count=$max_attempts --max-attempt-filter=$attempt_filter"
fi

[ "$model" = "transformer" ] && model_options="$model_options --n-blocks $n_blocks"

[ -n "$attention_heads" ] && { [ -z "${model##sakt*}" ] || [ -z "${model##*transformer*}" ]; } && model_options="$model_options --n-heads=$attention_heads"
[ -n "$n_blocks" ] && { [ -z "${model##sakt*}" ] || [ -z "${model##*transformer*}" ]; } && model_options="$model_options --n-blocks=$n_blocks"
[ -n "$onehot_input" ] && model_options="$model_options --onehot-input"
[ -n "$output_per_skill" ] && model_options="$model_options --output-per-skill"

model_dir="${k}fold/$dataset/$model"

script_path="$model_dir/$(echo "$model_options" | sed 's/--//g; s/ /__/g; s/[,=]/_/g')".sh
mkdir -p "$script_dir/$model_dir"

echo -e "#!/bin/bash
python -W ignore main.py  $data --model $model $cols $model_options --early-stopping 10 --epochs 100" > "$script_dir/$script_path"

[ "$verbosity" -lt 1 ] || echo "Wrote $script_dir/$script_path"
