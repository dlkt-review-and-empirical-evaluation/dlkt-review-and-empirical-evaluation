#!/bin/bash

# Provides functions used in more than one script
# This script should be sourced in the calling script: . script_utils.sh

usage() {
  # Example use in script: usage $0 1

  #This works for following arg parsing format (arg parsing should be first switch case)  TODO: make more general
  #while [ -n "$1" ]; do
  #  case $1 in
  #    -m | -modifier)       m="true"; shift ;;
  #    -s | -single-value)   s="$2"; [ -z "$2" ] && usage 1 "$2"; shift ;;
  #    --multi-value)        while [ -n "$2" ] && [ "${2:0:1}" != "-" ]; do multi_value="$multi_value $2"; [ -z "$2" ] && usage 1 "$2"; shift; done ;;
  #    -h | --help)          help ;;
  #    *)                    usage 1 "$2" ;;
  #  esac
  #  shift
  #done

  exit_code=${1:-0}
  [ -n "$2" ] && echo "Invalid argument $2"
  positional_arguments=$(sed -n -nE '/\-z\s"?\$1.*shift.*/p' "$0" | sed -E 's/.* ([^ ]+)=\$1.*/\U\1/g' | tr '\n' ' ')
  echo "Usage: $(basename $0) $positional_arguments"
  echo 'Options: '
  # Select optional arg lines and extract option and variable names
  opt_start=$(($(sed -nE '/case +\$1 +in/=' "$0" | sed -n '1p') + 1))
  opt_end=$(($(tail -n +$opt_start "$0" | sed -n '/esac/=' | sed -n '1p') + $opt_start - 3))
  sed -n "$opt_start,$opt_end p" "$0" | sed -E 's/\)(.*)=.*/ \U\1/g; s/WHILE.*DO\s([^\s]+).*/\1.../g; s/\).*//g; s/^/  /g'
  [ -n "$exit_code" ] && exit "$exit_code"
}

contains() {
  # Check if item is in list, eg. contains "1 2 3" 1 --> success, contains "1 2" 0 --> fail
  local array=$1
  local item=$2
  for element in $array; do
    if [ "$element" = "$item" ]; then
      return 0
    fi
  done
  echo "$item not found in $array"
  return 1
}

join_space_separated() {
  vals=$1
  delim=${2:-,}
  echo "$vals" | tr ' ' "$delim"
}
