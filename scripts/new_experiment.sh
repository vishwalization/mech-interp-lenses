#!/usr/bin/env bash
# usage: ./scripts/new_experiment.sh exp006_logit_lens
NAME="$1"
if [ -z "$NAME" ]; then
    echo "Usage: $0 <experiment_name>"
    exit 1
fi
DIR="experiments/$NAME"
if [ -d "$DIR" ]; then
    echo "Error: $DIR already exists"
    exit 1
fi
cp -r experiments/_template "$DIR"
# set the name inside the copied config
sed -i '' "s/^name:.*/name: $NAME/" "$DIR/config.yaml"
echo "Created $DIR — now edit its config.yaml and run.py"
