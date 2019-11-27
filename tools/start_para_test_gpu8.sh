#!/bin/bash
config=$1
model=$2
out=$3

for idx in 0 1 2 3 4 5 6 7
do
        tmux kill-session -t "set$idx"
done

for idx in 0 1 2 3 4 5 6 7
do
	tmux new-session -d -s "set$idx" \; send-keys "mmd" Enter \; send-keys "CUDA_VISIBLE_DEVICES=$idx python ./tools/test_lvis_split_parallel.py $config $model --out ./$out"_set"$idx.pkl --eval segm --set $idx --total_set_num 8" Enter \;
done

