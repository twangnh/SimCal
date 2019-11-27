#!/bin/bash
for idx in 0 1 2 3 4 5 6 7
do
	tmux kill-session -t "set$idx"
done
