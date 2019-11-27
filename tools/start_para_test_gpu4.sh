#!/bin/bash
for idx in 0 1 2 3
do
	tmux new-session -d -s "set$idx" \; 
	send-keys "mmd" Enter \; 
	send-keys "python ./tools/test_lvis.py configs/mask_rcnn_r50_fpn_1x_lvis.py /home/wangtao/prj/liyu_mmdet/work_dirs/mask_rcnn_r50_fpn_1x_lvis_liyu_finetune_imglevelsampler/epoch_12.pth --out ./set$idx.pkl --eval segm --set $idx" Enter \;
done

