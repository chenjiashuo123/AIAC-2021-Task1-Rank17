#!/bin/bash
echo "[Info] Pretrain bert with tag"
time python train_text_model_focal_2.py
echo "[Info] Pretrain video transformers with tag"
time python train_video_model.py
echo "[Info] Pretrain mulit_bert with tag"
time python train_multi_bert_tag.py
echo "[Info] Generate finetune datas"
time python finetune_data_gen.py
time python sort_label.py
echo "[Info] finetune model 1"
time python finetune_simnet_13_unsup_all.py
echo "[Info] finetune model 2"
time python finetune_multi_bert_unfrozen_1_unsup_all.py