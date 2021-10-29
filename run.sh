#!/bin/bash
echo "[Info] Evaluate multi bert"
time python evaluate_multi_bert.py
echo "[Info] Evaluate simnet_13"
time python evaluate.py
echo "[Info] Fusion result"
time python post_process.py
echo "[Info] Zip result"
zip result/result.zip result/result.json
