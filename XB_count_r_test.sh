#!/usr/bin/env bash
# This is the experiment 3.3.1 writen in my report
# Accuracy test
# case={14, 39, 57, 118}
case=$1
matrix_type=(2 5)
for i in ${matrix_type[@]}
do
    python runpf.py --case_number $case -m $i --verbose 0 >>experiment_log/XB_r.log
done

echo ----------------------------------------------------------------->>experiment_log/XB_r.log
