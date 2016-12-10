#!/bin/bash
# This is the experiment 3.1 writen in my report
# Accuracy test
# case={39, 57}
case=57
matrix_type=(0 1 2 3 4)
for i in ${matrix_type[@]}
do
    python runpf.py --case_number $case -m $i --verbose 0 >>experiment_log/accuracy.log
done

echo ----------------------------------------------------------------->>experiment_log/accuracy.log
