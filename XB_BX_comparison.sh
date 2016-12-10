#!/usr/bin/env bash
case=$1
matrix_type=(2 3)
scale=(1 1.5 2 2.5 3)
for i in ${matrix_type[@]}
do
    for j in ${scale[@]}
    do
        python runpf.py --case_number $case -m $i --scale $j --verbose 0 >>experiment_log/XB_BX_comparison.log
    done
done

echo ----------------------------------------------------------------->>experiment_log/XB_BX_comparison.log
