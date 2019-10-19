#!/usr/bin/env bash

nvidia-smi nvlink -sc 0bz >/dev/null
space='    '
while :
do
    for i in `seq 0 $1`; 
    do
        old[${i}]=`nvidia-smi nvlink -g 0 -i ${i} | grep -o Tx0:.*KBytes | awk '{s+=$2} END {printf "%.0f", s}'`
    done
    sleep 1
    for i in `seq 0 $1`; 
    do
        new[${i}]=`nvidia-smi nvlink -g 0 -i ${i} | grep -o Tx0:.*KBytes | awk '{s+=$2} END {printf "%.0f", s}'`
    done
    for i in `seq 0 $1`; 
    do
        echo -n "$(((new[$i] - old[$i])/1024))MB/s${space}"
        old[${i}]=$new[${i}]
    done
    echo 
 done
