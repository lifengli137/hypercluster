#!/usr/bin/env bash
space='    '
while :
    do
        for i in `seq 0 $1`; 
        do
        old[${i}]=`cat /sys/class/infiniband/mlx5_${i}/ports/1/counters/port_xmit_data`
    done
        sleep 1
        for i in `seq 0 $1`; 
    do
        new[${i}]=`cat /sys/class/infiniband/mlx5_${i}/ports/1/counters/port_xmit_data`
    done
    for i in `seq 0 $1`; 
    do
        echo -n "$(((new[$i] - old[$i])/262144))MB/s${space}"
    old[${i}]=$new[${i}]
    done
    echo 
done
