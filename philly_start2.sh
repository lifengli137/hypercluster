#!/bin/bash

container=`cat $PHILLY_SCRATCH_DIR/mpi-hosts | head -n 1 | awk '{print $1}' | tr - _`
index=`cat $PHILLY_RUNTIME_CONFIG | jq --arg name "$container" -r '.containers[$name].index'`
master_ip=`cat $PHILLY_RUNTIME_CONFIG | jq --arg name "$container" -r '.containers[$name].ip'`
master_port_start=`cat $PHILLY_RUNTIME_CONFIG | jq --arg name "$container" -r '.containers[$name].portRangeStart'`
master_port_end=`cat $PHILLY_RUNTIME_CONFIG | jq --arg name "$container" -r '.containers[$name].portRangeEnd'`

export MASTER_ADDR=$master_ip
export MASTER_PORT=$((master_port_start+7))
echo $MASTER_ADDR
echo $MASTER_PORT

python3 $1 $2 $3
