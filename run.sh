#!/bin/bash

iid_start=2
iid_end=3
pids=(1 3 4 5 16 23)
bchm_id_start=0
bchm_id_end=13

MAX_CONCURRENT=6

count=0

for ((iid=$iid_start; iid<=$iid_end; iid++)); do
    for pid in "${pids[@]}"; do
        for ((bchm_id=$bchm_id_start; bchm_id<=$bchm_id_end; bchm_id++)); do
            if [ $count -ge $MAX_CONCURRENT ]; then
                wait -n
                count=$((count-1))
            fi
            echo "Starting: python discovery.py $iid $pid $bchm_id"
            python discovery.py $iid $pid $bchm_id &
            count=$((count+1))
        done
    done
done

wait
echo "All commands have been executed."
