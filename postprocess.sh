#!/bin/bash

instance_ids=(2 3 4 5)
problem_ids=(1 3 4 5 16 23)
epsilons_start=0
epsilons_end=7
bchm_id_start=0
bchm_id_end=13

# MAX_CONCURRENT=3

count=0

for instance_id in "${instance_ids[@]}"; do
    for problem_id in "${problem_ids[@]}"; do
        for ((epsilon=$epsilons_start; epsilon<=$epsilons_end; epsilon++)); do
            for ((bchm_id=$bchm_id_start; bchm_id<=$bchm_id_end; bchm_id++)); do
                # if [ $count -ge $MAX_CONCURRENT ]; then
                #     wait -n
                #     count=$((count-1))
                # fi
                # python postprocess.py $instance_id $problem_id $epsilon $bchm_id &
                python time_series_discovery.py $instance_id $problem_id $epsilon $bchm_id
                # count=$((count+1))
            done
        done
    done
done

wait
echo "All commands have been executed."
