
iid_start=2
iid_end=2
pids=(4)
bchm_id_start=0
bchm_id_end=13

# 遍历所有参数组合
for ((iid=$iid_start; iid<=$iid_end; iid++)); do
    for pid in "${pids[@]}"; do
        for ((bchm_id=$bchm_id_start; bchm_id<=$bchm_id_end; bchm_id++)); do
            echo "Running: python discovery.py $iid $pid $bchm_id"
            nohup python discovery.py $iid $pid $bchm_id &
        done
    done
done

echo "All commands have been executed."