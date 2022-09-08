ibs=(0 1 2 3 6 7 8 9)

echo "IB tests for the odd number of GPUs: "
for i in 1,3,5,7,9,11,13,15 1,3 1,5 1,7 1,9 1,11 1,13 1,15 3,5 3,7 3,9 3,11 3,13 3,15 5,7 5,9 5,11 5,13 5,15 7,9 7,11 7,13 7,15 9,11 9,13 9,15 11,13 11,15 13,15; do
        echo "CUDA_VISIBLE_DEVICES=$i"
        idxs=`echo $i | tr ',' ' '`
        for j in $idxs; do
                idx=$((j/2))
                echo "Monitor IB${ibs[$idx]}'s throughputs.";
                done

        n=`echo $i | tr -cd , | wc -c`;
        ((n++))
        CUDA_VISIBLE_DEVICES=$i mpirun --allow-run-as-root -np $n -mca pml ucx --mca btl ^vader,tcp,openib -x NCCL_SHM_DISABLE=1 -x NCCL_P2P_DISABLE=1 -x NCCL_IB_DISABLE=0  -x LD_LIBRARY_PATH=/opt/nccl-2.8.4-1/build/lib /opt/nccl-tests-2.11.0/build/alltoall_perf -b 2G -e 2G -c 1 -w 20 -n 50;

done
