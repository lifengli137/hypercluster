ibs=(0 1 2 3 6 7 8 9)

echo "IB tests for the even number of GPUs: "
for i in 0,2,4,6,8,10,12,14 0,2 0,4 0,6 0,8 0,10 0,12 0,14 2,4 2,6 2,8 2,10 2,12 2,14 4,6 4,8 4,10 4,12 4,14 6,8 6,10 6,12 6,14 8,10 8,12 8,14 10,12 10,14 12,14; do
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
