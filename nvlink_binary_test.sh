echo "NVLINK tests through binary partitioning GPUs: "
for i in 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 0,1,2,3,4,5,6,7 8,9,10,11,12,13,14,15 0,1,2,3 4,5,6,7 8,9,10,11 12,13,14,15 0,1 2,3 4,5 6,7 8,9 10,11 12,13 14,15 ; do 
	echo "CUDA_VISIBLE_DEVICES=$i"
	n=`echo $i | tr -cd , | wc -c`;
	((n++))
	CUDA_VISIBLE_DEVICES=$i mpirun --allow-run-as-root -np $n -mca pml ucx --mca btl ^vader,tcp,openib -x LD_LIBRARY_PATH=/opt/nccl-2.8.4-1/build/lib /opt/nccl-tests-2.11.0/build/alltoall_perf -b 2G -e 2G -c 1 -w 20 -n 50; 
done
