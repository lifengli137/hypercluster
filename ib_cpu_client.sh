for i in 0 1 2 3 6 7 8 9; do ib_write_bw $1 -d mlx5_$i -i 1 -a --report_gbits -F;sleep 2; done
