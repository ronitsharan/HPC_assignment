#!/bin/bash
export KMP_AFFINITY=compact  #Rule to bind core to thread for OMP thread with Intel compiler for parallel version
export OMP_NUM_THREADS=8     #Set OMP number of threads for parallel version
export BLISLAB_IC_NT=8       #Set BLISLAB number of threads for parallel version
k_start=240
k_end=2400
k_blocksize=240
echo "run_step3_st=["
echo -e "%m\t%n\t%k\t%MY_GFLOPS\t%REF_GFLOPS"
for (( k=k_start; k<=k_end; k+=k_blocksize ))
do
    ./a.out     $k $k $k
done
echo "];"
