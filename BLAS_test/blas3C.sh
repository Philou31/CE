#!/bin/bash -l

source /scratch/algo/leleux/start_system.gnu_intel.sh

export KMP_AFFINITY=scatter,granularity=fine

for n in 1 2 4 8 16 24
do
	for i in {1..10}
	do
		export OMP_NUM_THREADS=$n
		export KMP_NUM_THREADS=$n
		./blas3C
	done
done
