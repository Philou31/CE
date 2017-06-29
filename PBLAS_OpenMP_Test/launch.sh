#!/bin/bash

export I_MPI_PIN_DOMAIN=socket
export KMP_AFFINITY=compact,granularity=fine

for n in 1 2 4 8 16 24
do
	export MKL_NUM_THREADS=24
	export OMP_NUM_THREADS=24
	for i in {1..10}
	do
		./main
	done
done
