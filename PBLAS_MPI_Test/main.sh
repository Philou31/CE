#!/bin/bash

for n in 1 2 4 8 16 24
do
	for i in {1..10}
	do
 		export MKL_NUM_THREADS=$n
		export OMP_NUM_THREADS=$n
		./main
	done
done
