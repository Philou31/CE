#!/bin/bash -x
#SBATCH --job-name=Mumps_execute_1
#SBATCH --mail-user=leleux@cerfacs.fr
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --partition=prod

source /scratch/algo/leleux/start_system/start_system.intel_2017.sh
           export FORT_BUFFERED=$use_buffered_io

RESULT=$(./blas3C_sequential 1000 | cut -d "." -f 1)

if [ $RESULT -gt 38 ]
then
	echo "Weird node: Peak speed per core greater than 40GFlop/s"
else
	for b in 128 256 512
	do
	        for n in 1 4 12
	        do
			export MKL_NUM_THREADS=$n
			export OMP_NUM_THREADS=$n
			time mpirun -n 2 ./main 10000 1 2 $b
		done
	done
fi
