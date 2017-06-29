#!/bin/bash -l

#SBATCH --partition prod
#SBATCH --job-name test_1
#SBATCH --output=/scratch/algo/leleux/CE/SCALAPACK_Test/sbatch_test_1.out
#SBATCH --error=/scratch/algo/leleux/CE/SCALAPACK_Test/sbatch_test_1.err
#SBATCH --mail-user leleux@cerfacs.fr
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

cd /scratch/algo/leleux/CE/SCALAPACK_Test
source /scratch/algo/leleux/start_system.intel_2017.sh

RESULT=$(./blas3C_sequential 1000 | cut -d "." -f 1)

if [ $RESULT -gt 38 ]
then
	echo "Weird node: Peak speed per core greater than 40GFlop/s"
else
	export I_MPI_PIN_DOMAIN=socket
	export KMP_AFFINITY=compact,granularity=fine

	for b in 16 32 64 128 256 512 1024 2048 4096 1 2 4 8
	do
	        for n in 1 2 4 8 12
	        do
			export MKL_NUM_THREADS=$n
			export OMP_NUM_THREADS=$n
			mpirun -n 1 ./main 30000 1 1 $b
		done
	done
fi
