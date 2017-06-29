source /scratch/algo/leleux/start_system/start_system.intel_2017.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
time ./blas3C 10000

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
time ./blas3C 10000

export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
time ./blas3C 10000
