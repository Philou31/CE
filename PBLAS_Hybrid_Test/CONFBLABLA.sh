#!/bin/bash

mpiicpc -o main main.cpp -DMKL_LP64 -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 -liomp5 -lpthread
