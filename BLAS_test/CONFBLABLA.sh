#!/bin/bash

icc -o blas3C blas3C.c  -DMKL_LP64 -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread
