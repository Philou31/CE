// STL 
#include <iostream> 
#include <cstdlib>
#include "mpi.h" 
#include <time.h>
#include <math.h>

using std::cout; 
using std::endl; 

extern "C"
{ 
	void Cblacs_pinfo (int* mypnum, int* nprocs); 
	void Cblacs_get (int context, int request, int* value); 
	int Cblacs_gridinit (int* context, char * order, int np_row, int np_col); 
	void Cblacs_gridinfo (int context, int* np_row, int* np_col, int* my_row, int* my_col); 
	void Cblacs_gridexit (int context); 
	void Cblacs_exit (int error_code); 
	int numroc_ (int *n, int *nb, int *iproc, int *isrcproc, int *nprocs); 
	void descinit_ (int *desc, int *m, int *n, int *mb, int *nb, int *irsrc, int *icsrc, int *ictxt, int *lld, int *info); 
	void pdgemm_ (char *TRANSA, char *TRANSB, int * M, int * N, int * K, double * ALPHA, 
	double * A, int * IA, int * JA, int * DESCA, double * B, int * IB, int * JB, int * DESCB, 
	double * BETA, double * C, int * IC, int * JC, int * DESCC ); 
}

int main(int argc, char **argv) 
{ 

//////////////////////////////////////////////////
// Initialize Communications
//////////////////////////////////////////////////
MPI_Status status;
MPI::Init(argc, argv); 
int rank = MPI::COMM_WORLD.Get_rank(); 
int size = MPI::COMM_WORLD.Get_size(); 
char processor_name[MPI_MAX_PROCESSOR_NAME];
int namelen = 0;
MPI_Get_processor_name(processor_name, &namelen);
int val=0;
// Info MPI
if (size!=1) {
	if (rank!=0)
		MPI_Recv(&val, 1, MPI_INTEGER, rank-1, 0, MPI::COMM_WORLD, &status);
	cout << "MPI_proc\tSize\tCPU\tProcessor\n";
	cout << rank << "\t" << size << "\t" << sched_getcpu() << "\t" << processor_name << "\n";
	if (rank==0)
		MPI_Send(&val, 1, MPI_INTEGER, 1, 0, MPI::COMM_WORLD);
	else {
		MPI_Send(&val, 1, MPI_INTEGER, (rank+1)%size, 0, MPI::COMM_WORLD);
	}
	MPI_Barrier(MPI::COMM_WORLD);
}

//////////////////////////////////////////////////
// Initialize Computation
//////////////////////////////////////////////////
// Data 
int n = atoi(argv[1]); // nrows=ncols of a squared matrix 

// Initialize BLACS grid 
int nprow = atoi(argv[2]); // grid size
int npcol = atoi(argv[3]); // grid size 
//if (size > 3)
//	nprow = 2;
//if (size > 1)
//	npcol = 2;
int nb = atoi(argv[4]); // number of blocks
int iam, nprocs, ictxt; 
int myrow, mycol; 
Cblacs_pinfo (&iam, &nprocs); 
Cblacs_get (-1, 0, &ictxt); 
Cblacs_gridinit (&ictxt, /*order*/"Col", nprow, npcol); 
Cblacs_gridinfo (ictxt, &nprow, &npcol, &myrow, &mycol); 

// Work only the process in the process grid 
if ((myrow>-1)&&(mycol>-1)&&(myrow<nprow)&&(mycol<npcol)) 
{ 
	//////////////////////////////////////////////////
	// Initialize Data
	//////////////////////////////////////////////////
	// Compute the size of the local matrices 
	int izero = 0; 
	int np = numroc_(&n , &nb, &myrow, &izero, &nprow); 
	int nq = numroc_(&n , &nb, &mycol, &izero, &npcol); 

	// Allocate and fill the matrices A and B 
	double * A = new double [np*nq]; 
	double * B = new double [np*nq]; 

	// Generate random A matrix 
	int k=0; 
	for (int i=0; i<np; i++) 
		for (int j=0; j<nq; j++) { 
//			A[k] = ((double) rand()) / ((double) RAND_MAX) - 0.5; 
//			B[k] = ((double) rand()) / ((double) RAND_MAX) - 0.5; 
			A[k] = k; 
			B[k] = 2*k; 
			k++; 
		} 

	// Initialize the array descriptor for the matrix A and B 
	int itemp = (np<1 ? 1 : np);
	int descA[9]; 
	int descB[9]; 
	int info; 
	descinit_(descA, &n, &n , &nb, &nb, &izero, &izero, &ictxt, &itemp, &info); 
	descinit_(descB, &n, &n, &nb, &nb, &izero, &izero, &ictxt, &itemp, &info); 

	//////////////////////////////////////////////////
	// Compute A*B
	//////////////////////////////////////////////////
        // Coefficients
        double coeff_one=1.0;
        double coeff_zero=0.0;
        int coeff_index=1;

	// Information On computation
	if (iam==0) 
	{ 
		cout << endl; 
		cout << "***********************************************" << endl; 
		cout << " Example of ScaLAPACK routine call: (PDGESV) " << endl; 
		cout << "***********************************************" << endl; 
		cout << endl; 
		cout << " n = " << n << endl; 
		cout << " grid = " << nprow << ", " << npcol << endl; 
		cout << " blocks = " << nb << "x" << nb << endl; 
		cout << " np = " << np << endl; 
		cout << " nq = " << nq << endl;
	}

        // Call PBLAS PDGEMM routine
	double MPIt1 = MPI::Wtime();
        clock_t begin = clock();
        pdgemm_ ("N", "N", &n, &n, &n, &coeff_one, A, &coeff_index, &coeff_index,
                descA, B, &coeff_index, &coeff_index, descB, &coeff_zero, B,
                &coeff_index, &coeff_index, descB);
        clock_t end = clock();
        double MPIt2 = MPI::Wtime();

	// Information On computation
	if (iam==0) 
	{ 
		cout << endl; 
                double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
                cout << "Total CPU time during (pdgemm) = " << time_spent << endl;
                cout << "Elapsed time during (pdgemm) = " << time_spent/size << endl;
                double new_n=1.0*n/1000;
                double gflops=1.0*new_n*new_n*(2*n+2)/(1000.0*time_spent);
                cout << "GFlop/s (pdgemm) = " << gflops/size << endl;
                cout << endl;
	        double MPIelapsed = MPIt2-MPIt1;
                cout << "MPI total CPU time during (pdgemm) = " << MPIelapsed << endl;
                cout << "MPI elapsed time during (pdgemm) = " << MPIelapsed/size << endl;
                gflops=1.0*new_n*new_n*(2*n+2)/(1000.0*MPIelapsed);
                cout << "MPI GFlop/s (pdgemm) = " << gflops/size << endl;
                cout << "Info returned by (pdgemm) = " << info << endl;
                cout << endl;
//double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
//cout << nprow << "\t" << npcol << "\t" << nb << "\t" << time_spent << "\n";
	} 

	//////////////////////////////////////////////////
	// Finalization
	//////////////////////////////////////////////////
	// Clean up 
	delete [] A; 
	delete [] B; 

	// Grid exit 
	Cblacs_gridexit(0); 
} 

if(iam==0) 
{ 
	cout << "***********************************************\n"; 
	cout << " END \n"; 
	cout << "***********************************************\n"; 
}

MPI::Finalize(); 

return 0; 
} 
