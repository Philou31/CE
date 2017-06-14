// STL 
#include <iostream> 
#include <cstdlib>

// MPI (must be before #define REAL) 
#include "mpi.h" 
#include <omp.h>

using std::cout; 
using std::endl; 

extern "C" // {{{ 
{ 
	void Cblacs_pinfo (int* mypnum, int* nprocs); 
	void Cblacs_get (int context, int request, int* value); 
	int Cblacs_gridinit (int* context, char * order, int np_row, int np_col); 
	void Cblacs_gridinfo (int context, int* np_row, int* np_col, int* my_row, int* my_col); 
	void Cblacs_gridexit (int context); 
	void Cblacs_exit (int error_code); 
	int numroc_ (int *n, int *nb, int *iproc, int *isrcproc, int *nprocs); 
	void descinit_ (int *desc, int *m, int *n, int *mb, int *nb, int *irsrc, int *icsrc, int *ictxt, int *lld, int *info); 
	double pdlamch_ (int *ictxt , char *cmach); 
	double pdlange_ (char *norm, int *m, int *n, double *A, int *ia, int *ja, int *desca, double *work); 
	void pdlacpy_ (char *uplo, int *m, int *n, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb); 
	void pdgesv_ (int *n, int *nrhs, double *A, int *ia, int *ja, int *desca, int* ipiv, double *B, int *ib, int *jb, int *descb, int *info); 
	void pdgemm_ (char *TRANSA, char *TRANSB, int * M, int * N, int * K, double * ALPHA, 
	double * A, int * IA, int * JA, int * DESCA, double * B, int * IB, int * JB, int * DESCB, 
	double * BETA, double * C, int * IC, int * JC, int * DESCC ); 
	int indxg2p_ (int *indxglob, int *nb, int *iproc, int *isrcproc, int *nprocs); 
} // }}} 

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
/*	cout << "MPI_proc\tSize\tCPU\tProcessor\n";
	cout << rank << "\t" << size << "\t" << sched_getcpu() << "\t" << processor_name << "\n";
	
	cout << "MPI_proc\tOMP_thread\tSize\tCPU\n";*/
}
// Info OpenMP
int nthreads;
#pragma omp parallel shared(nthreads)
{
	int tid = omp_get_thread_num();
	nthreads = omp_get_num_threads();
/*	#pragma omp critical
	{
		cout << rank << "\t" << tid << "\t" << nthreads << "\t" << sched_getcpu() << "\n";
	}*/
}
// MPI Synchro
if (size!=1) {
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
int nrhs = 1; // number of Right-Hand-Side columns 

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
	int nqrhs = numroc_(&nrhs, &nb, &mycol, &izero, &npcol); 

	// Allocate and fill the matrices A and B 
	double * A = new double [np*nq]; 
	double * Acpy = new double [np*nq]; 
	double * B = new double [np*nqrhs]; 
	double * X = new double [np*nqrhs]; 
	double * R = new double [np*nqrhs]; 
	int * ippiv = new int [np+nb]; 

	// Generate random A matrix 
	int k=0; 
	for (int i=0; i<np; i++) 
		for (int j=0; j<nq; j++) { 
//			A[k] = ((double) rand()) / ((double) RAND_MAX) - 0.5; 
			A[k] = k;
		}
	for (int i=0; i<np; i++) 
		for (int j=0; j<nqrhs; j++) { 
//			B[k] = ((double) rand()) / ((double) RAND_MAX) - 0.5; 
			B[k] = 2*k; 
			k++; 
		} 

	// Initialize the array descriptor for the matrix A and B 
	int itemp = (np<1 ? 1 : np);
	int descA[9]; 
	int descB[9]; 
	int info; 
	descinit_(descA, &n, &n , &nb, &nb, &izero, &izero, &ictxt, &itemp, &info); 
	descinit_(descB, &n, &nrhs, &nb, &nb, &izero, &izero, &ictxt, &itemp, &info); 

	// Make a copy of A and the rhs for checking purposes 
	int ione = 1; 
	pdlacpy_("All", &n, &n , A, &ione, &ione, descA, Acpy, &ione, &ione, descA); 
	pdlacpy_("All", &n, &nrhs, B, &ione, &ione, descB, X , &ione, &ione, descB); 

	//////////////////////////////////////////////////
	// Solve Ax=B
	//////////////////////////////////////////////////
	// Call ScaLAPACK PDGESV routine 
	double MPIt1 = MPI::Wtime(); 
	pdgesv_(&n, &nrhs, A, &ione, &ione, descA, ippiv, X, &ione, &ione, descB, &info); 
	double MPIt2 = MPI::Wtime(); 
	double MPIelapsed = MPIt2-MPIt1; 

	//////////////////////////////////////////////////
	// Display Metrics
	//////////////////////////////////////////////////
	//// Froebenius norm = ||A * X - B|| / ( ||X|| * ||A|| * eps * N ) 
	double mone=-1.0; 
	double pone= 1.0; 
	pdlacpy_("All", &n, &nrhs, B, &ione, &ione, descB, R, &ione, &ione, descB); 
	pdgemm_ ("N", "N", &n, &nrhs, &n, &pone, Acpy, &ione, &ione, descA, X, &ione, &ione, descB, &mone, R, &ione, &ione, descB); 
	double work = 0.0; 
	double eps = pdlamch_(&ictxt, "Epsilon"); 
	double AnormF = pdlange_("F", &n, &n , A, &ione, &ione, descA, &work); 
	double XnormF = pdlange_("F", &n, &nrhs, X, &ione, &ione, descB, &work); 
	double RnormF = pdlange_("F", &n, &nrhs, R, &ione, &ione, descB, &work); 
	double residF = RnormF/(AnormF*XnormF*eps*static_cast<double>(n));

	// Information On computation
	if (iam==0) 
	{ 
/*		cout << endl; 
		cout << "***********************************************" << endl; 
		cout << " Example of ScaLAPACK routine call: (PDGESV) " << endl; 
		cout << "***********************************************" << endl; 
		cout << endl; 
		cout << " n = " << n << endl; 
		cout << " nrhs = " << nrhs << endl; 
		cout << " grid = " << nprow << ", " << npcol << endl; 
		cout << " blocks = " << nb << "x" << nb << endl; 
		cout << " np = " << np << endl; 
		cout << " nq = " << nq << endl; 
		cout << " nqrhs = " << nqrhs << endl; 
		cout << endl; 
		cout << "MPI elapsed time during (pdgesv) = " << MPIelapsed << endl; 
		double new_n=1.0*n/1000;
		cout << "GFlop/s (pdgesv) = " << ((2.0*new_n*new_n*n/3.0) + 2.0*new_n*new_n)/(MPIelapsed*1000) << endl; 
		cout << "Info returned by (pdgesv) = " << info << endl; 
		cout << endl; 
		cout << "Froebenius norm (residual) = " << residF << endl; 
		cout << endl; */
cout << size << "\t" << nthreads << "\t" << nprow << "\t" << npcol << "\t" << nb << "\t" << MPIelapsed << "\n";
	} 

	//////////////////////////////////////////////////
	// Finalization
	//////////////////////////////////////////////////
	// Clean up 
	delete [] A; 
	delete [] Acpy; 
	delete [] B; 
	delete [] X; 
	delete [] R; 
	delete [] ippiv; 

	// Grid exit 
	Cblacs_gridexit(0); 
} 

if(iam==0) 
{ 
/*	cout << "***********************************************\n"; 
	cout << " END \n"; 
	cout << "***********************************************\n"; */
} 

MPI::Finalize(); 

return 0; 
} 
