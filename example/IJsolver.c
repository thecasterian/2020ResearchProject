/* Solve the following linear equation:
   [  1  2  5  1 |  0 ]
   [  3 -4  0 -2 |  5 ]
   [  4  0  2 -1 |  1 ]
   [  1 -2 -4 -3 | -4 ] */

#include <stdio.h>

#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"
#include "_hypre_utilities.h"

int main(int argc, char *argv[]) {
    int rank, num_process;

    int ilower, iupper;

    HYPRE_IJMatrix A;
    HYPRE_ParCSRMatrix parcsr_A;
    HYPRE_IJVector b;
    HYPRE_ParVector par_b;
    HYPRE_IJVector x;
    HYPRE_ParVector par_x;

    HYPRE_Solver solver;

    int num_iteration;
    double final_res;

    /* Initialize. */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_process);

    HYPRE_Init();

    /* Must run with two processes. */
    if (num_process != 2 && rank == 0) {
        printf("Must run with two processes!\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    /* Each process has only two rows: process 0 has first and second row;
       process 1 has third and forth row. */
    if (rank == 0) {
        ilower = 1;
        iupper = 2;
    }
    else {
        ilower = 3;
        iupper = 4;
    }

    /* Create the matrix. Since the given linear equation has a square (4x4)
       matrix, jlower and jupper should be identical to ilower and iupper,
       respectively. */
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A);
    HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(A);

    /* Set the elements. */
    if (rank == 0) {
        /* Process 0 has two rows. */
        int nrows = 2;
        /* Process 0 has the first and second row. */
        int rows[2] = {1, 2};
        /* The first row has 4 non-zero elements while the second has only 3. */
        int ncols[2] = {4, 3};
        /* The first four entries of `cols` are the column number of non-zero
           elements in first row. The later three entries are the column number
           of non-zero elements in second row.  */
        int cols[7] = {1, 2, 3, 4, 1, 2, 4};
        /* The value of non-zero elements in the first row and second row. */
        double values[7] = {1, 2, 5, 1, 3, -4, -2};

        HYPRE_IJMatrixSetValues(A, nrows, ncols, rows, cols, values);
    }
    else {
        int nrows = 2;
        int rows[2] = {3, 4};
        int ncols[2] = {3, 4};
        int cols[7] = {1, 3, 4, 1, 2, 3, 4};
        double values[7] = {4, 2, -1, 1, -2, -4, -3};
        HYPRE_IJMatrixSetValues(A, nrows, ncols, rows, cols, values);
    }

    /* Assemble after setting the elements. */
    HYPRE_IJMatrixAssemble(A);

    /* Get the parcsr matrix object. */
    HYPRE_IJMatrixGetObject(A, (void **)&parcsr_A);

    /* Create the vectors: RHS vector and solution vector. Like rows of matrix,
       each process has only two rows of vectors. */
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &b);
    HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(b);

    HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &x);
    HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(x);

    /* Set the elements. Solution vector will be set zero vector. */
    if (rank == 0) {
        /* Process 0 has two rows. */
        int nrows = 2;
        /* Process 0 has the first and second row. */
        int rows[2] = {1, 2};
        /* Value of elements in the first and second row. */
        double b_values[2] = {0, 5};
        double x_values[2] = {0, 0};

        HYPRE_IJVectorSetValues(b, nrows, rows, b_values);
        HYPRE_IJVectorSetValues(x, nrows, rows, x_values);
    }
    else {
        int nrows = 2;
        int rows[2] = {3, 4};
        double b_values[2] = {1, -4};
        double x_values[2] = {0, 0};

        HYPRE_IJVectorSetValues(b, nrows, rows, b_values);
        HYPRE_IJVectorSetValues(x, nrows, rows, x_values);
    }

    /* Assemble and get the par vector object. */
    HYPRE_IJVectorAssemble(b);
    HYPRE_IJVectorAssemble(x);
    HYPRE_IJVectorGetObject(b, (void **)&par_b);
    HYPRE_IJVectorGetObject(x, (void **)&par_x);

    /* Create solver. We will use BiCGSTAB solver here. */
    HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, &solver);

    /* Set some solver parameters. */
    HYPRE_BiCGSTABSetMaxIter(solver, 100);      /* Maximum iterations. */
    HYPRE_BiCGSTABSetTol(solver, 1e-6);         /* Convergence criteria. */
    HYPRE_ParCSRBiCGSTABSetLogging(solver, 1);  /* Log run info. */
    HYPRE_BiCGSTABSetPrintLevel(solver, 2);     /* Print info every iteraion. */

    /* Setup and solve. */
    HYPRE_ParCSRBiCGSTABSetup(solver, parcsr_A, par_b, par_x);
    HYPRE_ParCSRBiCGSTABSolve(solver, parcsr_A, par_b, par_x);

    /* Get run info. */
    HYPRE_BiCGSTABGetNumIterations(solver, &num_iteration);
    HYPRE_BiCGSTABGetFinalRelativeResidualNorm(solver, &final_res);
    if (rank == 0) {
        printf("Iterations = %d\n", num_iteration);
        printf("Final Relative Residual Norm = %e\n", final_res);
    }

    /* Clean up. */
    HYPRE_IJMatrixDestroy(A);
    HYPRE_IJVectorDestroy(b);
    HYPRE_IJVectorDestroy(x);
    HYPRE_ParCSRBiCGSTABDestroy(solver);

    /* Finalize. */
    HYPRE_Finalize();
    MPI_Finalize();

    return 0;
}
