#include <stdio.h>
#include <assert.h>

/* Struct linear solvers header */
#include "HYPRE_struct_ls.h"

#include "vis.c"

const double h = 1. / 32;

int main(int argc, char *argv[])
{
   int myid, num_procs;

   HYPRE_StructGrid     grid;
   HYPRE_StructStencil  stencil;
   HYPRE_StructMatrix   A;
   HYPRE_StructVector   b;
   HYPRE_StructVector   x;
   HYPRE_StructSolver   solver;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   /* Initialize HYPRE */
   HYPRE_Init();

   if (num_procs != 2)
   {
      if (myid == 0)
         printf("Must run with 2 processors!\n");
      MPI_Finalize();
      return 0;
   }

   /* 1. Set up a grid. Each processor describes the piece
      of the grid that it owns. */
   {
      HYPRE_StructGridCreate(MPI_COMM_WORLD, 2, &grid);

      if (myid == 0)
      {
         int ilower[2] = {1, 1}, iupper[2] = {16, 32};
         HYPRE_StructGridSetExtents(grid, ilower, iupper);
      }
      else if (myid == 1)
      {
         int ilower[2] = {17, 1}, iupper[2] = {32, 32};
         HYPRE_StructGridSetExtents(grid, ilower, iupper);
      }

      HYPRE_StructGridAssemble(grid);
   }

   /* 2. Define the discretization stencil */
   {
      HYPRE_StructStencilCreate(2, 5, &stencil);

      /* Stencil offset: center, up, right, down, left */
      int offsets[5][2] = {{0, 0}, {0, 1}, {1, 0}, {0, -1}, {-1, 0}};

      for (int i = 0; i < 5; i++)
         HYPRE_StructStencilSetElement(stencil, i, offsets[i]);
   }

   /* 3. Set up a Struct Matrix */
   {
      HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, &A);
      HYPRE_StructMatrixInitialize(A);

      if (myid == 0)
      {
         int ilower[2] = {1, 1}, iupper[2] = {16, 32};
         int stencil_indices[5] = {0, 1, 2, 3, 4};
         double values[2560]; /* 512 grid points, each with 5 stencil entries */

         for (int i = 0; i < 2560; i += 5)
         {
            values[i] = 4.;
            for (int j = 1; j < 5; j++)
               values[i+j] = -1.;
         }
         HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 5, stencil_indices, values);
      }
      else if (myid == 1)
      {
         int ilower[2] = {17, 1}, iupper[2] = {32, 32};
         int stencil_indices[5] = {0, 1, 2, 3, 4};
         double values[2560]; /* 512 grid points, each with 5 stencil entries */

         for (int i = 0; i < 2560; i += 5)
         {
            values[i] = 4.;
            for (int j = 1; j < 5; j++)
               values[i+j] = -1.;
         }
         HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 5, stencil_indices, values);
      }
      /* Set the coefficients of boundary grid points */
      if (myid == 0)
      {
         double values[64];
         for (int i = 0; i < 64; i += 2)
         {
            values[i] = 5.;
            values[i+1] = 0.;
         }
         {
            /* upper boundary */
            int ilower[2] = {1, 32}, iupper[2] = {16, 32};
            int stencil_indices[2] = {0, 1};
            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 2, stencil_indices, values);
         }
         {
            /* lower boundary */
            int ilower[2] = {1, 1}, iupper[2] = {16, 1};
            int stencil_indices[2] = {0, 3};
            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 2, stencil_indices, values);
         }
         {
            /* left boundary */
            int ilower[2] = {1, 1}, iupper[2] = {1, 32};
            int stencil_indices[2] = {0, 4};
            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 2, stencil_indices, values);
         }
         values[0] = 6.;
         {
            /* upper left corner */
            int i[2] = {1, 32};
            int stencil_indices[1] = {0};
            HYPRE_StructMatrixSetBoxValues(A, i, i, 1, stencil_indices, values);
         }
         {
            /* lower left corner */
            int i[2] = {1, 1};
            int stencil_indices[1] = {0};
            HYPRE_StructMatrixSetBoxValues(A, i, i, 1, stencil_indices, values);
         }
      }
      else if (myid == 1)
      {
         double values[64];
         for (int i = 0; i < 64; i += 2)
         {
            values[i] = 5.;
            values[i+1] = 0.;
         }
         {
            /* upper boundary */
            int ilower[2] = {17, 32}, iupper[2] = {32, 32};
            int stencil_indices[2] = {0, 1};
            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 2, stencil_indices, values);
         }
         {
            /* right boundary */
            int ilower[2] = {32, 1}, iupper[2] = {32, 32};
            int stencil_indices[2] = {0, 2};
            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 2, stencil_indices, values);
         }
         {
            /* lower boundary */
            int ilower[2] = {17, 1}, iupper[2] = {32, 1};
            int stencil_indices[2] = {0, 3};
            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 2, stencil_indices, values);
         }
         values[0] = 6.;
         {
            /* upper right corner */
            int i[2] = {32, 32};
            int stencil_indices[1] = {0};
            HYPRE_StructMatrixSetBoxValues(A, i, i, 1, stencil_indices, values);
         }
         {
            /* lower right corner */
            int i[2] = {32, 1};
            int stencil_indices[1] = {0};
            HYPRE_StructMatrixSetBoxValues(A, i, i, 1, stencil_indices, values);
         }
      }

      HYPRE_StructMatrixAssemble(A);
   }

   /* 4. Set up Struct Vectors for b and x.  Each processor sets the vectors
      corresponding to its boxes. */
   {
      HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &b);
      HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &x);
      HYPRE_StructVectorInitialize(b);
      HYPRE_StructVectorInitialize(x);

      /* Set the vector coefficients */
      if (myid == 0)
      {
         int ilower[2] = {1, 1}, iupper[2] = {16, 32};
         double values[512]; /* 512 grid points */

         for (int i = 0; i < 512; i++)
            values[i] = h*h;
         HYPRE_StructVectorSetBoxValues(b, ilower, iupper, values);

         for (int i = 0; i < 512; i++)
            values[i] = 0.;
         HYPRE_StructVectorSetBoxValues(x, ilower, iupper, values);
      }
      else if (myid == 1)
      {
         int ilower[2] = {17, 1}, iupper[2] = {32, 32};
         double values[512]; /* 32 grid points */

         for (int i = 0; i < 512; i++)
            values[i] = h*h;
         HYPRE_StructVectorSetBoxValues(b, ilower, iupper, values);

         for (int i = 0; i < 512; i++)
            values[i] = 1.;
         HYPRE_StructVectorSetBoxValues(x, ilower, iupper, values);
      }

      HYPRE_StructVectorAssemble(b);
      HYPRE_StructVectorAssemble(x);
   }

   /* 5. Set up and use a solver (See the Reference Manual for descriptions
      of all of the options.) */
   {
      /* Create an empty PCG Struct solver */
      HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver);

      /* Set some parameters */
      HYPRE_StructPCGSetTol(solver, 1.e-6); /* convergence tolerance */
      HYPRE_StructPCGSetPrintLevel(solver, 1); /* amount of info. printed */

      /* Setup and solve */
      HYPRE_StructPCGSetup(solver, A, b, x);
      HYPRE_StructPCGSolve(solver, A, b, x);
   }

   /* GLVis */
   GLVis_PrintStructGrid(grid, "vis/test.mesh", myid, NULL, NULL);
   GLVis_PrintStructVector(x, "vis/test.sol", myid);
   GLVis_PrintData("vis/test.data", myid, num_procs);

   /* Free memory */
   HYPRE_StructGridDestroy(grid);
   HYPRE_StructStencilDestroy(stencil);
   HYPRE_StructMatrixDestroy(A);
   HYPRE_StructVectorDestroy(b);
   HYPRE_StructVectorDestroy(x);
   HYPRE_StructPCGDestroy(solver);

   /* Finalize Hypre */
   HYPRE_Finalize();

   /* Finalize MPI */
   MPI_Finalize();

   return 0;
}