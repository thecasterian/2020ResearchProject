#include "ibm3d.h"

#ifndef __USE_POSIX199309
#define __USE_POSIX199309
#endif
#include <time.h>
#include <math.h>

#include "utils.h"

static inline void calc_N(IBMSolver *);
static inline void calc_u_star(IBMSolver *, double *, double *, double *);
static inline void calc_u_tilde(IBMSolver *);
static inline void calc_U_star(IBMSolver *);
static inline void calc_p_prime(IBMSolver *, double *);
static inline void update_next(IBMSolver *);

static void interp_stag_vel(IBMSolver *);
static void autosave(IBMSolver *);
static void update_bc(IBMSolver *);
static void adj_exchange(IBMSolver *);
static void update_ghost_and_solid(IBMSolver *);

void IBMSolver_iterate(IBMSolver *solver, int num_time_steps, bool verbose) {
    struct timespec t_start, t_end;
    long elapsed_time, hour, min, sec;
    double final_norm_u1, final_norm_u2, final_norm_u3, final_norm_p;

    update_bc(solver);
    adj_exchange(solver);
    interp_stag_vel(solver);
    update_ghost_and_solid(solver);

    calc_N(solver);
    SWAP(solver->N1, solver->N1_prev);
    SWAP(solver->N2, solver->N2_prev);
    SWAP(solver->N3, solver->N3_prev);

    if (verbose) {
        clock_gettime(CLOCK_REALTIME, &t_start);
    }

    while (solver->iter < num_time_steps) {
        calc_N(solver);
        calc_u_star(solver, &final_norm_u1, &final_norm_u2, &final_norm_u3);
        calc_u_tilde(solver);
        calc_U_star(solver);
        calc_p_prime(solver, &final_norm_p);
        update_next(solver);

        solver->iter++;

        /* Print iteration results. */
        if (verbose && solver->rank == 0) {
            clock_gettime(CLOCK_REALTIME, &t_end);
            elapsed_time = (t_end.tv_sec*1000+t_end.tv_nsec/1000000)
                - (t_start.tv_sec*1000+t_start.tv_nsec/1000000);

            if (solver->iter % 10 == 1) {
                printf("\n  iter       u1 res       u2 res       u3 res        p res       time\n");
                printf("---------------------------------------------------------------------\n");
            }

            hour = elapsed_time / 3600000;
            min = elapsed_time / 60000 % 60;
            sec = elapsed_time / 1000 % 60 + (elapsed_time % 1000 > 500);
            if (sec >= 60) {
                sec -= 60;
                min += 1;
            }
            if (min >= 60) {
                min -= 60;
                hour += 1;
            }

            printf(
                "%6d   %10.4e   %10.4e   %10.4e   %10.4e   %02ld:%02ld:%02ld\n",
                solver->iter,
                final_norm_u1, final_norm_u2, final_norm_u3, final_norm_p,
                hour, min, sec
            );
        }

        /* Autosave. */
        if (solver->autosave_period != 0 && solver->iter % solver->autosave_period == 0) {
            if (solver->rank == 0) {
                printf("\nAutosave...\n\n");
            }
            autosave(solver);
        }
    }
}

static inline void calc_N(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    const double *const dx = solver->dx;
    const double *const dy = solver->dy;
    const double *const dz = solver->dz;

    const int (*const flag)[Ny+2][Nz+2] = solver->flag;

    const double (*const u1)[Ny+2][Nz+2] = solver->u1;
    const double (*const u2)[Ny+2][Nz+2] = solver->u2;
    const double (*const u3)[Ny+2][Nz+2] = solver->u3;

    const double (*const U1)[Ny+2][Nz+2] = solver->U1;
    const double (*const U2)[Ny+1][Nz+2] = solver->U2;
    const double (*const U3)[Ny+2][Nz+1] = solver->U3;

    double (*const N1)[Ny+2][Nz+2] = solver->N1;
    double (*const N2)[Ny+2][Nz+2] = solver->N2;
    double (*const N3)[Ny+2][Nz+2] = solver->N3;

    double u1_w, u1_e, u1_s, u1_n, u1_d, u1_u;
    double u2_w, u2_e, u2_s, u2_n, u2_d, u2_u;
    double u3_w, u3_e, u3_s, u3_n, u3_d, u3_u;

    FOR_ALL_CELL (i, j, k) {
        if (flag[i][j][k] == FLAG_FLUID) {
            u1_w = (u1[i-1][j][k]*dx[i]+u1[i][j][k]*dx[i-1]) / (dx[i-1]+dx[i]);
            u2_w = (u2[i-1][j][k]*dx[i]+u2[i][j][k]*dx[i-1]) / (dx[i-1]+dx[i]);
            u3_w = (u3[i-1][j][k]*dx[i]+u3[i][j][k]*dx[i-1]) / (dx[i-1]+dx[i]);

            u1_e = (u1[i][j][k]*dx[i+1]+u1[i+1][j][k]*dx[i]) / (dx[i]+dx[i+1]);
            u2_e = (u2[i][j][k]*dx[i+1]+u2[i+1][j][k]*dx[i]) / (dx[i]+dx[i+1]);
            u3_e = (u3[i][j][k]*dx[i+1]+u3[i+1][j][k]*dx[i]) / (dx[i]+dx[i+1]);

            u1_s = (u1[i][j-1][k]*dy[j]+u1[i][j][k]*dy[j-1]) / (dy[j-1]+dy[j]);
            u2_s = (u2[i][j-1][k]*dy[j]+u2[i][j][k]*dy[j-1]) / (dy[j-1]+dy[j]);
            u3_s = (u3[i][j-1][k]*dy[j]+u3[i][j][k]*dy[j-1]) / (dy[j-1]+dy[j]);

            u1_n = (u1[i][j][k]*dy[j+1]+u1[i][j+1][k]*dy[j]) / (dy[j]+dy[j+1]);
            u2_n = (u2[i][j][k]*dy[j+1]+u2[i][j+1][k]*dy[j]) / (dy[j]+dy[j+1]);
            u3_n = (u3[i][j][k]*dy[j+1]+u3[i][j+1][k]*dy[j]) / (dy[j]+dy[j+1]);

            u1_d = (u1[i][j][k-1]*dz[k]+u1[i][j][k]*dz[k-1]) / (dz[k-1]+dz[k]);
            u2_d = (u2[i][j][k-1]*dz[k]+u2[i][j][k]*dz[k-1]) / (dz[k-1]+dz[k]);
            u3_d = (u3[i][j][k-1]*dz[k]+u3[i][j][k]*dz[k-1]) / (dz[k-1]+dz[k]);

            u1_u = (u1[i][j][k]*dz[k+1]+u1[i][j][k+1]*dz[k]) / (dz[k]+dz[k+1]);
            u2_u = (u2[i][j][k]*dz[k+1]+u2[i][j][k+1]*dz[k]) / (dz[k]+dz[k+1]);
            u3_u = (u3[i][j][k]*dz[k+1]+u3[i][j][k+1]*dz[k]) / (dz[k]+dz[k+1]);

            /* Ni = d(U1ui)/dx + d(U2ui)/dy + d(U3ui)/dz */
            N1[i][j][k] = (U1[i][j][k]*u1_e-U1[i-1][j][k]*u1_w) / dx[i]
                + (U2[i][j][k]*u1_n-U2[i][j-1][k]*u1_s) / dy[j]
                + (U3[i][j][k]*u1_u-U3[i][j][k-1]*u1_d) / dz[k];
            N2[i][j][k] = (U1[i][j][k]*u2_e-U1[i-1][j][k]*u2_w) / dx[i]
                + (U2[i][j][k]*u2_n-U2[i][j-1][k]*u2_s) / dy[j]
                + (U3[i][j][k]*u2_u-U3[i][j][k-1]*u2_d) / dz[k];
            N3[i][j][k] = (U1[i][j][k]*u3_e-U1[i-1][j][k]*u3_w) / dx[i]
                + (U2[i][j][k]*u3_n-U2[i][j-1][k]*u3_s) / dy[j]
                + (U3[i][j][k]*u3_u-U3[i][j][k-1]*u3_d) / dz[k];
        }
        else {
            N1[i][j][k] = N2[i][j][k] = N3[i][j][k] = NAN;
        }
    }
}

static inline void calc_u_star(
    IBMSolver *solver,
    double *final_norm_u1, double *final_norm_u2, double *final_norm_u3
) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;
    const int Nx_global = solver->Nx_global;
    const double dt = solver->dt;

    const double *const xc = solver->xc;
    const double *const yc = solver->yc;
    const double *const zc = solver->zc;

    const double *const kx_W = solver->kx_W;
    const double *const kx_E = solver->kx_E;
    const double *const ky_S = solver->ky_S;
    const double *const ky_N = solver->ky_N;
    const double *const kz_D = solver->kz_D;
    const double *const kz_U = solver->kz_U;

    const int (*const flag)[Ny+2][Nz+2] = solver->flag;

    const double (*const u1)[Ny+2][Nz+2] = solver->u1;
    const double (*const u2)[Ny+2][Nz+2] = solver->u2;
    const double (*const u3)[Ny+2][Nz+2] = solver->u3;

    double (*const u1_star)[Ny+2][Nz+2] = solver->u1_star;
    double (*const u2_star)[Ny+2][Nz+2] = solver->u2_star;
    double (*const u3_star)[Ny+2][Nz+2] = solver->u3_star;

    const double (*const p)[Ny+2][Nz+2] = solver->p;

    const double (*const N1)[Ny+2][Nz+2] = solver->N1;
    const double (*const N2)[Ny+2][Nz+2] = solver->N2;
    const double (*const N3)[Ny+2][Nz+2] = solver->N3;
    const double (*const N1_prev)[Ny+2][Nz+2] = solver->N1_prev;
    const double (*const N2_prev)[Ny+2][Nz+2] = solver->N2_prev;
    const double (*const N3_prev)[Ny+2][Nz+2] = solver->N3_prev;

    int hypre_ierr = 0;

    /* u1_star. */
    FOR_ALL_CELL (i, j, k) {
        double *const rhs = &solver->vector_values[LOCL_CELL_IDX(i, j, k)-1];
        if (flag[i][j][k] == FLAG_FLUID) {
            *rhs = -dt/2 * (3*N1[i][j][k] - N1_prev[i][j][k])
                - dt * (p[i+1][j][k] - p[i-1][j][k]) / (xc[i+1] - xc[i-1])
                + (1-kx_W[i]-kx_E[i]-ky_S[j]-ky_N[j]-kz_D[k]-kz_U[k]) * u1[i][j][k]
                + kx_W[i]*u1[i-1][j][k] + kx_E[i]*u1[i+1][j][k]
                + ky_S[j]*u1[i][j-1][k] + ky_N[j]*u1[i][j+1][k]
                + kz_D[k]*u1[i][j][k-1] + kz_U[k]*u1[i][j][k+1];

            /* West. */
            if (LOCL_TO_GLOB(i) == 1) {
                switch (solver->bc_type[3]) {
                case BC_VELOCITY_INLET:
                    *rhs += 2*solver->bc_val[3]*kx_W[i];
                    break;
                default:;
                }
            }

            /* East. */
            if (LOCL_TO_GLOB(i) == Nx_global) {
                switch (solver->bc_type[1]) {
                case BC_VELOCITY_INLET:
                    *rhs += 2*solver->bc_val[1]*kx_E[i];
                default:;
                }
            }

            /* South. */
            if (j == 1) {
                switch (solver->bc_type[2]) {
                default:;
                }
            }

            /* North. */
            if (j == Ny) {
                switch (solver->bc_type[0]) {
                default:;
                }
            }

            /* Down. */
            if (k == 1) {
                switch (solver->bc_type[4]) {
                default:;
                }
            }

            /* Up. */
            if (k == Nz) {
                switch (solver->bc_type[5]) {
                default:;
                }
            }
        }
        else {
            *rhs = 0;
        }
    }
    HYPRE_IJVectorSetValues(solver->b, Nx*Ny*Nz, solver->vector_rows, solver->vector_values);
    HYPRE_IJVectorSetValues(solver->x, Nx*Ny*Nz, solver->vector_rows, solver->vector_zeros);
    HYPRE_IJVectorAssemble(solver->b);
    HYPRE_IJVectorAssemble(solver->x);
    HYPRE_IJVectorGetObject(solver->b, (void **)&solver->par_b);
    HYPRE_IJVectorGetObject(solver->x, (void **)&solver->par_x);

    HYPRE_ParCSRBiCGSTABSetup(solver->linear_solver, solver->parcsr_A_u1, solver->par_b, solver->par_x);
    hypre_ierr = HYPRE_ParCSRBiCGSTABSolve(solver->linear_solver, solver->parcsr_A_u1, solver->par_b, solver->par_x);
    if (hypre_ierr) {
        printf("error: floating pointer error raised in u1_star\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    HYPRE_IJVectorGetValues(solver->x, Nx*Ny*Nz, solver->vector_rows, solver->vector_res);
    FOR_ALL_CELL (i, j, k) {
        u1_star[i][j][k] = solver->vector_res[LOCL_CELL_IDX(i, j, k)-1];
    }
    HYPRE_BiCGSTABGetFinalRelativeResidualNorm(solver->linear_solver, final_norm_u1);

    /* u2_star. */
    FOR_ALL_CELL (i, j, k) {
        double *const rhs = &solver->vector_values[LOCL_CELL_IDX(i, j, k)-1];
        if (flag[i][j][k] == FLAG_FLUID) {
            *rhs = -dt/2 * (3*N2[i][j][k] - N2_prev[i][j][k])
                - dt * (p[i][j+1][k] - p[i][j-1][k]) / (yc[j+1] - yc[j-1])
                + (1-kx_W[i]-kx_E[i]-ky_S[j]-ky_N[j]-kz_D[k]-kz_U[k]) * u2[i][j][k]
                + kx_W[i]*u2[i-1][j][k] + kx_E[i]*u2[i+1][j][k]
                + ky_S[j]*u2[i][j-1][k] + ky_N[j]*u2[i][j+1][k]
                + kz_D[k]*u2[i][j][k-1] + kz_U[k]*u2[i][j][k+1];

            /* West. */
            if (LOCL_TO_GLOB(i) == 1) {
                switch (solver->bc_type[3]) {
                default:;
                }
            }

            /* East. */
            if (LOCL_TO_GLOB(i) == Nx_global) {
                switch (solver->bc_type[1]) {
                default:;
                }
            }

            /* South. */
            if (j == 1) {
                switch (solver->bc_type[2]) {
                case BC_VELOCITY_INLET:
                    *rhs += 2*solver->bc_val[2]*ky_S[j];
                    break;
                default:;
                }
            }

            /* North. */
            if (j == Ny) {
                switch (solver->bc_type[0]) {
                case BC_VELOCITY_INLET:
                    *rhs += 2*solver->bc_val[0]*ky_N[j];
                    break;
                default:;
                }
            }

            /* Down. */
            if (k == 1) {
                switch (solver->bc_type[4]) {
                default:;
                }
            }

            /* Up. */
            if (k == Nz) {
                switch (solver->bc_type[5]) {
                default:;
                }
            }
        }
        else {
            *rhs = 0;
        }
    }
    HYPRE_IJVectorSetValues(solver->b, Nx*Ny*Nz, solver->vector_rows, solver->vector_values);
    HYPRE_IJVectorSetValues(solver->x, Nx*Ny*Nz, solver->vector_rows, solver->vector_zeros);
    HYPRE_IJVectorAssemble(solver->b);
    HYPRE_IJVectorAssemble(solver->x);
    HYPRE_IJVectorGetObject(solver->b, (void **)&solver->par_b);
    HYPRE_IJVectorGetObject(solver->x, (void **)&solver->par_x);

    HYPRE_ParCSRBiCGSTABSetup(solver->linear_solver, solver->parcsr_A_u2, solver->par_b, solver->par_x);
    hypre_ierr = HYPRE_ParCSRBiCGSTABSolve(solver->linear_solver, solver->parcsr_A_u2, solver->par_b, solver->par_x);
    if (hypre_ierr) {
        printf("error: floating pointer error raised in u2_star\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    HYPRE_IJVectorGetValues(solver->x, Nx*Ny*Nz, solver->vector_rows, solver->vector_res);
    FOR_ALL_CELL (i, j, k) {
        u2_star[i][j][k] = solver->vector_res[LOCL_CELL_IDX(i, j, k)-1];
    }
    HYPRE_BiCGSTABGetFinalRelativeResidualNorm(solver->linear_solver, final_norm_u2);

    /* u3_star. */
    FOR_ALL_CELL (i, j, k) {
        double *const rhs = &solver->vector_values[LOCL_CELL_IDX(i, j, k)-1];
        if (flag[i][j][k] == FLAG_FLUID) {
            *rhs = -dt/2 * (3*N3[i][j][k] - N3_prev[i][j][k])
                - dt * (p[i][j][k+1] - p[i][j][k-1]) / (zc[k+1] - zc[k-1])
                + (1-kx_W[i]-kx_E[i]-ky_S[j]-ky_N[j]-kz_D[k]-kz_U[k]) * u3[i][j][k]
                + kx_W[i]*u3[i-1][j][k] + kx_E[i]*u3[i+1][j][k]
                + ky_S[j]*u3[i][j-1][k] + ky_N[j]*u3[i][j+1][k]
                + kz_D[k]*u3[i][j][k-1] + kz_U[k]*u3[i][j][k+1];

            /* West. */
            if (LOCL_TO_GLOB(i) == 1) {
                switch (solver->bc_type[3]) {
                default:;
                }
            }

            /* East. */
            if (LOCL_TO_GLOB(i) == Nx_global) {
                switch (solver->bc_type[1]) {
                default:;
                }
            }

            /* South. */
            if (j == 1) {
                switch (solver->bc_type[2]) {
                default:;
                }
            }

            /* North. */
            if (j == Ny) {
                switch (solver->bc_type[0]) {
                default:;
                }
            }

            /* Down. */
            if (k == 1) {
                switch (solver->bc_type[4]) {
                case BC_VELOCITY_INLET:
                    *rhs += 2*solver->bc_val[4]*kz_D[k];
                    break;
                default:;
                }
            }

            /* Up. */
            if (k == Nz) {
                switch (solver->bc_type[5]) {
                case BC_VELOCITY_INLET:
                    *rhs += 2*solver->bc_val[5]*kz_U[k];
                    break;
                default:;
                }
            }
        }
        else {
            *rhs = 0;
        }
    }
    HYPRE_IJVectorSetValues(solver->b, Nx*Ny*Nz, solver->vector_rows, solver->vector_values);
    HYPRE_IJVectorSetValues(solver->x, Nx*Ny*Nz, solver->vector_rows, solver->vector_zeros);
    HYPRE_IJVectorAssemble(solver->b);
    HYPRE_IJVectorAssemble(solver->x);
    HYPRE_IJVectorGetObject(solver->b, (void **)&solver->par_b);
    HYPRE_IJVectorGetObject(solver->x, (void **)&solver->par_x);

    HYPRE_ParCSRBiCGSTABSetup(solver->linear_solver, solver->parcsr_A_u3, solver->par_b, solver->par_x);
    hypre_ierr = HYPRE_ParCSRBiCGSTABSolve(solver->linear_solver, solver->parcsr_A_u3, solver->par_b, solver->par_x);
    if (hypre_ierr) {
        printf("error: floating pointer error raised in u3_star\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    HYPRE_IJVectorGetValues(solver->x, Nx*Ny*Nz, solver->vector_rows, solver->vector_res);
    FOR_ALL_CELL (i, j, k) {
        u3_star[i][j][k] = solver->vector_res[LOCL_CELL_IDX(i, j, k)-1];
    }
    HYPRE_BiCGSTABGetFinalRelativeResidualNorm(solver->linear_solver, final_norm_u3);

    /* Exchange u_star for boundary processes. */
    if (solver->num_process > 1) {
        if (solver->rank == 0) {
            /* Receive from next process. */
            MPI_Recv(u1_star[Nx+1], (Ny+2)*(Nz+2), MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(u2_star[Nx+1], (Ny+2)*(Nz+2), MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(u3_star[Nx+1], (Ny+2)*(Nz+2), MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else if (solver->rank == 1) {
            /* Send to previous process. */
            MPI_Send(u1_star[1], (Ny+2)*(Nz+2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            MPI_Send(u2_star[1], (Ny+2)*(Nz+2), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            MPI_Send(u3_star[1], (Ny+2)*(Nz+2), MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
        }

        if (solver->rank == solver->num_process-1) {
            /* Receive from previous process. */
            MPI_Recv(u1_star[0], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(u2_star[0], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-2, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(u3_star[0], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-2, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else if (solver->rank == solver->num_process-2) {
            /* Send to next process. */
            MPI_Send(u1_star[Nx], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-1, 0, MPI_COMM_WORLD);
            MPI_Send(u2_star[Nx], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-1, 1, MPI_COMM_WORLD);
            MPI_Send(u3_star[Nx], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-1, 2, MPI_COMM_WORLD);
        }
    }

    /* Boundary condition. */

    /* West. */
    if (solver->ilower == 1) {
        switch (solver->bc_type[3]) {
        case BC_PRESSURE_OUTLET:
            for (int j = 1; j <= Ny; j++) {
                for (int k = 1; k <= Nz; k++) {
                    u1_star[0][j][k] = (u1_star[1][j][k]*(xc[2]-xc[0])-u1_star[2][j][k]*(xc[1]-xc[0])) / (xc[2]-xc[1]);
                    u2_star[0][j][k] = (u2_star[1][j][k]*(xc[2]-xc[0])-u2_star[2][j][k]*(xc[1]-xc[0])) / (xc[2]-xc[1]);
                    u3_star[0][j][k] = (u3_star[1][j][k]*(xc[2]-xc[0])-u3_star[2][j][k]*(xc[1]-xc[0])) / (xc[2]-xc[1]);
                }
            }
            break;
        default:;
        }
    }

    /* East. */
    if (solver->iupper == Nx_global) {
        switch (solver->bc_type[1]) {
        case BC_PRESSURE_OUTLET:
            for (int j = 1; j <= Ny; j++) {
                for (int k = 1; k <= Nz; k++) {
                    u1_star[Nx+1][j][k] = (u1_star[Nx][j][k]*(xc[Nx+1]-xc[Nx-1])-u1_star[Nx-1][j][k]*(xc[Nx+1]-xc[Nx])) / (xc[Nx]-xc[Nx-1]);
                    u2_star[Nx+1][j][k] = (u2_star[Nx][j][k]*(xc[Nx+1]-xc[Nx-1])-u2_star[Nx-1][j][k]*(xc[Nx+1]-xc[Nx])) / (xc[Nx]-xc[Nx-1]);
                    u3_star[Nx+1][j][k] = (u3_star[Nx][j][k]*(xc[Nx+1]-xc[Nx-1])-u3_star[Nx-1][j][k]*(xc[Nx+1]-xc[Nx])) / (xc[Nx]-xc[Nx-1]);
                }
            }
            break;
        default:;
        }
    }

    /* South. */
    switch (solver->bc_type[2]) {
    case BC_PRESSURE_OUTLET:
        for (int i = 0; i <= Nx+1; i++) {
            for (int k = 1; k <= Nz; k++) {
                u1_star[i][0][k] = (u1_star[i][1][k]*(yc[2]-yc[0])-u1_star[i][2][k]*(yc[1]-yc[0])) / (yc[2]-yc[1]);
                u2_star[i][0][k] = (u2_star[i][1][k]*(yc[2]-yc[0])-u2_star[i][2][k]*(yc[1]-yc[0])) / (yc[2]-yc[1]);
                u3_star[i][0][k] = (u3_star[i][1][k]*(yc[2]-yc[0])-u3_star[i][2][k]*(yc[1]-yc[0])) / (yc[2]-yc[1]);
            }
        }
        break;
    default:;
    }

    /* North. */
    switch (solver->bc_type[0]) {
    case BC_PRESSURE_OUTLET:
        for (int i = 0; i <= Nx+1; i++) {
            for (int k = 1; k <= Nz; k++) {
                u1_star[i][Ny+1][k] = (u1_star[i][Ny][k]*(yc[Ny+1]-yc[Ny-1])-u1_star[i][Ny-1][k]*(yc[Ny+1]-yc[Ny])) / (yc[Ny]-yc[Ny-1]);
                u2_star[i][Ny+1][k] = (u2_star[i][Ny][k]*(yc[Ny+1]-yc[Ny-1])-u2_star[i][Ny-1][k]*(yc[Ny+1]-yc[Ny])) / (yc[Ny]-yc[Ny-1]);
                u3_star[i][Ny+1][k] = (u3_star[i][Ny][k]*(yc[Ny+1]-yc[Ny-1])-u3_star[i][Ny-1][k]*(yc[Ny+1]-yc[Ny])) / (yc[Ny]-yc[Ny-1]);
            }
        }
        break;
    default:;
    }

    /* Down. */
    switch (solver->bc_type[4]) {
    case BC_PRESSURE_OUTLET:
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                u1_star[i][j][0] = (u1_star[i][j][1]*(zc[2]-zc[0])-u1_star[i][j][2]*(zc[1]-zc[0])) / (zc[2]-zc[1]);
                u2_star[i][j][0] = (u2_star[i][j][1]*(zc[2]-zc[0])-u2_star[i][j][2]*(zc[1]-zc[0])) / (zc[2]-zc[1]);
                u3_star[i][j][0] = (u3_star[i][j][1]*(zc[2]-zc[0])-u3_star[i][j][2]*(zc[1]-zc[0])) / (zc[2]-zc[1]);
            }
        }
        break;
    default:;
    }

    /* Up. */
    switch (solver->bc_type[5]) {
    case BC_PRESSURE_OUTLET:
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                u1_star[i][j][Nz+1] = (u1_star[i][j][Nz]*(zc[Nz+1]-zc[Nz-1])-u1_star[i][j][Nz-1]*(zc[Nz+1]-zc[Nz])) / (zc[Nz]-zc[Nz-1]);
                u2_star[i][j][Nz+1] = (u2_star[i][j][Nz]*(zc[Nz+1]-zc[Nz-1])-u2_star[i][j][Nz-1]*(zc[Nz+1]-zc[Nz])) / (zc[Nz]-zc[Nz-1]);
                u3_star[i][j][Nz+1] = (u3_star[i][j][Nz]*(zc[Nz+1]-zc[Nz-1])-u3_star[i][j][Nz-1]*(zc[Nz+1]-zc[Nz])) / (zc[Nz]-zc[Nz-1]);
            }
        }
        break;
    default:;
    }
}

static inline void calc_u_tilde(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;
    const int Nx_global = solver->Nx_global;
    const double dt = solver->dt;

    const double *const xc = solver->xc;
    const double *const yc = solver->yc;
    const double *const zc = solver->zc;

    const int (*const flag)[Ny+2][Nz+2] = solver->flag;

    const double (*const u1_star)[Ny+2][Nz+2] = solver->u1_star;
    const double (*const u2_star)[Ny+2][Nz+2] = solver->u2_star;
    const double (*const u3_star)[Ny+2][Nz+2] = solver->u3_star;

    double (*const u1_tilde)[Ny+2][Nz+2] = solver->u1_tilde;
    double (*const u2_tilde)[Ny+2][Nz+2] = solver->u2_tilde;
    double (*const u3_tilde)[Ny+2][Nz+2] = solver->u3_tilde;

    const double (*const p)[Ny+2][Nz+2] = solver->p;

    FOR_ALL_CELL (i, j, k) {
        if (flag[i][j][k] == FLAG_FLUID) {
            u1_tilde[i][j][k] = u1_star[i][j][k] + dt * (p[i+1][j][k] - p[i-1][j][k]) / (xc[i+1] - xc[i-1]);
            u2_tilde[i][j][k] = u2_star[i][j][k] + dt * (p[i][j+1][k] - p[i][j-1][k]) / (yc[j+1] - yc[j-1]);
            u3_tilde[i][j][k] = u3_star[i][j][k] + dt * (p[i][j][k+1] - p[i][j][k-1]) / (zc[k+1] - zc[k-1]);
        }
    }

    /* West. */
    if (solver->ilower == 1) {
        switch (solver->bc_type[3]) {
        case BC_PRESSURE_OUTLET:
            for (int j = 1; j <= Ny; j++) {
                for (int k = 1; k <= Nz; k++) {
                    u1_tilde[0][j][k] = u1_star[0][j][k] + dt * (p[1][j][k] - p[0][j][k]) / (xc[1] - xc[0]);
                    u2_tilde[0][j][k] = u2_star[0][j][k] + dt * (p[0][j+1][k] - p[0][j-1][k]) / (yc[j+1] - yc[j-1]);
                    u3_tilde[0][j][k] = u3_star[0][j][k] + dt * (p[0][j][k+1] - p[0][j][k-1]) / (zc[k+1] - zc[k-1]);
                }
            }
            break;
        case BC_ALL_PERIODIC:
        case BC_VELOCITY_PERIODIC:
            if (solver->num_process == 1) {
                for (int j = 1; j <= Ny; j++) {
                    for (int k = 1; k <= Nz; k++) {
                        u1_tilde[0][j][k] = u1_tilde[Nx][j][k];
                        u2_tilde[0][j][k] = u2_tilde[Nx][j][k];
                        u3_tilde[0][j][k] = u3_tilde[Nx][j][k];
                    }
                }
            }
            else {
                MPI_Send(u1_tilde[1], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-1, 0, MPI_COMM_WORLD);
                MPI_Send(u2_tilde[1], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-1, 1, MPI_COMM_WORLD);
                MPI_Send(u3_tilde[1], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-1, 2, MPI_COMM_WORLD);

                MPI_Recv(u1_tilde[0], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(u2_tilde[0], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(u3_tilde[0], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            break;
        default:;
        }
    }

    /* East. */
    if (solver->iupper == Nx_global) {
        switch (solver->bc_type[1]) {
        case BC_PRESSURE_OUTLET:
            for (int j = 1; j <= Ny; j++) {
                for (int k = 1; k <= Nz; k++) {
                    u1_tilde[Nx+1][j][k] = u1_star[Nx+1][j][k] + dt * (p[Nx+1][j][k] - p[Nx][j][k]) / (xc[Nx+1] - xc[Nx]);
                    u2_tilde[Nx+1][j][k] = u2_star[Nx+1][j][k] + dt * (p[Nx+1][j+1][k] - p[Nx+1][j-1][k]) / (yc[j+1] - yc[j-1]);
                    u3_tilde[Nx+1][j][k] = u3_star[Nx+1][j][k] + dt * (p[Nx+1][j][k+1] - p[Nx+1][j][k-1]) / (zc[k+1] - zc[k-1]);
                }
            }
            break;
        case BC_ALL_PERIODIC:
        case BC_VELOCITY_PERIODIC:
            if (solver->num_process == 1) {
                for (int j = 1; j <= Ny; j++) {
                    for (int k = 1; k <= Nz; k++) {
                        u1_tilde[Nx+1][j][k] = u1_tilde[1][j][k];
                        u2_tilde[Nx+1][j][k] = u2_tilde[1][j][k];
                        u3_tilde[Nx+1][j][k] = u3_tilde[1][j][k];
                    }
                }
            }
            else {
                MPI_Recv(u1_tilde[Nx+1], (Ny+2)*(Nz+2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(u2_tilde[Nx+1], (Ny+2)*(Nz+2), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(u3_tilde[Nx+1], (Ny+2)*(Nz+2), MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                MPI_Send(u1_tilde[Nx], (Ny+2)*(Nz+2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                MPI_Send(u2_tilde[Nx], (Ny+2)*(Nz+2), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
                MPI_Send(u3_tilde[Nx], (Ny+2)*(Nz+2), MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
            }
            break;
        default:;
        }
    }

    /* South. */
    switch (solver->bc_type[2]) {
    case BC_PRESSURE_OUTLET:
        for (int i = 0; i <= Nx+1; i++) {
            for (int k = 1; k <= Nz; k++) {
                u1_tilde[i][0][k] = u1_star[i][0][k] + dt * (p[i+1][0][k] - p[i-1][0][k]) / (xc[i+1] - xc[i-1]);
                u2_tilde[i][0][k] = u2_star[i][0][k] + dt * (p[i][1][k] - p[i][0][k]) / (yc[1] - yc[0]);
                u3_tilde[i][0][k] = u3_star[i][0][k] + dt * (p[i][0][k+1] - p[i][0][k-1]) / (zc[k+1] - zc[k-1]);
            }
        }
        break;
    case BC_ALL_PERIODIC:
    case BC_VELOCITY_PERIODIC:
        for (int i = 0; i <= Nx+1; i++) {
            for (int k = 1; k <= Nz; k++) {
                u1_tilde[i][0][k] = u1_tilde[i][Ny][k];
                u2_tilde[i][0][k] = u2_tilde[i][Ny][k];
                u3_tilde[i][0][k] = u3_tilde[i][Ny][k];
            }
        }
        break;
    default:;
    }

    /* North. */
    switch (solver->bc_type[0]) {
    case BC_PRESSURE_OUTLET:
        for (int i = 0; i <= Nx+1; i++) {
            for (int k = 1; k <= Nz; k++) {
                u1_tilde[i][Ny+1][k] = u1_star[i][Ny+1][k] + dt * (p[i+1][Ny+1][k] - p[i-1][Ny+1][k]) / (xc[i+1] - xc[i-1]);
                u2_tilde[i][Ny+1][k] = u2_star[i][Ny+1][k] + dt * (p[i][Ny+1][k] - p[i][Ny][k]) / (yc[Ny+1] - yc[Ny]);
                u3_tilde[i][Ny+1][k] = u3_star[i][Ny+1][k] + dt * (p[i][Ny+1][k+1] - p[i][Ny+1][k-1]) / (zc[k+1] - zc[k-1]);
            }
        }
        break;
    case BC_ALL_PERIODIC:
    case BC_VELOCITY_PERIODIC:
        for (int i = 0; i <= Nx+1; i++) {
            for (int k = 1; k <= Nz; k++) {
                u1_tilde[i][Ny+1][k] = u1_tilde[i][1][k];
                u2_tilde[i][Ny+1][k] = u2_tilde[i][1][k];
                u3_tilde[i][Ny+1][k] = u3_tilde[i][1][k];
            }
        }
        break;
    default:;
    }

    /* Down. */
    switch (solver->bc_type[4]) {
    case BC_PRESSURE_OUTLET:
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                u1_tilde[i][j][0] = u1_star[i][j][0] + dt * (p[i+1][j][0] - p[i-1][j][0]) / (xc[i+1] - xc[i-1]);
                u2_tilde[i][j][0] = u2_star[i][j][0] + dt * (p[i][j+1][0] - p[i][j-1][0]) / (yc[j+1] - yc[j-1]);
                u3_tilde[i][j][0] = u3_star[i][j][0] + dt * (p[i][j][1] - p[i][j][0]) / (zc[1] - zc[0]);
            }
        }
        break;
    case BC_ALL_PERIODIC:
    case BC_VELOCITY_PERIODIC:
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                u1_tilde[i][j][0] = u1_tilde[i][j][Nz];
                u2_tilde[i][j][0] = u2_tilde[i][j][Nz];
                u3_tilde[i][j][0] = u3_tilde[i][j][Nz];
            }
        }
        break;
    default:;
    }

    /* Up. */
    switch (solver->bc_type[5]) {
    case BC_PRESSURE_OUTLET:
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                u1_tilde[i][j][Nz+1] = u1_star[i][j][Nz+1] + dt * (p[i+1][j][Nz+1] - p[i-1][j][Nz+1]) / (xc[i+1] - xc[i-1]);
                u2_tilde[i][j][Nz+1] = u2_star[i][j][Nz+1] + dt * (p[i][j+1][Nz+1] - p[i][j-1][Nz+1]) / (yc[j+1] - yc[j-1]);
                u3_tilde[i][j][Nz+1] = u3_star[i][j][Nz+1] + dt * (p[i][j][Nz+1] - p[i][j][Nz]) / (zc[Nz+1] - zc[Nz]);
            }
        }
        break;
    case BC_ALL_PERIODIC:
    case BC_VELOCITY_PERIODIC:
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                u1_tilde[i][j][Nz+1] = u1_tilde[i][j][1];
                u2_tilde[i][j][Nz+1] = u2_tilde[i][j][1];
                u3_tilde[i][j][Nz+1] = u3_tilde[i][j][1];
            }
        }
        break;
    default:;
    }

    /* Exchange u1_tilde between the adjacent processes. */
    if (solver->rank != solver->num_process-1) {
        /* Send to next process. */
        MPI_Send(u1_tilde[Nx], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->rank+1, 0, MPI_COMM_WORLD);
    }
    if (solver->rank != 0) {
        /* Receive from previous process. */
        MPI_Recv(u1_tilde[0], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        /* Send to previous process. */
        MPI_Send(u1_tilde[1], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->rank-1, 0, MPI_COMM_WORLD);
    }
    if (solver->rank != solver->num_process-1) {
        /* Receive from next process. */
        MPI_Recv(u1_tilde[Nx+1], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

static inline void calc_U_star(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;
    const double dt = solver->dt;

    const double *const dx = solver->dx;
    const double *const dy = solver->dy;
    const double *const dz = solver->dz;
    const double *const xc = solver->xc;
    const double *const yc = solver->yc;
    const double *const zc = solver->zc;

    const int (*const flag)[Ny+2][Nz+2] = solver->flag;

    const double (*const u1_star)[Ny+2][Nz+2] = solver->u1_star;
    const double (*const u2_star)[Ny+2][Nz+2] = solver->u2_star;
    const double (*const u3_star)[Ny+2][Nz+2] = solver->u3_star;

    const double (*const u1_tilde)[Ny+2][Nz+2] = solver->u1_tilde;
    const double (*const u2_tilde)[Ny+2][Nz+2] = solver->u2_tilde;
    const double (*const u3_tilde)[Ny+2][Nz+2] = solver->u3_tilde;

    double (*const U1_star)[Ny+2][Nz+2] = solver->U1_star;
    double (*const U2_star)[Ny+1][Nz+2] = solver->U2_star;
    double (*const U3_star)[Ny+2][Nz+1] = solver->U3_star;

    const double (*const p)[Ny+2][Nz+2] = solver->p;

    FOR_ALL_XSTAG (i, j, k) {
        U1_star[i][j][k] = (u1_tilde[i][j][k]*dx[i+1] + u1_tilde[i+1][j][k]*dx[i]) / (dx[i] + dx[i+1])
            - dt * (p[i+1][j][k] - p[i][j][k]) / (xc[i+1] - xc[i]);
    }
    FOR_ALL_YSTAG (i, j, k) {
        U2_star[i][j][k] = (u2_tilde[i][j][k]*dy[j+1] + u2_tilde[i][j+1][k]*dy[j]) / (dy[j] + dy[j+1])
            - dt * (p[i][j+1][k] - p[i][j][k]) / (yc[j+1] - yc[j]);
    }
    FOR_ALL_ZSTAG (i, j, k) {
        U3_star[i][j][k] = (u3_tilde[i][j][k]*dz[k+1] + u3_tilde[i][j][k+1]*dz[k]) / (dz[k] + dz[k+1])
            - dt * (p[i][j][k+1] - p[i][j][k]) / (zc[k+1] - zc[k]);
    }

    /* Boundary conditions. */

    /* West. */
    if (solver->ilower == 1) {
        switch (solver->bc_type[3]) {
        case BC_VELOCITY_INLET:
            for (int j = 1; j <= Ny; j++) {
                for (int k = 1; k <= Nz; k++) {
                    U1_star[0][j][k] = solver->bc_val[3];
                }
            }
            break;
        case BC_STATIONARY_WALL:
        case BC_FREE_SLIP_WALL:
            for (int j = 1; j <= Ny; j++) {
                for (int k = 1; k <= Nz; k++) {
                    U1_star[0][j][k] = 0;
                }
            }
            break;
        default:;
        }
    }

    /* East. */
    if (solver->iupper == solver->Nx_global) {
        switch (solver->bc_type[1]) {
        case BC_VELOCITY_INLET:
            for (int j = 1; j <= Ny; j++) {
                for (int k = 1; k <= Nz; k++) {
                    U1_star[Nx][j][k] = solver->bc_val[1];
                }
            }
            break;
        case BC_STATIONARY_WALL:
        case BC_FREE_SLIP_WALL:
            for (int j = 1; j <= Ny; j++) {
                for (int k = 1; k <= Nz; k++) {
                    U1_star[Nx][j][k] = 0;
                }
            }
            break;
        default:;
        }
    }

    /* South. */
    switch (solver->bc_type[2]) {
    case BC_VELOCITY_INLET:
        for (int i = 1; i <= Nx; i++) {
            for (int k = 1; k <= Nz; k++) {
                U2_star[i][0][k] = solver->bc_val[2];
            }
        }
        break;
    case BC_STATIONARY_WALL:
    case BC_FREE_SLIP_WALL:
        for (int i = 1; i <= Nx; i++) {
            for (int k = 1; k <= Nz; k++) {
                U2_star[i][0][k] = 0;
            }
        }
        break;
    default:;
    }

    /* North. */
    switch (solver->bc_type[0]) {
    case BC_VELOCITY_INLET:
        for (int i = 1; i <= Nx; i++) {
            for (int k = 1; k <= Nz; k++) {
                U2_star[i][Ny][k] = solver->bc_val[0];
            }
        }
        break;
    case BC_STATIONARY_WALL:
    case BC_FREE_SLIP_WALL:
        for (int i = 1; i <= Nx; i++) {
            for (int k = 1; k <= Nz; k++) {
                U2_star[i][Ny][k] = 0;
            }
        }
        break;
    default:;
    }

    /* Down. */
    switch (solver->bc_type[4]) {
    case BC_VELOCITY_INLET:
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                U3_star[i][j][0] = solver->bc_val[4];
            }
        }
        break;
    case BC_STATIONARY_WALL:
    case BC_FREE_SLIP_WALL:
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                U3_star[i][j][0] = 0;
            }
        }
        break;
    default:;
    }

    /* Up. */
    switch (solver->bc_type[5]) {
    case BC_VELOCITY_INLET:
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                U3_star[i][j][Nz] = solver->bc_val[5];
            }
        }
        break;
    case BC_STATIONARY_WALL:
    case BC_FREE_SLIP_WALL:
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                U3_star[i][j][Nz] = 0;
            }
        }
        break;
    default:;
    }

    FOR_ALL_XSTAG (i, j, k) {
        if ((flag[i][j][k] == FLAG_FLUID && flag[i+1][j][k] == FLAG_GHOST) || (flag[i][j][k] == FLAG_GHOST && flag[i+1][j][k] == FLAG_FLUID)) {
            U1_star[i][j][k] = (u1_star[i][j][k] * dx[i+1] + u1_star[i+1][j][k] * dx[i]) / (dx[i] + dx[i+1]);
        }
    }
    FOR_ALL_YSTAG (i, j, k) {
        if ((flag[i][j][k] == FLAG_FLUID && flag[i][j+1][k] == FLAG_GHOST) || (flag[i][j][k] == FLAG_GHOST && flag[i][j+1][k] == FLAG_FLUID)) {
            U2_star[i][j][k] = (u2_star[i][j][k] * dy[j+1] + u2_star[i][j+1][k] * dy[j]) / (dy[j] + dy[j+1]);
        }
    }
    FOR_ALL_ZSTAG (i, j, k) {
        if ((flag[i][j][k] == FLAG_FLUID && flag[i][j][k+1] == FLAG_GHOST) || (flag[i][j][k] == FLAG_GHOST && flag[i][j][k+1] == FLAG_FLUID)) {
            U3_star[i][j][k] = (u3_star[i][j][k] * dz[k+1] + u3_star[i][j][k+1] * dz[k]) / (dz[k] + dz[k+1]);
        }
    }
}

static inline void calc_p_prime(IBMSolver *solver, double *final_norm_p) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;
    const int Nx_global = solver->Nx_global;
    const double Re = solver->Re;

    const double *const dx = solver->dx;
    const double *const dy = solver->dy;
    const double *const dz = solver->dz;

    const int (*const flag)[Ny+2][Nz+2] = solver->flag;

    const double (*const U1_star)[Ny+2][Nz+2] = solver->U1_star;
    const double (*const U2_star)[Ny+1][Nz+2] = solver->U2_star;
    const double (*const U3_star)[Ny+2][Nz+1] = solver->U3_star;

    double (*const p_prime)[Ny+2][Nz+2] = solver->p_prime;
    double (*const p_coeffsum)[Ny+2][Nz+2] = solver->p_coeffsum;

    int hypre_ierr = 0;

    FOR_ALL_CELL (i, j, k) {
        double *const rhs = &solver->vector_values[LOCL_CELL_IDX(i, j, k)-1];
        if (flag[i][j][k] == 1) {
            *rhs = -1/(2*Re) * (
                    (U1_star[i][j][k] - U1_star[i-1][j][k]) / dx[i]
                    + (U2_star[i][j][k] - U2_star[i][j-1][k]) / dy[j]
                    + (U3_star[i][j][k] - U3_star[i][j][k-1]) / dz[k]
                );
            *rhs /= p_coeffsum[i][j][k];
        }
        else {
            *rhs = 0;
        }
    }

    HYPRE_IJVectorSetValues(solver->b, Nx*Ny*Nz, solver->vector_rows, solver->vector_values);
    HYPRE_IJVectorSetValues(solver->x, Nx*Ny*Nz, solver->vector_rows, solver->vector_zeros);

    HYPRE_IJVectorAssemble(solver->b);
    HYPRE_IJVectorAssemble(solver->x);

    HYPRE_IJVectorGetObject(solver->b, (void **)&solver->par_b);
    HYPRE_IJVectorGetObject(solver->x, (void **)&solver->par_x);

    switch (solver->linear_solver_type) {
    case SOLVER_AMG:
        hypre_ierr = HYPRE_BoomerAMGSolve(solver->linear_solver_p, solver->parcsr_A_p, solver->par_b, solver->par_x);
        break;
    case SOLVER_PCG:
        hypre_ierr = HYPRE_ParCSRPCGSolve(solver->linear_solver_p, solver->parcsr_A_p, solver->par_b, solver->par_x);
        break;
    case SOLVER_BiCGSTAB:
        hypre_ierr = HYPRE_ParCSRBiCGSTABSolve(solver->linear_solver_p, solver->parcsr_A_p, solver->par_b, solver->par_x);
        break;
    case SOLVER_GMRES:
        hypre_ierr = HYPRE_ParCSRGMRESSolve(solver->linear_solver_p, solver->parcsr_A_p, solver->par_b, solver->par_x);
        break;
    default:;
    }
    if (hypre_ierr) {
        printf("error: floating pointer error raised in p_prime\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    HYPRE_IJVectorGetValues(solver->x, Nx*Ny*Nz, solver->vector_rows, solver->vector_res);
    FOR_ALL_CELL (i, j, k) {
        p_prime[i][j][k] = solver->vector_res[LOCL_CELL_IDX(i, j, k)-1];
    }

    switch (solver->linear_solver_type) {
    case SOLVER_AMG:
        HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver->linear_solver_p, final_norm_p);
        break;
    case SOLVER_PCG:
        HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(solver->linear_solver_p, final_norm_p);
        break;
    case SOLVER_BiCGSTAB:
        HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm(solver->linear_solver_p, final_norm_p);
        break;
    case SOLVER_GMRES:
        HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(solver->linear_solver_p, final_norm_p);
        break;
    default:;
    }

    /* West. */
    if (solver->ilower == 1) {
        switch (solver->bc_type[3]) {
        case BC_VELOCITY_INLET:
        case BC_STATIONARY_WALL:
        case BC_FREE_SLIP_WALL:
            for (int j = 1; j <= Ny; j++) {
                for (int k = 1; k <= Nz; k++) {
                    p_prime[0][j][k] = p_prime[1][j][k];
                }
            }
            break;
        case BC_PRESSURE_OUTLET:
        case BC_VELOCITY_PERIODIC:
            for (int j = 1; j <= Ny; j++) {
                for (int k = 1; k <= Nz; k++) {
                    p_prime[0][j][k] = -p_prime[1][j][k];
                }
            }
            break;
        case BC_ALL_PERIODIC:
            if (solver->num_process == 1) {
                for (int j = 1; j <= Ny; j++) {
                    for (int k = 1; k <= Nz; k++) {
                        p_prime[0][j][k] = p_prime[Nx][j][k];
                    }
                }
            }
            else {
                MPI_Send(p_prime[1], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-1, 0, MPI_COMM_WORLD);
                MPI_Recv(p_prime[0], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            break;
        default:;
        }
    }

    /* East. */
    if (solver->iupper == Nx_global) {
        switch (solver->bc_type[1]) {
        case BC_VELOCITY_INLET:
        case BC_STATIONARY_WALL:
        case BC_FREE_SLIP_WALL:
            for (int j = 1; j <= Ny; j++) {
                for (int k = 1; k <= Nz; k++) {
                    p_prime[Nx+1][j][k] = p_prime[Nx][j][k];
                }
            }
            break;
        case BC_PRESSURE_OUTLET:
        case BC_VELOCITY_PERIODIC:
            for (int j = 1; j <= Ny; j++) {
                for (int k = 1; k <= Nz; k++) {
                    p_prime[Nx+1][j][k] = -p_prime[Nx][j][k];
                }
            }
            break;
        case BC_ALL_PERIODIC:
            if (solver->num_process == 1) {
                for (int j = 1; j <= Ny; j++) {
                    for (int k = 1; k <= Nz; k++) {
                        p_prime[Nx+1][j][k] = p_prime[1][j][k];
                    }
                }
            }
            else {
                MPI_Recv(p_prime[Nx+1], (Ny+2)*(Nz+2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(p_prime[Nx], (Ny+2)*(Nz+2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
            break;
        default:;
        }
    }

    /* South. */
    switch (solver->bc_type[2]) {
    case BC_VELOCITY_INLET:
    case BC_STATIONARY_WALL:
    case BC_FREE_SLIP_WALL:
        for (int i = 0; i <= Nx+1; i++) {
            for (int k = 1; k <= Nz; k++) {
                p_prime[i][0][k] = p_prime[i][1][k];
            }
        }
        break;
    case BC_PRESSURE_OUTLET:
    case BC_VELOCITY_PERIODIC:
        for (int i = 0; i <= Nx+1; i++) {
            for (int k = 1; k <= Nz; k++) {
                p_prime[i][0][k] = -p_prime[i][1][k];
            }
        }
        break;
    case BC_ALL_PERIODIC:
        for (int i = 0; i <= Nx+1; i++) {
            for (int k = 1; k <= Nz; k++) {
                p_prime[i][0][k] = p_prime[i][Ny][k];
            }
        }
        break;
    default:;
    }

    /* North. */
    switch (solver->bc_type[0]) {
    case BC_VELOCITY_INLET:
    case BC_STATIONARY_WALL:
    case BC_FREE_SLIP_WALL:
        for (int i = 0; i <= Nx+1; i++) {
            for (int k = 1; k <= Nz; k++) {
                p_prime[i][Ny+1][k] = p_prime[i][Ny][k];
            }
        }
        break;
    case BC_PRESSURE_OUTLET:
    case BC_VELOCITY_PERIODIC:
        for (int i = 0; i <= Nx+1; i++) {
            for (int k = 1; k <= Nz; k++) {
                p_prime[i][Ny+1][k] = -p_prime[i][Ny][k];
            }
        }
        break;
    case BC_ALL_PERIODIC:
        for (int i = 0; i <= Nx+1; i++) {
            for (int k = 1; k <= Nz; k++) {
                p_prime[i][Ny+1][k] = p_prime[i][1][k];
            }
        }
        break;
    default:;
    }

    /* Down. */
    switch (solver->bc_type[4]) {
    case BC_VELOCITY_INLET:
    case BC_STATIONARY_WALL:
    case BC_FREE_SLIP_WALL:
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                p_prime[i][j][0] = p_prime[i][j][1];
            }
        }
        break;
    case BC_PRESSURE_OUTLET:
    case BC_VELOCITY_PERIODIC:
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                p_prime[i][j][0] = -p_prime[i][j][1];
            }
        }
        break;
    case BC_ALL_PERIODIC:
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                p_prime[i][j][0] = p_prime[i][j][Nz];
            }
        }
        break;
    default:;
    }

    /* Up. */
    switch (solver->bc_type[5]) {
    case BC_VELOCITY_INLET:
    case BC_STATIONARY_WALL:
    case BC_FREE_SLIP_WALL:
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                p_prime[i][j][Nz+1] = p_prime[i][j][Nz];
            }
        }
        break;
    case BC_PRESSURE_OUTLET:
    case BC_VELOCITY_PERIODIC:
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                p_prime[i][j][Nz+1] = -p_prime[i][j][Nz];
            }
        }
        break;
    case BC_ALL_PERIODIC:
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                p_prime[i][j][Nz+1] = p_prime[i][j][1];
            }
        }
        break;
    default:;
    }

    /* Exchange p_prime between the adjacent processes. */
    if (solver->rank != solver->num_process-1) {
        /* Send to next process. */
        MPI_Send(p_prime[Nx], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->rank+1, 0, MPI_COMM_WORLD);
    }
    if (solver->rank != 0) {
        /* Receive from previous process. */
        MPI_Recv(p_prime[0], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        /* Send to previous process. */
        MPI_Send(p_prime[1], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->rank-1, 0, MPI_COMM_WORLD);
    }
    if (solver->rank != solver->num_process-1) {
        /* Receive from next process. */
        MPI_Recv(p_prime[Nx+1], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

static inline void update_next(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;
    const double dt = solver->dt;

    const double *const xc = solver->xc;
    const double *const yc = solver->yc;
    const double *const zc = solver->zc;

    const int (*const flag)[Ny+2][Nz+2] = solver->flag;

    const double (*const p)[Ny+2][Nz+2] = solver->p;

    const double (*const u1_star)[Ny+2][Nz+2] = solver->u1_star;
    const double (*const u2_star)[Ny+2][Nz+2] = solver->u2_star;
    const double (*const u3_star)[Ny+2][Nz+2] = solver->u3_star;

    const double (*const U1_star)[Ny+2][Nz+2] = solver->U1_star;
    const double (*const U2_star)[Ny+1][Nz+2] = solver->U2_star;
    const double (*const U3_star)[Ny+2][Nz+1] = solver->U3_star;

    const double (*const p_prime)[Ny+2][Nz+2] = solver->p_prime;

    double (*const u1_next)[Ny+2][Nz+2] = solver->u1_next;
    double (*const u2_next)[Ny+2][Nz+2] = solver->u2_next;
    double (*const u3_next)[Ny+2][Nz+2] = solver->u3_next;
    double (*const U1_next)[Ny+2][Nz+2] = solver->U1_next;
    double (*const U2_next)[Ny+1][Nz+2] = solver->U2_next;
    double (*const U3_next)[Ny+2][Nz+1] = solver->U3_next;
    double (*const p_next)[Ny+2][Nz+2] = solver->p_next;

    /* Calculate p_next. */
    FOR_ALL_CELL (i, j, k) {
        if (flag[i][j][k] == FLAG_FLUID || flag[i][j][k] == FLAG_GHOST) {
            p_next[i][j][k] = p[i][j][k] + p_prime[i][j][k];
        }
        else if (flag[i][j][k] == FLAG_SOLID) {
            p_next[i][j][k] = NAN;
        }
    }

    /* Calculate u_next. */
    FOR_ALL_CELL (i, j, k) {
        u1_next[i][j][k] = u1_star[i][j][k] - dt * (p_prime[i+1][j][k] - p_prime[i-1][j][k]) / (xc[i+1] - xc[i-1]);
        u2_next[i][j][k] = u2_star[i][j][k] - dt * (p_prime[i][j+1][k] - p_prime[i][j-1][k]) / (yc[j+1] - yc[j-1]);
        u3_next[i][j][k] = u3_star[i][j][k] - dt * (p_prime[i][j][k+1] - p_prime[i][j][k-1]) / (zc[k+1] - zc[k-1]);
    }

    /* Calculate U_next. */
    FOR_ALL_XSTAG (i, j, k) {
        U1_next[i][j][k] = U1_star[i][j][k] - dt * (p_prime[i+1][j][k] - p_prime[i][j][k]) / (xc[i+1] - xc[i]);
    }
    FOR_ALL_YSTAG (i, j, k) {
        U2_next[i][j][k] = U2_star[i][j][k] - dt * (p_prime[i][j+1][k] - p_prime[i][j][k]) / (yc[j+1] - yc[j]);
    }
    FOR_ALL_ZSTAG (i, j, k) {
        U3_next[i][j][k] = U3_star[i][j][k] - dt * (p_prime[i][j][k+1] - p_prime[i][j][k]) / (zc[k+1] - zc[k]);
    }

    /* Update for next time step. */
    SWAP(solver->u1, solver->u1_next);
    SWAP(solver->u2, solver->u2_next);
    SWAP(solver->u3, solver->u3_next);
    SWAP(solver->U1, solver->U1_next);
    SWAP(solver->U2, solver->U2_next);
    SWAP(solver->U3, solver->U3_next);
    SWAP(solver->p, solver->p_next);
    SWAP(solver->N1_prev, solver->N1);
    SWAP(solver->N2_prev, solver->N2);
    SWAP(solver->N3_prev, solver->N3);

    /* Set boundary conditions. */
    update_bc(solver);

    /* Exchange u1, u2, u3, and p between the adjacent processes. */
    adj_exchange(solver);

    /* Set boundary condition on obstacle surface. */
    update_ghost_and_solid(solver);
}

static void interp_stag_vel(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    const double *const dx = solver->dx;
    const double *const dy = solver->dy;
    const double *const dz = solver->dz;

    double (*const u1)[Ny+2][Nz+2] = solver->u1;
    double (*const u2)[Ny+2][Nz+2] = solver->u2;
    double (*const u3)[Ny+2][Nz+2] = solver->u3;

    double (*const U1)[Ny+2][Nz+2] = solver->U1;
    double (*const U2)[Ny+1][Nz+2] = solver->U2;
    double (*const U3)[Ny+2][Nz+1] = solver->U3;

    for (int i = 0; i <= Nx; i++) {
        for (int j = 0; j <= Ny+1; j++) {
            for (int k = 0; k <= Nz+1; k++) {
                U1[i][j][k] = (u1[i][j][k]*dx[i+1] + u1[i+1][j][k]*dx[i]) / (dx[i]+dx[i+1]);
            }
        }
    }
    for (int i = 0; i <= Nx+1; i++) {
        for (int j = 0; j <= Ny; j++) {
            for (int k = 0; k <= Nz+1; k++) {
                U2[i][j][k] = (u2[i][j][k]*dy[j+1] + u2[i][j+1][k]*dy[j]) / (dy[j]+dy[j+1]);
            }
        }
    }
    for (int i = 0; i <= Nx+1; i++) {
        for (int j = 0; j <= Ny+1; j++) {
            for (int k = 0; k <= Nz; k++) {
                U3[i][j][k] = (u3[i][j][k]*dz[k+1] + u3[i][j][k+1]*dz[k]) / (dz[k]+dz[k+1]);
            }
        }
    }
}

static void autosave(IBMSolver *solver) {
    char filename_u1[100], filename_u2[100], filename_u3[100], filename_p[100];

    snprintf(filename_u1, 100, "%s-%05d", solver->autosave_u1, solver->iter);
    snprintf(filename_u2, 100, "%s-%05d", solver->autosave_u2, solver->iter);
    snprintf(filename_u3, 100, "%s-%05d", solver->autosave_u3, solver->iter);
    snprintf(filename_p, 100, "%s-%05d", solver->autosave_p, solver->iter);

    IBMSolver_export_results(
        solver, filename_u1, filename_u2, filename_u3, filename_p
    );
}

static void update_bc(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;
    const int Nx_global = solver->Nx_global;

    const double *const xc = solver->xc;
    const double *const yc = solver->yc;
    const double *const zc = solver->zc;

    double (*const u1)[Ny+2][Nz+2] = solver->u1;
    double (*const u2)[Ny+2][Nz+2] = solver->u2;
    double (*const u3)[Ny+2][Nz+2] = solver->u3;
    double (*const p)[Ny+2][Nz+2] = solver->p;

    /* Exchange u_next for boundary processes. */
    if (solver->num_process > 1) {
        if (solver->rank == 0) {
            /* Receive from next process. */
            MPI_Recv(u1[Nx+1], (Ny+2)*(Nz+2), MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(u2[Nx+1], (Ny+2)*(Nz+2), MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(u3[Nx+1], (Ny+2)*(Nz+2), MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else if (solver->rank == 1) {
            /* Send to previous process. */
            MPI_Send(u1[1], (Ny+2)*(Nz+2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            MPI_Send(u2[1], (Ny+2)*(Nz+2), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            MPI_Send(u3[1], (Ny+2)*(Nz+2), MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
        }

        if (solver->rank == solver->num_process-1) {
            /* Receive from previous process. */
            MPI_Recv(u1[0], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(u2[0], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-2, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(u3[0], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-2, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else if (solver->rank == solver->num_process-2) {
            /* Send to next process. */
            MPI_Send(u1[Nx], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-1, 0, MPI_COMM_WORLD);
            MPI_Send(u2[Nx], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-1, 1, MPI_COMM_WORLD);
            MPI_Send(u3[Nx], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-1, 2, MPI_COMM_WORLD);
        }
    }

    /* Set velocity boundary conditions. */

    /* West. */
    if (solver->ilower == 1) {
        switch (solver->bc_type[3]) {
        case BC_VELOCITY_INLET:
            for (int j = 1; j <= Ny; j++) {
                for (int k = 1; k <= Nz; k++) {
                    u1[0][j][k] = 2*solver->bc_val[3] - u1[1][j][k];
                    u2[0][j][k] = -u2[1][j][k];
                    u3[0][j][k] = -u3[1][j][k];
                }
            }
            break;
        case BC_PRESSURE_OUTLET:
            for (int j = 1; j <= Ny; j++) {
                for (int k = 1; k <= Nz; k++) {
                    u1[0][j][k] = (u1[1][j][k]*(xc[2]-xc[0])-u1[2][j][k]*(xc[1]-xc[0])) / (xc[2]-xc[1]);
                    u2[0][j][k] = (u2[1][j][k]*(xc[2]-xc[0])-u2[2][j][k]*(xc[1]-xc[0])) / (xc[2]-xc[1]);
                    u3[0][j][k] = (u3[1][j][k]*(xc[2]-xc[0])-u3[2][j][k]*(xc[1]-xc[0])) / (xc[2]-xc[1]);
                }
            }
            break;
        case BC_STATIONARY_WALL:
            for (int j = 1; j <= Ny; j++) {
                for (int k = 1; k <= Nz; k++) {
                    u1[0][j][k] = -u1[1][j][k];
                    u2[0][j][k] = -u2[1][j][k];
                    u3[0][j][k] = -u3[1][j][k];
                }
            }
            break;
        case BC_FREE_SLIP_WALL:
            for (int j = 1; j <= Ny; j++) {
                for (int k = 1; k <= Nz; k++) {
                    u1[0][j][k] = -u1[1][j][k];
                    u2[0][j][k] = u2[1][j][k];
                    u3[0][j][k] = u3[1][j][k];
                }
            }
            break;
        case BC_ALL_PERIODIC:
        case BC_VELOCITY_PERIODIC:
            if (solver->num_process == 1) {
                for (int j = 1; j <= Ny; j++) {
                    for (int k = 1; k <= Nz; k++) {
                        u1[0][j][k] = u1[Nx][j][k];
                        u2[0][j][k] = u2[Nx][j][k];
                        u3[0][j][k] = u3[Nx][j][k];
                    }
                }
            }
            else {
                MPI_Send(u1[1], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-1, 0, MPI_COMM_WORLD);
                MPI_Send(u2[1], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-1, 1, MPI_COMM_WORLD);
                MPI_Send(u3[1], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-1, 2, MPI_COMM_WORLD);

                MPI_Recv(u1[0], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(u2[0], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(u3[0], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            break;
        default:;
        }
    }

    /* East. */
    if (solver->iupper == Nx_global) {
        switch (solver->bc_type[1]) {
        case BC_VELOCITY_INLET:
            for (int j = 1; j <= Ny; j++) {
                for (int k = 1; k <= Nz; k++) {
                    u1[Nx+1][j][k] = 2*solver->bc_val[1] - u1[Nx][j][k];
                    u2[Nx+1][j][k] = -u2[Nx][j][k];
                    u3[Nx+1][j][k] = -u3[Nx][j][k];
                }
            }
            break;
        case BC_PRESSURE_OUTLET:
            for (int j = 1; j <= Ny; j++) {
                for (int k = 1; k <= Nz; k++) {
                    u1[Nx+1][j][k] = (u1[Nx][j][k]*(xc[Nx+1]-xc[Nx-1])-u1[Nx-1][j][k]*(xc[Nx+1]-xc[Nx])) / (xc[Nx]-xc[Nx-1]);
                    u2[Nx+1][j][k] = (u2[Nx][j][k]*(xc[Nx+1]-xc[Nx-1])-u2[Nx-1][j][k]*(xc[Nx+1]-xc[Nx])) / (xc[Nx]-xc[Nx-1]);
                    u3[Nx+1][j][k] = (u3[Nx][j][k]*(xc[Nx+1]-xc[Nx-1])-u3[Nx-1][j][k]*(xc[Nx+1]-xc[Nx])) / (xc[Nx]-xc[Nx-1]);
                }
            }
            break;
        case BC_STATIONARY_WALL:
            for (int j = 1; j <= Ny; j++) {
                for (int k = 1; k <= Nz; k++) {
                    u1[Nx+1][j][k] = -u1[Nx][j][k];
                    u2[Nx+1][j][k] = -u2[Nx][j][k];
                    u3[Nx+1][j][k] = -u3[Nx][j][k];
                }
            }
            break;
        case BC_FREE_SLIP_WALL:
            for (int j = 1; j <= Ny; j++) {
                for (int k = 1; k <= Nz; k++) {
                    u1[Nx+1][j][k] = -u1[Nx][j][k];
                    u2[Nx+1][j][k] = u2[Nx][j][k];
                    u3[Nx+1][j][k] = u3[Nx][j][k];
                }
            }
            break;
        case BC_ALL_PERIODIC:
        case BC_VELOCITY_PERIODIC:
            if (solver->num_process == 1) {
                for (int j = 1; j <= Ny; j++) {
                    for (int k = 1; k <= Nz; k++) {
                        u1[Nx+1][j][k] = u1[1][j][k];
                        u2[Nx+1][j][k] = u2[1][j][k];
                        u3[Nx+1][j][k] = u3[1][j][k];
                    }
                }
            }
            else {
                MPI_Recv(u1[Nx+1], (Ny+2)*(Nz+2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(u2[Nx+1], (Ny+2)*(Nz+2), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(u3[Nx+1], (Ny+2)*(Nz+2), MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                MPI_Send(u1[Nx], (Ny+2)*(Nz+2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                MPI_Send(u2[Nx], (Ny+2)*(Nz+2), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
                MPI_Send(u3[Nx], (Ny+2)*(Nz+2), MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
            }
            break;
        default:;
        }
    }

    /* South. */
    switch (solver->bc_type[2]) {
    case BC_VELOCITY_INLET:
        for (int i = 0; i <= Nx+1; i++) {
            for (int k = 1; k <= Nz; k++) {
                u1[i][0][k] = -u1[i][1][k];
                u2[i][0][k] = 2*solver->bc_val[2] - u2[i][1][k];
                u3[i][0][k] = -u3[i][1][k];
            }
        }
        break;
    case BC_PRESSURE_OUTLET:
        for (int i = 0; i <= Nx+1; i++) {
            for (int k = 1; k <= Nz; k++) {
                u1[i][0][k] = (u1[i][1][k]*(yc[2]-yc[0])-u1[i][2][k]*(yc[1]-yc[0])) / (yc[2]-yc[1]);
                u2[i][0][k] = (u2[i][1][k]*(yc[2]-yc[0])-u2[i][2][k]*(yc[1]-yc[0])) / (yc[2]-yc[1]);
                u3[i][0][k] = (u3[i][1][k]*(yc[2]-yc[0])-u3[i][2][k]*(yc[1]-yc[0])) / (yc[2]-yc[1]);
            }
        }
        break;
    case BC_STATIONARY_WALL:
        for (int i = 0; i <= Nx+1; i++) {
            for (int k = 1; k <= Nz; k++) {
                u1[i][0][k] = -u1[i][1][k];
                u2[i][0][k] = -u2[i][1][k];
                u3[i][0][k] = -u3[i][1][k];
            }
        }
        break;
    case BC_FREE_SLIP_WALL:
        for (int i = 0; i <= Nx+1; i++) {
            for (int k = 1; k <= Nz; k++) {
                u1[i][0][k] = u1[i][1][k];
                u2[i][0][k] = -u2[i][1][k];
                u3[i][0][k] = u3[i][1][k];
            }
        }
        break;
    case BC_ALL_PERIODIC:
    case BC_VELOCITY_PERIODIC:
        for (int i = 0; i <= Nx+1; i++) {
            for (int k = 1; k <= Nz; k++) {
                u1[i][0][k] = u1[i][Ny][k];
                u2[i][0][k] = u2[i][Ny][k];
                u3[i][0][k] = u3[i][Ny][k];
            }
        }
        break;
    default:;
    }

    /* North. */
    switch (solver->bc_type[0]) {
    case BC_VELOCITY_INLET:
        for (int i = 0; i <= Nx+1; i++) {
            for (int k = 1; k <= Nz; k++) {
                u1[i][Ny+1][k] = -u1[i][Ny][k];
                u2[i][Ny+1][k] = 2*solver->bc_val[0] - u2[i][Ny][k];
                u3[i][Ny+1][k] = -u3[i][Ny][k];
            }
        }
        break;
    case BC_PRESSURE_OUTLET:
        for (int i = 0; i <= Nx+1; i++) {
            for (int k = 1; k <= Nz; k++) {
                u1[i][Ny+1][k] = (u1[i][Ny][k]*(yc[Ny+1]-yc[Ny-1])-u1[i][Ny-1][k]*(yc[Ny+1]-yc[Ny])) / (yc[Ny]-yc[Ny-1]);
                u2[i][Ny+1][k] = (u2[i][Ny][k]*(yc[Ny+1]-yc[Ny-1])-u2[i][Ny-1][k]*(yc[Ny+1]-yc[Ny])) / (yc[Ny]-yc[Ny-1]);
                u3[i][Ny+1][k] = (u3[i][Ny][k]*(yc[Ny+1]-yc[Ny-1])-u3[i][Ny-1][k]*(yc[Ny+1]-yc[Ny])) / (yc[Ny]-yc[Ny-1]);
            }
        }
        break;
    case BC_STATIONARY_WALL:
        for (int i = 0; i <= Nx+1; i++) {
            for (int k = 1; k <= Nz; k++) {
                u1[i][Ny+1][k] = -u1[i][Ny][k];
                u2[i][Ny+1][k] = -u2[i][Ny][k];
                u3[i][Ny+1][k] = -u3[i][Ny][k];
            }
        }
        break;
    case BC_FREE_SLIP_WALL:
        for (int i = 0; i <= Nx+1; i++) {
            for (int k = 1; k <= Nz; k++) {
                u1[i][Ny+1][k] = u1[i][Ny][k];
                u2[i][Ny+1][k] = -u2[i][Ny][k];
                u3[i][Ny+1][k] = u3[i][Ny][k];
            }
        }
        break;
    case BC_ALL_PERIODIC:
    case BC_VELOCITY_PERIODIC:
        for (int i = 0; i <= Nx+1; i++) {
            for (int k = 1; k <= Nz; k++) {
                u1[i][Ny+1][k] = u1[i][1][k];
                u2[i][Ny+1][k] = u2[i][1][k];
                u3[i][Ny+1][k] = u3[i][1][k];
            }
        }
        break;
    default:;
    }

    /* Down. */
    switch (solver->bc_type[4]) {
    case BC_VELOCITY_INLET:
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                u1[i][j][0] = -u1[i][j][1];
                u2[i][j][0] = -u2[i][j][1];
                u3[i][j][0] = 2*solver->bc_val[4] - u3[i][j][1];
            }
        }
        break;
    case BC_PRESSURE_OUTLET:
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                u1[i][j][0] = (u1[i][j][1]*(zc[2]-zc[0])-u1[i][j][2]*(zc[1]-zc[0])) / (zc[2]-zc[1]);
                u2[i][j][0] = (u2[i][j][1]*(zc[2]-zc[0])-u2[i][j][2]*(zc[1]-zc[0])) / (zc[2]-zc[1]);
                u3[i][j][0] = (u3[i][j][1]*(zc[2]-zc[0])-u3[i][j][2]*(zc[1]-zc[0])) / (zc[2]-zc[1]);
            }
        }
        break;
    case BC_STATIONARY_WALL:
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                u1[i][j][0] = -u1[i][j][1];
                u2[i][j][0] = -u2[i][j][1];
                u3[i][j][0] = -u3[i][j][1];
            }
        }
        break;
    case BC_FREE_SLIP_WALL:
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                u1[i][j][0] = u1[i][j][1];
                u2[i][j][0] = u2[i][j][1];
                u3[i][j][0] = -u3[i][j][1];
            }
        }
        break;
    case BC_ALL_PERIODIC:
    case BC_VELOCITY_PERIODIC:
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                u1[i][j][0] = u1[i][j][Nz];
                u2[i][j][0] = u2[i][j][Nz];
                u3[i][j][0] = u3[i][j][Nz];
            }
        }
        break;
    default:;
    }

    /* Up. */
    switch (solver->bc_type[5]) {
    case BC_VELOCITY_INLET:
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                u1[i][j][Nz+1] = -u1[i][j][Nz];
                u2[i][j][Nz+1] = -u2[i][j][Nz];
                u3[i][j][Nz+1] = 2*solver->bc_val[5] - u3[i][j][Nz];
            }
        }
        break;
    case BC_PRESSURE_OUTLET:
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                u1[i][j][Nz+1] = (u1[i][j][Nz]*(zc[Nz+1]-zc[Nz-1])-u1[i][j][Nz-1]*(zc[Nz+1]-zc[Nz])) / (zc[Nz]-zc[Nz-1]);
                u2[i][j][Nz+1] = (u2[i][j][Nz]*(zc[Nz+1]-zc[Nz-1])-u2[i][j][Nz-1]*(zc[Nz+1]-zc[Nz])) / (zc[Nz]-zc[Nz-1]);
                u3[i][j][Nz+1] = (u3[i][j][Nz]*(zc[Nz+1]-zc[Nz-1])-u3[i][j][Nz-1]*(zc[Nz+1]-zc[Nz])) / (zc[Nz]-zc[Nz-1]);
            }
        }
        break;
    case BC_STATIONARY_WALL:
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                u1[i][j][Nz+1] = -u1[i][j][Nz];
                u2[i][j][Nz+1] = -u2[i][j][Nz];
                u3[i][j][Nz+1] = -u3[i][j][Nz];
            }
        }
        break;
    case BC_FREE_SLIP_WALL:
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                u1[i][j][Nz+1] = u1[i][j][Nz];
                u2[i][j][Nz+1] = u2[i][j][Nz];
                u3[i][j][Nz+1] = -u3[i][j][Nz];
            }
        }
        break;
    case BC_ALL_PERIODIC:
    case BC_VELOCITY_PERIODIC:
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                u1[i][j][Nz+1] = u1[i][j][1];
                u2[i][j][Nz+1] = u2[i][j][1];
                u3[i][j][Nz+1] = u3[i][j][1];
            }
        }
        break;
    default:;
    }

    /* Set pressure boundary conditions. */

    /* West. */
    if (solver->ilower == 1) {
        switch (solver->bc_type[3]) {
        case BC_VELOCITY_INLET:
        case BC_STATIONARY_WALL:
        case BC_FREE_SLIP_WALL:
            for (int j = 1; j <= Ny; j++) {
                for (int k = 1; k <= Nz; k++) {
                    p[0][j][k] = p[1][j][k];
                }
            }
            break;
        case BC_PRESSURE_OUTLET:
        case BC_VELOCITY_PERIODIC:
            for (int j = 1; j <= Ny; j++) {
                for (int k = 1; k <= Nz; k++) {
                    p[0][j][k] = 2*solver->bc_val[3] - p[1][j][k];
                }
            }
            break;
        case BC_ALL_PERIODIC:
            if (solver->num_process == 1) {
                for (int j = 1; j <= Ny; j++) {
                    for (int k = 1; k <= Nz; k++) {
                        p[0][j][k] = p[Nx][j][k];
                    }
                }
            }
            else {
                MPI_Send(p[1], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-1, 0, MPI_COMM_WORLD);
                MPI_Recv(p[0], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            break;
        default:;
        }
    }

    /* East. */
    if (solver->iupper == Nx_global) {
        switch (solver->bc_type[1]) {
        case BC_VELOCITY_INLET:
        case BC_STATIONARY_WALL:
        case BC_FREE_SLIP_WALL:
            for (int j = 1; j <= Ny; j++) {
                for (int k = 1; k <= Nz; k++) {
                    p[Nx+1][j][k] = p[Nx][j][k];
                }
            }
            break;
        case BC_PRESSURE_OUTLET:
        case BC_VELOCITY_PERIODIC:
            for (int j = 1; j <= Ny; j++) {
                for (int k = 1; k <= Nz; k++) {
                    p[Nx+1][j][k] = 2*solver->bc_val[1] - p[Nx][j][k];
                }
            }
            break;
        case BC_ALL_PERIODIC:
            if (solver->num_process == 1) {
                for (int j = 1; j <= Ny; j++) {
                    for (int k = 1; k <= Nz; k++) {
                        p[Nx+1][j][k] = p[1][j][k];
                    }
                }
            }
            else {
                MPI_Recv(p[Nx+1], (Ny+2)*(Nz+2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(p[Nx], (Ny+2)*(Nz+2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
            break;
        default:;
        }
    }

    /* South. */
    switch (solver->bc_type[2]) {
    case BC_VELOCITY_INLET:
    case BC_STATIONARY_WALL:
    case BC_FREE_SLIP_WALL:
        for (int i = 0; i <= Nx+1; i++) {
            for (int k = 1; k <= Nz; k++) {
                p[i][0][k] = p[i][1][k];
            }
        }
        break;
    case BC_PRESSURE_OUTLET:
    case BC_VELOCITY_PERIODIC:
        for (int i = 0; i <= Nx+1; i++) {
            for (int k = 1; k <= Nz; k++) {
                p[i][0][k] = 2*solver->bc_val[2] - p[i][1][k];
            }
        }
        break;
    case BC_ALL_PERIODIC:
        for (int i = 0; i <= Nx+1; i++) {
            for (int k = 1; k <= Nz; k++) {
                p[i][0][k] = p[i][Ny][k];
            }
        }
        break;
    default:;
    }

    /* North. */
    switch (solver->bc_type[0]) {
    case BC_VELOCITY_INLET:
    case BC_STATIONARY_WALL:
    case BC_FREE_SLIP_WALL:
        for (int i = 0; i <= Nx+1; i++) {
            for (int k = 1; k <= Nz; k++) {
                p[i][Ny+1][k] = p[i][Ny][k];
            }
        }
        break;
    case BC_PRESSURE_OUTLET:
    case BC_VELOCITY_PERIODIC:
        for (int i = 0; i <= Nx+1; i++) {
            for (int k = 1; k <= Nz; k++) {
                p[i][Ny+1][k] = 2*solver->bc_val[0] - p[i][Ny][k];
            }
        }
        break;
    case BC_ALL_PERIODIC:
        for (int i = 0; i <= Nx+1; i++) {
            for (int k = 1; k <= Nz; k++) {
                p[i][Ny+1][k] = p[i][1][k];
            }
        }
        break;
    default:;
    }

    /* Down. */
    switch (solver->bc_type[4]) {
    case BC_VELOCITY_INLET:
    case BC_STATIONARY_WALL:
    case BC_FREE_SLIP_WALL:
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                p[i][j][0] = p[i][j][1];
            }
        }
        break;
    case BC_PRESSURE_OUTLET:
    case BC_VELOCITY_PERIODIC:
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                p[i][j][0] = 2*solver->bc_val[4] - p[i][j][1];
            }
        }
        break;
    case BC_ALL_PERIODIC:
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                p[i][j][0] = p[i][j][Nz];
            }
        }
        break;
    default:;
    }

    /* Up. */
    switch (solver->bc_type[5]) {
    case BC_VELOCITY_INLET:
    case BC_STATIONARY_WALL:
    case BC_FREE_SLIP_WALL:
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                p[i][j][Nz+1] = p[i][j][Nz];
            }
        }
        break;
    case BC_PRESSURE_OUTLET:
    case BC_VELOCITY_PERIODIC:
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                p[i][j][Nz+1] = 2*solver->bc_val[5] - p[i][j][Nz];
            }
        }
        break;
    case BC_ALL_PERIODIC:
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                p[i][j][Nz+1] = p[i][j][1];
            }
        }
        break;
    default:;
    }
}

static void adj_exchange(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    double (*const u1)[Ny+2][Nz+2] = solver->u1;
    double (*const u2)[Ny+2][Nz+2] = solver->u2;
    double (*const u3)[Ny+2][Nz+2] = solver->u3;
    double (*const p)[Ny+2][Nz+2] = solver->p;

    if (solver->rank != solver->num_process-1) {
        /* Send to next process. */
        MPI_Send(u1[Nx], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->rank+1, 0, MPI_COMM_WORLD);
        MPI_Send(u2[Nx], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->rank+1, 1, MPI_COMM_WORLD);
        MPI_Send(u3[Nx], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->rank+1, 2, MPI_COMM_WORLD);
        MPI_Send(p[Nx], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->rank+1, 3, MPI_COMM_WORLD);
    }
    if (solver->rank != 0) {
        /* Receive from previous process. */
        MPI_Recv(u1[0], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(u2[0], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(u3[0], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->rank-1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(p[0], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->rank-1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        /* Send to previous process. */
        MPI_Send(u1[1], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->rank-1, 0, MPI_COMM_WORLD);
        MPI_Send(u2[1], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->rank-1, 1, MPI_COMM_WORLD);
        MPI_Send(u3[1], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->rank-1, 2, MPI_COMM_WORLD);
        MPI_Send(p[1], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->rank-1, 3, MPI_COMM_WORLD);
    }
    if (solver->rank != solver->num_process-1) {
        /* Receive from next process. */
        MPI_Recv(u1[Nx+1], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(u2[Nx+1], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->rank+1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(u3[Nx+1], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->rank+1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(p[Nx+1], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->rank+1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

static void update_ghost_and_solid(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    const double (*const dx) = solver->dx;
    const double (*const dy) = solver->dy;
    const double (*const dz) = solver->dz;
    const double (*const xc) = solver->xc;
    const double (*const yc) = solver->yc;
    const double (*const zc) = solver->zc;

    const double (*const lvset)[Ny+2][Nz+2] = solver->lvset;
    const int (*const flag)[Ny+2][Nz+2] = solver->flag;

    double (*const u1)[Ny+2][Nz+2] = solver->u1;
    double (*const u2)[Ny+2][Nz+2] = solver->u2;
    double (*const u3)[Ny+2][Nz+2] = solver->u3;

    double (*const U1)[Ny+2][Nz+2] = solver->U1;
    double (*const U2)[Ny+1][Nz+2] = solver->U2;
    double (*const U3)[Ny+2][Nz+1] = solver->U3;

    double (*const p)[Ny+2][Nz+2] = solver->p;

    FOR_ALL_CELL (i, j, k) {
        if (flag[i][j][k] == FLAG_GHOST) {
            Vector n, m;

            n.x = (lvset[i+1][j][k] - lvset[i-1][j][k]) / (xc[i+1] - xc[i-1]);
            n.y = (lvset[i][j+1][k] - lvset[i][j-1][k]) / (yc[j+1] - yc[j-1]);
            n.z = (lvset[i][j][k+1] - lvset[i][j][k-1]) / (zc[k+1] - zc[k-1]);

            m = Vector_lincom(
                1, (Vector){xc[i], yc[j], zc[k]},
                -2*lvset[i][j][k], n
            );

            const int im = upper_bound(Nx+2, xc, m.x) - 1;
            const int jm = upper_bound(Ny+2, yc, m.y) - 1;
            const int km = upper_bound(Nz+2, zc, m.z) - 1;

            const double xl = xc[im], xu = xc[im+1];
            const double yl = yc[jm], yu = yc[jm+1];
            const double zl = zc[km], zu = zc[km+1];
            const double vol = (xu - xl) * (yu - yl) * (zu - zl);

            double interp_coeff[8];
            interp_coeff[0] = (xu-m.x)*(yu-m.y)*(zu-m.z) / vol;
            interp_coeff[1] = (xu-m.x)*(yu-m.y)*(m.z-zl) / vol;
            interp_coeff[2] = (xu-m.x)*(m.y-yl)*(zu-m.z) / vol;
            interp_coeff[3] = (xu-m.x)*(m.y-yl)*(m.z-zl) / vol;
            interp_coeff[4] = (m.x-xl)*(yu-m.y)*(zu-m.z) / vol;
            interp_coeff[5] = (m.x-xl)*(yu-m.y)*(m.z-zl) / vol;
            interp_coeff[6] = (m.x-xl)*(m.y-yl)*(zu-m.z) / vol;
            interp_coeff[7] = (m.x-xl)*(m.y-yl)*(m.z-zl) / vol;

            double center_coeff = 1;
            double coeff_sum = 0;
            double sum_u1 = 0, sum_u2 = 0, sum_u3 = 0;

            for (int l = 0; l < 8; l++) {
                int ni = im + !!(l & 4);
                int nj = jm + !!(l & 2);
                int nk = km + !!(l & 1);

                if (ni < 0 || ni > Nx+1 || nj < 0 || nj > Ny+1 || nk < 0 || nk > Nz+1) {
                    printf("error!\n");
                    continue;
                }
                if (flag[ni][nj][nk] == FLAG_SOLID) {
                    continue;
                }

                coeff_sum += interp_coeff[l];

                if (ni == i && nj == j && nk == k) {
                    center_coeff += interp_coeff[l];
                }
                else {
                    sum_u1 += u1[ni][nj][nk] * interp_coeff[l];
                    sum_u2 += u2[ni][nj][nk] * interp_coeff[l];
                    sum_u3 += u3[ni][nj][nk] * interp_coeff[l];
                }
            }

            u1[i][j][k] = -sum_u1 / center_coeff / coeff_sum;
            u2[i][j][k] = -sum_u2 / center_coeff / coeff_sum;
            u3[i][j][k] = -sum_u3 / center_coeff / coeff_sum;
        }
        else if (flag[i][j][k] == FLAG_SOLID) {
            u1[i][j][k] = u2[i][j][k] = u3[i][j][k] = p[i][j][k] = NAN;
        }
    }

    FOR_ALL_XSTAG (i, j, k) {
        if ((flag[i][j][k] == FLAG_FLUID && flag[i+1][j][k] == FLAG_GHOST) || (flag[i][j][k] == FLAG_GHOST && flag[i+1][j][k] == FLAG_FLUID)) {
            U1[i][j][k] = (u1[i][j][k] * dx[i+1] + u1[i+1][j][k] * dx[i]) / (dx[i] + dx[i+1]);
        }
        else if (flag[i][j][k] == FLAG_SOLID || flag[i+1][j][k] == FLAG_SOLID) {
            U1[i][j][k] = NAN;
        }
    }
    FOR_ALL_YSTAG (i, j, k) {
        if ((flag[i][j][k] == FLAG_FLUID && flag[i][j+1][k] == FLAG_GHOST) || (flag[i][j][k] == FLAG_GHOST && flag[i][j+1][k] == FLAG_FLUID)) {
            U2[i][j][k] = (u2[i][j][k] * dy[j+1] + u2[i][j+1][k] * dy[j]) / (dy[j] + dy[j+1]);
        }
        else if (flag[i][j][k] == FLAG_SOLID || flag[i][j+1][k] == FLAG_SOLID) {
            U2[i][j][k] = NAN;
        }
    }
    FOR_ALL_ZSTAG (i, j, k) {
        if ((flag[i][j][k] == FLAG_FLUID && flag[i][j][k+1] == FLAG_GHOST) || (flag[i][j][k] == FLAG_GHOST && flag[i][j][k+1] == FLAG_FLUID)) {
            U3[i][j][k] = (u3[i][j][k] * dz[k+1] + u3[i][j][k+1] * dz[k]) / (dz[k] + dz[k+1]);
        }
        else if (flag[i][j][k] == FLAG_SOLID || flag[i][j][k+1] == FLAG_SOLID) {
            U3[i][j][k] = NAN;
        }
    }
}
