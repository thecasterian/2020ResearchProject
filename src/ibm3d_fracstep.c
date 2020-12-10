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
static void update_outer(IBMSolver *);
static void adj_exchange(IBMSolver *);
static void update_ghost(IBMSolver *);

static double bc_val_u1(IBMSolver *, IBMSolverDirection, double, double, double);
static double bc_val_u2(IBMSolver *, IBMSolverDirection, double, double, double);
static double bc_val_u3(IBMSolver *, IBMSolverDirection, double, double, double);
static double bc_val_p(IBMSolver *, IBMSolverDirection, double, double, double);

/**
 * @brief Iterate solver for \p num_time_steps steps.
 *
 * @param solver IBMSolver.
 * @param num_time_steps Number of time steps to iterate.
 * @param verbose Print iteration info? (T/F)
 */
void IBMSolver_iterate(IBMSolver *solver, int num_time_steps, bool verbose) {
    struct timespec t_start, t_end;
    long elapsed_time, hour, min, sec;
    double final_norm_u1, final_norm_u2, final_norm_u3, final_norm_p;
    int i = 0;
    double start_time = solver->time;

    update_outer(solver);
    adj_exchange(solver);
    interp_stag_vel(solver);
    update_ghost(solver);

    calc_N(solver);
    SWAP(solver->N1, solver->N1_prev);
    SWAP(solver->N2, solver->N2_prev);
    SWAP(solver->N3, solver->N3_prev);

    if (verbose) {
        clock_gettime(CLOCK_REALTIME, &t_start);
    }

    while (i < num_time_steps) {
        calc_N(solver); break;
        calc_u_star(solver, &final_norm_u1, &final_norm_u2, &final_norm_u3);
        calc_u_tilde(solver);
        calc_U_star(solver);
        calc_p_prime(solver, &final_norm_p);
        update_next(solver);

        i++;
        solver->iter++;
        solver->time = start_time + solver->dt * i;

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
        if (solver->autosave_period > 0 && solver->iter % solver->autosave_period == 0) {
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

    const double *const u1 = solver->u1;
    const double *const u2 = solver->u2;
    const double *const u3 = solver->u3;

    const double *const U1 = solver->U1;
    const double *const U2 = solver->U2;
    const double *const U3 = solver->U3;

    double *const N1 = solver->N1;
    double *const N2 = solver->N2;
    double *const N3 = solver->N3;

    double u1_w, u1_e, u1_s, u1_n, u1_d, u1_u;
    double u2_w, u2_e, u2_s, u2_n, u2_d, u2_u;
    double u3_w, u3_e, u3_s, u3_n, u3_d, u3_u;

    FOR_INNER_CELL (i, j, k) {
        if (c3e(solver->flag, i, j, k) == FLAG_FLUID) {
            u1_w = (c3e(u1, i-1, j, k)*c1e(dx, i) + c3e(u1, i, j, k)*c1e(dx, i-1))
                / (c1e(dx, i-1) + c1e(dx, i));
            u2_w = (c3e(u2, i-1, j, k)*c1e(dx, i) + c3e(u2, i, j, k)*c1e(dx, i-1))
                / (c1e(dx, i-1) + c1e(dx, i));
            u3_w = (c3e(u3, i-1, j, k)*c1e(dx, i) + c3e(u3, i, j, k)*c1e(dx, i-1))
                / (c1e(dx, i-1) + c1e(dx, i));

            u1_e = (c3e(u1, i, j, k)*c1e(dx, i+1) + c3e(u1, i+1, j, k)*c1e(dx, i))
                / (c1e(dx, i) + c1e(dx, i+1));
            u2_e = (c3e(u2, i, j, k)*c1e(dx, i+1) + c3e(u2, i+1, j, k)*c1e(dx, i))
                / (c1e(dx, i) + c1e(dx, i+1));
            u3_e = (c3e(u3, i, j, k)*c1e(dx, i+1) + c3e(u3, i+1, j, k)*c1e(dx, i))
                / (c1e(dx, i) + c1e(dx, i+1));

            u1_s = (c3e(u1, i, j-1, k)*c1e(dy, j) + c3e(u1, i, j, k)*c1e(dy, j-1))
                / (c1e(dy, j-1) + c1e(dy, j));
            u2_s = (c3e(u2, i, j-1, k)*c1e(dy, j) + c3e(u2, i, j, k)*c1e(dy, j-1))
                / (c1e(dy, j-1) + c1e(dy, j));
            u3_s = (c3e(u3, i, j-1, k)*c1e(dy, j) + c3e(u3, i, j, k)*c1e(dy, j-1))
                / (c1e(dy, j-1) + c1e(dy, j));

            u1_n = (c3e(u1, i, j, k)*c1e(dy, j+1) + c3e(u1, i, j+1, k)*c1e(dy, j))
                / (c1e(dy, j) + c1e(dy, j+1));
            u2_n = (c3e(u2, i, j, k)*c1e(dy, j+1) + c3e(u2, i, j+1, k)*c1e(dy, j))
                / (c1e(dy, j) + c1e(dy, j+1));
            u3_n = (c3e(u3, i, j, k)*c1e(dy, j+1) + c3e(u3, i, j+1, k)*c1e(dy, j))
                / (c1e(dy, j) + c1e(dy, j+1));

            u1_d = (c3e(u1, i, j, k-1)*c1e(dz, k) + c3e(u1, i, j, k)*c1e(dz, k-1))
                / (c1e(dz, k-1) + c1e(dz, k));
            u2_d = (c3e(u2, i, j, k-1)*c1e(dz, k) + c3e(u2, i, j, k)*c1e(dz, k-1))
                / (c1e(dz, k-1) + c1e(dz, k));
            u3_d = (c3e(u3, i, j, k-1)*c1e(dz, k) + c3e(u3, i, j, k)*c1e(dz, k-1))
                / (c1e(dz, k-1) + c1e(dz, k));

            u1_u = (c3e(u1, i, j, k)*c1e(dz, k+1) + c3e(u1, i, j, k+1)*c1e(dz, k))
                / (c1e(dz, k) + c1e(dz, k+1));
            u2_u = (c3e(u2, i, j, k)*c1e(dz, k+1) + c3e(u2, i, j, k+1)*c1e(dz, k))
                / (c1e(dz, k) + c1e(dz, k+1));
            u3_u = (c3e(u3, i, j, k)*c1e(dz, k+1) + c3e(u3, i, j, k+1)*c1e(dz, k))
                / (c1e(dz, k) + c1e(dz, k+1));

            /* Ni = d(U1ui)/dx + d(U2ui)/dy + d(U3ui)/dz */
            c3e(N1, i, j, k) = (xse(U1, i+1, j, k)*u1_e - xse(U1, i, j, k)*u1_w) / c1e(dx, i)
                + (yse(U2, i, j+1, k)*u1_n - yse(U2, i, j, k)*u1_s) / c1e(dy, j)
                + (zse(U3, i, j, k+1)*u1_u - zse(U3, i, j, k)*u1_d) / c1e(dz, k);
            c3e(N2, i, j, k) = (xse(U1, i+1, j, k)*u2_e - xse(U1, i, j, k)*u2_w) / c1e(dx, i)
                + (yse(U2, i, j+1, k)*u2_n - yse(U2, i, j, k)*u2_s) / c1e(dy, j)
                + (zse(U3, i, j, k+1)*u2_u - zse(U3, i, j, k)*u2_d) / c1e(dz, k);
            c3e(N3, i, j, k) = (xse(U1, i+1, j, k)*u3_e - xse(U1, i, j, k)*u3_w) / c1e(dx, i)
                + (yse(U2, i, j+1, k)*u3_n - yse(U2, i, j, k)*u3_s) / c1e(dy, j)
                + (zse(U3, i, j, k+1)*u3_u - zse(U3, i, j, k)*u3_d) / c1e(dz, k);
        }
        else {
            c3e(N1, i, j, k) = c3e(N2, i, j, k) = c3e(N3, i, j, k) = NAN;
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

    const double *const u1 = solver->u1;
    const double *const u2 = solver->u2;
    const double *const u3 = solver->u3;

    double *const u1_star = solver->u1_star;
    double *const u2_star = solver->u2_star;
    double *const u3_star = solver->u3_star;

    const double *const p = solver->p;

    const double *const N1 = solver->N1;
    const double *const N2 = solver->N2;
    const double *const N3 = solver->N3;
    const double *const N1_prev = solver->N1_prev;
    const double *const N2_prev = solver->N2_prev;
    const double *const N3_prev = solver->N3_prev;

    const double xmin = solver->xmin, xmax = solver->xmax;
    const double ymin = solver->ymin, ymax = solver->ymax;
    const double zmin = solver->zmin, zmax = solver->zmax;

    int ifirst, ilast, jfirst, jlast, kfirst, klast;
    int idx;

    int hypre_ierr = 0;

    ifirst = solver->ri == 0 ? -2 : 0;
    ilast = solver->ri != solver->Px-1 ? Nx : Nx+2;
    jfirst = solver->rj == 0 ? -2 : 0;
    jlast = solver->rj != solver->Py-1 ? Ny : Ny+2;
    kfirst = solver->rk == 0 ? -2 : 0;
    klast = solver->rk != solver->Pz-1 ? Nz : Nz+2;

    /* u1_star. */

    memcpy(solver->vector_values, solver->vector_zeros, sizeof(double)*(solver->idx_last-solver->idx_first));

    FOR_INNER_CELL (i, j, k) {
        idx = c3e(solver->cell_idx, i, j, k) - solver->idx_first;
        if (c3e(solver->flag, i, j, k) == FLAG_FLUID) {
            solver->vector_values[idx]
                = -dt/2 * (3*c3e(N1, i, j, k) - c3e(N1_prev, i, j, k))
                - dt * (c3e(p, i+1, j, k) - c3e(p, i-1, j, k)) / (c1e(xc, i+1) - c1e(xc, i-1))
                + (1-c1e(kx_W, i)-c1e(kx_E, i)-c1e(ky_S, j)-c1e(ky_N, j)-c1e(kz_D, k)-c1e(kz_U, k))*c3e(u1, i, j, k)
                + c1e(kx_W, i)*c3e(u1, i-1, j, k) + c1e(kx_E, i)*c3e(u1, i+1, j, k)
                + c1e(ky_S, j)*c3e(u1, i, j-1, k) + c1e(ky_N, j)*c3e(u1, i, j+1, k)
                + c1e(kz_D, k)*c3e(u1, i, j, k-1) + c1e(kz_U, k)*c3e(u1, i, j, k+1);
        }
    }

    if (solver->ri == 0) {
        switch (solver->bc[3].type) {
        case BC_VELOCITY_COMPONENT:
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    solver->vector_values[c3e(solver->cell_idx, -1, j, k) - solver->idx_first]
                        = solver->vector_values[c3e(solver->cell_idx, -2, j, k) - solver->idx_first]
                        = bc_val_u1(solver, DIR_WEST, xmin, c1e(yc, j), c1e(zc, k));
                }
            }
            break;
        default:;
        }
    }
    if (solver->ri == solver->Px-1) {
        switch (solver->bc[1].type) {
        case BC_VELOCITY_COMPONENT:
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    solver->vector_values[c3e(solver->cell_idx, Nx, j, k) - solver->idx_first]
                        = solver->vector_values[c3e(solver->cell_idx, Nx+1, j, k) - solver->idx_first]
                        = bc_val_u1(solver, DIR_EAST, xmax, c1e(yc, j), c1e(zc, k));
                }
            }
            break;
        default:;
        }
    }
    if (solver->rj == 0) {
        switch (solver->bc[2].type) {
        case BC_VELOCITY_COMPONENT:
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    solver->vector_values[c3e(solver->cell_idx, i, -1, k) - solver->idx_first]
                        = solver->vector_values[c3e(solver->cell_idx, i, -2, k) - solver->idx_first]
                        = bc_val_u1(solver, DIR_SOUTH, c1e(xc, i), ymin, c1e(zc, k));
                }
            }
        default:;
        }
    }
    if (solver->rj == solver->Py-1) {
        switch (solver->bc[0].type) {
        case BC_VELOCITY_COMPONENT:
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    solver->vector_values[c3e(solver->cell_idx, i, Ny, k) - solver->idx_first]
                        = solver->vector_values[c3e(solver->cell_idx, i, Ny+1, k) - solver->idx_first]
                        = bc_val_u1(solver, DIR_NORTH, c1e(xc, i), ymax, c1e(zc, k));
                }
            }
        default:;
        }
    }
    if (solver->rk == 0) {
        switch (solver->bc[4].type) {
        case BC_VELOCITY_COMPONENT:
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    solver->vector_values[c3e(solver->cell_idx, i, j, -1) - solver->idx_first]
                        = solver->vector_values[c3e(solver->cell_idx, i, j, -2) - solver->idx_first]
                        = bc_val_u1(solver, DIR_DOWN, c1e(xc, i), c1e(yc, j), zmin);
                }
            }
            break;
        default:;
        }
    }
    if (solver->rk == solver->Pz-1) {
        switch (solver->bc[5].type) {
        case BC_VELOCITY_COMPONENT:
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    solver->vector_values[c3e(solver->cell_idx, i, j, Nz) - solver->idx_first]
                        = solver->vector_values[c3e(solver->cell_idx, i, j, Nz+1) - solver->idx_first]
                        = bc_val_u1(solver, DIR_UP, c1e(xc, i), c1e(yc, j), zmax);
                }
            }
            break;
        default:;
        }
    }

    HYPRE_IJVectorSetValues(solver->b, solver->idx_last-solver->idx_first, solver->vector_rows, solver->vector_values);
    HYPRE_IJVectorSetValues(solver->x, solver->idx_last-solver->idx_first, solver->vector_rows, solver->vector_zeros);
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

    HYPRE_IJVectorGetValues(solver->x, solver->idx_last-solver->idx_first, solver->vector_rows, solver->vector_res);
    for (int i = ifirst; i < ilast; i++) {
        for (int j = jfirst; j < jlast; j++) {
            for (int k = kfirst; k < klast; k++) {
                c3e(u1_star, i, j, k) = solver->vector_res[c3e(solver->cell_idx, i, j, k)-solver->idx_first];
            }
        }
    }

    /* u2_star. */

    memcpy(solver->vector_values, solver->vector_zeros, sizeof(double)*(solver->idx_last-solver->idx_first));

    FOR_INNER_CELL (i, j, k) {
        idx = c3e(solver->cell_idx, i, j, k) - solver->idx_first;
        if (c3e(solver->flag, i, j, k) == FLAG_FLUID) {
            solver->vector_values[idx]
                = -dt/2 * (3*c3e(N2, i, j, k) - c3e(N2_prev, i, j, k))
                - dt * (c3e(p, i, j+1, k) - c3e(p, i, j-1, k)) / (c1e(yc, j+1) - c1e(yc, j-1))
                + (1-c1e(kx_W, i)-c1e(kx_E, i)-c1e(ky_S, j)-c1e(ky_N, j)-c1e(kz_D, k)-c1e(kz_U, k))*c3e(u2, i, j, k)
                + c1e(kx_W, i)*c3e(u2, i-1, j, k) + c1e(kx_E, i)*c3e(u2, i+1, j, k)
                + c1e(ky_S, j)*c3e(u2, i, j-1, k) + c1e(ky_N, j)*c3e(u2, i, j+1, k)
                + c1e(kz_D, k)*c3e(u2, i, j, k-1) + c1e(kz_U, k)*c3e(u2, i, j, k+1);
        }
    }

    if (solver->ri == 0) {
        switch (solver->bc[3].type) {
        case BC_VELOCITY_COMPONENT:
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    solver->vector_values[c3e(solver->cell_idx, -1, j, k) - solver->idx_first]
                        = solver->vector_values[c3e(solver->cell_idx, -2, j, k) - solver->idx_first]
                        = bc_val_u2(solver, DIR_WEST, xmin, c1e(yc, j), c1e(zc, k));
                }
            }
            break;
        default:;
        }
    }
    if (solver->ri == solver->Px-1) {
        switch (solver->bc[1].type) {
        case BC_VELOCITY_COMPONENT:
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    solver->vector_values[c3e(solver->cell_idx, Nx, j, k) - solver->idx_first]
                        = solver->vector_values[c3e(solver->cell_idx, Nx+1, j, k) - solver->idx_first]
                        = bc_val_u2(solver, DIR_EAST, xmax, c1e(yc, j), c1e(zc, k));
                }
            }
            break;
        default:;
        }
    }
    if (solver->rj == 0) {
        switch (solver->bc[2].type) {
        case BC_VELOCITY_COMPONENT:
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    solver->vector_values[c3e(solver->cell_idx, i, -1, k) - solver->idx_first]
                        = solver->vector_values[c3e(solver->cell_idx, i, -2, k) - solver->idx_first]
                        = bc_val_u2(solver, DIR_SOUTH, c1e(xc, i), ymin, c1e(zc, k));
                }
            }
        default:;
        }
    }
    if (solver->rj == solver->Py-1) {
        switch (solver->bc[0].type) {
        case BC_VELOCITY_COMPONENT:
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    solver->vector_values[c3e(solver->cell_idx, i, Ny, k) - solver->idx_first]
                        = solver->vector_values[c3e(solver->cell_idx, i, Ny+1, k) - solver->idx_first]
                        = bc_val_u2(solver, DIR_NORTH, c1e(xc, i), ymax, c1e(zc, k));
                }
            }
        default:;
        }
    }
    if (solver->rk == 0) {
        switch (solver->bc[4].type) {
        case BC_VELOCITY_COMPONENT:
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    solver->vector_values[c3e(solver->cell_idx, i, j, -1) - solver->idx_first]
                        = solver->vector_values[c3e(solver->cell_idx, i, j, -2) - solver->idx_first]
                        = bc_val_u2(solver, DIR_DOWN, c1e(xc, i), c1e(yc, j), zmin);
                }
            }
            break;
        default:;
        }
    }
    if (solver->rk == solver->Pz-1) {
        switch (solver->bc[5].type) {
        case BC_VELOCITY_COMPONENT:
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    solver->vector_values[c3e(solver->cell_idx, i, j, Nz) - solver->idx_first]
                        = solver->vector_values[c3e(solver->cell_idx, i, j, Nz+1) - solver->idx_first]
                        = bc_val_u2(solver, DIR_UP, c1e(xc, i), c1e(yc, j), zmax);
                }
            }
            break;
        default:;
        }
    }

    HYPRE_IJVectorSetValues(solver->b, solver->idx_last-solver->idx_first, solver->vector_rows, solver->vector_values);
    HYPRE_IJVectorSetValues(solver->x, solver->idx_last-solver->idx_first, solver->vector_rows, solver->vector_zeros);
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

    HYPRE_IJVectorGetValues(solver->x, solver->idx_last-solver->idx_first, solver->vector_rows, solver->vector_res);
    for (int i = ifirst; i < ilast; i++) {
        for (int j = jfirst; j < jlast; j++) {
            for (int k = kfirst; k < klast; k++) {
                c3e(u2_star, i, j, k) = solver->vector_res[c3e(solver->cell_idx, i, j, k)-solver->idx_first];
            }
        }
    }

    /* u3_star. */

    memcpy(solver->vector_values, solver->vector_zeros, sizeof(double)*(solver->idx_last-solver->idx_first));

    FOR_INNER_CELL (i, j, k) {
        idx = c3e(solver->cell_idx, i, j, k) - solver->idx_first;
        if (c3e(solver->flag, i, j, k) == FLAG_FLUID) {
            solver->vector_values[idx]
                = -dt/2 * (3*c3e(N3, i, j, k) - c3e(N3_prev, i, j, k))
                - dt * (c3e(p, i, j, k+1) - c3e(p, i, j, k-1)) / (c1e(zc, k+1) - c1e(zc, k-1))
                + (1-c1e(kx_W, i)-c1e(kx_E, i)-c1e(ky_S, j)-c1e(ky_N, j)-c1e(kz_D, k)-c1e(kz_U, k))*c3e(u3, i, j, k)
                + c1e(kx_W, i)*c3e(u3, i-1, j, k) + c1e(kx_E, i)*c3e(u3, i+1, j, k)
                + c1e(ky_S, j)*c3e(u3, i, j-1, k) + c1e(ky_N, j)*c3e(u3, i, j+1, k)
                + c1e(kz_D, k)*c3e(u3, i, j, k-1) + c1e(kz_U, k)*c3e(u3, i, j, k+1);
        }
    }

    if (solver->ri == 0) {
        switch (solver->bc[3].type) {
        case BC_VELOCITY_COMPONENT:
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    solver->vector_values[c3e(solver->cell_idx, -1, j, k) - solver->idx_first]
                        = solver->vector_values[c3e(solver->cell_idx, -2, j, k) - solver->idx_first]
                        = bc_val_u3(solver, DIR_WEST, xmin, c1e(yc, j), c1e(zc, k));
                }
            }
            break;
        default:;
        }
    }
    if (solver->ri == solver->Px-1) {
        switch (solver->bc[1].type) {
        case BC_VELOCITY_COMPONENT:
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    solver->vector_values[c3e(solver->cell_idx, Nx, j, k) - solver->idx_first]
                        = solver->vector_values[c3e(solver->cell_idx, Nx+1, j, k) - solver->idx_first]
                        = bc_val_u3(solver, DIR_EAST, xmax, c1e(yc, j), c1e(zc, k));
                }
            }
            break;
        default:;
        }
    }
    if (solver->rj == 0) {
        switch (solver->bc[2].type) {
        case BC_VELOCITY_COMPONENT:
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    solver->vector_values[c3e(solver->cell_idx, i, -1, k) - solver->idx_first]
                        = solver->vector_values[c3e(solver->cell_idx, i, -2, k) - solver->idx_first]
                        = bc_val_u3(solver, DIR_SOUTH, c1e(xc, i), ymin, c1e(zc, k));
                }
            }
        default:;
        }
    }
    if (solver->rj == solver->Py-1) {
        switch (solver->bc[0].type) {
        case BC_VELOCITY_COMPONENT:
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    solver->vector_values[c3e(solver->cell_idx, i, Ny, k) - solver->idx_first]
                        = solver->vector_values[c3e(solver->cell_idx, i, Ny+1, k) - solver->idx_first]
                        = bc_val_u3(solver, DIR_NORTH, c1e(xc, i), ymax, c1e(zc, k));
                }
            }
        default:;
        }
    }
    if (solver->rk == 0) {
        switch (solver->bc[4].type) {
        case BC_VELOCITY_COMPONENT:
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    solver->vector_values[c3e(solver->cell_idx, i, j, -1) - solver->idx_first]
                        = solver->vector_values[c3e(solver->cell_idx, i, j, -2) - solver->idx_first]
                        = bc_val_u3(solver, DIR_DOWN, c1e(xc, i), c1e(yc, j), zmin);
                }
            }
            break;
        default:;
        }
    }
    if (solver->rk == solver->Pz-1) {
        switch (solver->bc[5].type) {
        case BC_VELOCITY_COMPONENT:
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    solver->vector_values[c3e(solver->cell_idx, i, j, Nz) - solver->idx_first]
                        = solver->vector_values[c3e(solver->cell_idx, i, j, Nz+1) - solver->idx_first]
                        = bc_val_u3(solver, DIR_UP, c1e(xc, i), c1e(yc, j), zmax);
                }
            }
            break;
        default:;
        }
    }

    HYPRE_IJVectorSetValues(solver->b, solver->idx_last-solver->idx_first, solver->vector_rows, solver->vector_values);
    HYPRE_IJVectorSetValues(solver->x, solver->idx_last-solver->idx_first, solver->vector_rows, solver->vector_zeros);
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

    HYPRE_IJVectorGetValues(solver->x, solver->idx_last-solver->idx_first, solver->vector_rows, solver->vector_res);
    for (int i = ifirst; i < ilast; i++) {
        for (int j = jfirst; j < jlast; j++) {
            for (int k = kfirst; k < klast; k++) {
                c3e(u3_star, i, j, k) = solver->vector_res[c3e(solver->cell_idx, i, j, k)-solver->idx_first];
            }
        }
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

    FOR_INNER_CELL (i, j, k) {
        if (flag[i][j][k] == FLAG_FLUID) {
            u1_tilde[i][j][k] = u1_star[i][j][k] + dt * (p[i+1][j][k] - p[i-1][j][k]) / (xc[i+1] - xc[i-1]);
            u2_tilde[i][j][k] = u2_star[i][j][k] + dt * (p[i][j+1][k] - p[i][j-1][k]) / (yc[j+1] - yc[j-1]);
            u3_tilde[i][j][k] = u3_star[i][j][k] + dt * (p[i][j][k+1] - p[i][j][k-1]) / (zc[k+1] - zc[k-1]);
        }
    }

    /* West. */
    if (solver->ilower == 1) {
        switch (solver->bc[3].type) {
        case BC_PRESSURE:
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
        switch (solver->bc[1].type) {
        case BC_PRESSURE:
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
    switch (solver->bc[2].type) {
    case BC_PRESSURE:
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
    switch (solver->bc[0].type) {
    case BC_PRESSURE:
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
    switch (solver->bc[4].type) {
    case BC_PRESSURE:
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
    switch (solver->bc[5].type) {
    case BC_PRESSURE:
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

    const double xmin = solver->xmin, xmax = solver->xmax;
    const double ymin = solver->ymin, ymax = solver->ymax;
    const double zmin = solver->zmin, zmax = solver->zmax;

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
        switch (solver->bc[3].type) {
        case BC_VELOCITY_COMPONENT:
            for (int j = 1; j <= Ny; j++) {
                for (int k = 1; k <= Nz; k++) {
                    U1_star[0][j][k] = bc_val_u1(solver, DIR_WEST, xmin, yc[j], zc[k]);
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
        switch (solver->bc[1].type) {
        case BC_VELOCITY_COMPONENT:
            for (int j = 1; j <= Ny; j++) {
                for (int k = 1; k <= Nz; k++) {
                    U1_star[Nx][j][k] = bc_val_u1(solver, DIR_EAST, xmax, yc[j], zc[k]);
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
    switch (solver->bc[2].type) {
    case BC_VELOCITY_COMPONENT:
        for (int i = 1; i <= Nx; i++) {
            for (int k = 1; k <= Nz; k++) {
                U2_star[i][0][k] = bc_val_u2(solver, DIR_SOUTH, xc[i], ymin, zc[k]);
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
    switch (solver->bc[0].type) {
    case BC_VELOCITY_COMPONENT:
        for (int i = 1; i <= Nx; i++) {
            for (int k = 1; k <= Nz; k++) {
                U2_star[i][Ny][k] = bc_val_u2(solver, DIR_NORTH, xc[i], ymax, zc[k]);
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
    switch (solver->bc[4].type) {
    case BC_VELOCITY_COMPONENT:
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                U3_star[i][j][0] = bc_val_u3(solver, DIR_DOWN, xc[i], yc[j], zmin);
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
    switch (solver->bc[5].type) {
    case BC_VELOCITY_COMPONENT:
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                U3_star[i][j][Nz] = bc_val_u3(solver, DIR_UP, xc[i], yc[j], zmax);
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

    /* U_star between a fluid cell and a ghost cell. */
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

    // FOR_INNER_CELL (i, j, k) {
    //     double *const rhs = &solver->vector_values[LOCL_CELL_IDX(i, j, k)-1];
    //     if (flag[i][j][k] == 1) {
    //         *rhs = -1/(2*Re) * (
    //                 (U1_star[i][j][k] - U1_star[i-1][j][k]) / dx[i]
    //                 + (U2_star[i][j][k] - U2_star[i][j-1][k]) / dy[j]
    //                 + (U3_star[i][j][k] - U3_star[i][j][k-1]) / dz[k]
    //             );
    //         *rhs /= p_coeffsum[i][j][k];
    //     }
    //     else {
    //         *rhs = 0;
    //     }
    // }

    // HYPRE_IJVectorSetValues(solver->b, Nx*Ny*Nz, solver->vector_rows, solver->vector_values);
    // HYPRE_IJVectorSetValues(solver->x, Nx*Ny*Nz, solver->vector_rows, solver->vector_zeros);

    // HYPRE_IJVectorAssemble(solver->b);
    // HYPRE_IJVectorAssemble(solver->x);

    // HYPRE_IJVectorGetObject(solver->b, (void **)&solver->par_b);
    // HYPRE_IJVectorGetObject(solver->x, (void **)&solver->par_x);

    // switch (solver->linear_solver_type) {
    // case SOLVER_AMG:
    //     hypre_ierr = HYPRE_BoomerAMGSolve(solver->linear_solver_p, solver->parcsr_A_p, solver->par_b, solver->par_x);
    //     break;
    // case SOLVER_PCG:
    //     hypre_ierr = HYPRE_ParCSRPCGSolve(solver->linear_solver_p, solver->parcsr_A_p, solver->par_b, solver->par_x);
    //     break;
    // case SOLVER_BiCGSTAB:
    //     hypre_ierr = HYPRE_ParCSRBiCGSTABSolve(solver->linear_solver_p, solver->parcsr_A_p, solver->par_b, solver->par_x);
    //     break;
    // case SOLVER_GMRES:
    //     hypre_ierr = HYPRE_ParCSRGMRESSolve(solver->linear_solver_p, solver->parcsr_A_p, solver->par_b, solver->par_x);
    //     break;
    // default:;
    // }
    // if (hypre_ierr) {
    //     printf("error: floating pointer error raised in p_prime\n");
    //     MPI_Abort(MPI_COMM_WORLD, -1);
    // }

    // HYPRE_IJVectorGetValues(solver->x, Nx*Ny*Nz, solver->vector_rows, solver->vector_res);
    // FOR_INNER_CELL (i, j, k) {
    //     p_prime[i][j][k] = solver->vector_res[LOCL_CELL_IDX(i, j, k)-1];
    // }

    // switch (solver->linear_solver_type) {
    // case SOLVER_AMG:
    //     HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver->linear_solver_p, final_norm_p);
    //     break;
    // case SOLVER_PCG:
    //     HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(solver->linear_solver_p, final_norm_p);
    //     break;
    // case SOLVER_BiCGSTAB:
    //     HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm(solver->linear_solver_p, final_norm_p);
    //     break;
    // case SOLVER_GMRES:
    //     HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(solver->linear_solver_p, final_norm_p);
    //     break;
    // default:;
    // }

    // /* West. */
    // if (solver->ilower == 1) {
    //     switch (solver->bc[3].type) {
    //     case BC_VELOCITY_INLET:
    //     case BC_STATIONARY_WALL:
    //     case BC_FREE_SLIP_WALL:
    //         for (int j = 1; j <= Ny; j++) {
    //             for (int k = 1; k <= Nz; k++) {
    //                 p_prime[0][j][k] = p_prime[1][j][k];
    //             }
    //         }
    //         break;
    //     case BC_PRESSURE_OUTLET:
    //     case BC_VELOCITY_PERIODIC:
    //         for (int j = 1; j <= Ny; j++) {
    //             for (int k = 1; k <= Nz; k++) {
    //                 p_prime[0][j][k] = -p_prime[1][j][k];
    //             }
    //         }
    //         break;
    //     case BC_ALL_PERIODIC:
    //         if (solver->num_process == 1) {
    //             for (int j = 1; j <= Ny; j++) {
    //                 for (int k = 1; k <= Nz; k++) {
    //                     p_prime[0][j][k] = p_prime[Nx][j][k];
    //                 }
    //             }
    //         }
    //         else {
    //             MPI_Send(p_prime[1], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-1, 0, MPI_COMM_WORLD);
    //             MPI_Recv(p_prime[0], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->num_process-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //         }
    //         break;
    //     default:;
    //     }
    // }

    // /* East. */
    // if (solver->iupper == Nx_global) {
    //     switch (solver->bc[1].type) {
    //     case BC_VELOCITY_INLET:
    //     case BC_STATIONARY_WALL:
    //     case BC_FREE_SLIP_WALL:
    //         for (int j = 1; j <= Ny; j++) {
    //             for (int k = 1; k <= Nz; k++) {
    //                 p_prime[Nx+1][j][k] = p_prime[Nx][j][k];
    //             }
    //         }
    //         break;
    //     case BC_PRESSURE_OUTLET:
    //     case BC_VELOCITY_PERIODIC:
    //         for (int j = 1; j <= Ny; j++) {
    //             for (int k = 1; k <= Nz; k++) {
    //                 p_prime[Nx+1][j][k] = -p_prime[Nx][j][k];
    //             }
    //         }
    //         break;
    //     case BC_ALL_PERIODIC:
    //         if (solver->num_process == 1) {
    //             for (int j = 1; j <= Ny; j++) {
    //                 for (int k = 1; k <= Nz; k++) {
    //                     p_prime[Nx+1][j][k] = p_prime[1][j][k];
    //                 }
    //             }
    //         }
    //         else {
    //             MPI_Recv(p_prime[Nx+1], (Ny+2)*(Nz+2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //             MPI_Send(p_prime[Nx], (Ny+2)*(Nz+2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    //         }
    //         break;
    //     default:;
    //     }
    // }

    // /* South. */
    // switch (solver->bc[2].type) {
    // case BC_VELOCITY_INLET:
    // case BC_STATIONARY_WALL:
    // case BC_FREE_SLIP_WALL:
    //     for (int i = 0; i <= Nx+1; i++) {
    //         for (int k = 1; k <= Nz; k++) {
    //             p_prime[i][0][k] = p_prime[i][1][k];
    //         }
    //     }
    //     break;
    // case BC_PRESSURE_OUTLET:
    // case BC_VELOCITY_PERIODIC:
    //     for (int i = 0; i <= Nx+1; i++) {
    //         for (int k = 1; k <= Nz; k++) {
    //             p_prime[i][0][k] = -p_prime[i][1][k];
    //         }
    //     }
    //     break;
    // case BC_ALL_PERIODIC:
    //     for (int i = 0; i <= Nx+1; i++) {
    //         for (int k = 1; k <= Nz; k++) {
    //             p_prime[i][0][k] = p_prime[i][Ny][k];
    //         }
    //     }
    //     break;
    // default:;
    // }

    // /* North. */
    // switch (solver->bc[0].type) {
    // case BC_VELOCITY_INLET:
    // case BC_STATIONARY_WALL:
    // case BC_FREE_SLIP_WALL:
    //     for (int i = 0; i <= Nx+1; i++) {
    //         for (int k = 1; k <= Nz; k++) {
    //             p_prime[i][Ny+1][k] = p_prime[i][Ny][k];
    //         }
    //     }
    //     break;
    // case BC_PRESSURE_OUTLET:
    // case BC_VELOCITY_PERIODIC:
    //     for (int i = 0; i <= Nx+1; i++) {
    //         for (int k = 1; k <= Nz; k++) {
    //             p_prime[i][Ny+1][k] = -p_prime[i][Ny][k];
    //         }
    //     }
    //     break;
    // case BC_ALL_PERIODIC:
    //     for (int i = 0; i <= Nx+1; i++) {
    //         for (int k = 1; k <= Nz; k++) {
    //             p_prime[i][Ny+1][k] = p_prime[i][1][k];
    //         }
    //     }
    //     break;
    // default:;
    // }

    // /* Down. */
    // switch (solver->bc[4].type) {
    // case BC_VELOCITY_INLET:
    // case BC_STATIONARY_WALL:
    // case BC_FREE_SLIP_WALL:
    //     for (int i = 0; i <= Nx+1; i++) {
    //         for (int j = 0; j <= Ny+1; j++) {
    //             p_prime[i][j][0] = p_prime[i][j][1];
    //         }
    //     }
    //     break;
    // case BC_PRESSURE_OUTLET:
    // case BC_VELOCITY_PERIODIC:
    //     for (int i = 0; i <= Nx+1; i++) {
    //         for (int j = 0; j <= Ny+1; j++) {
    //             p_prime[i][j][0] = -p_prime[i][j][1];
    //         }
    //     }
    //     break;
    // case BC_ALL_PERIODIC:
    //     for (int i = 0; i <= Nx+1; i++) {
    //         for (int j = 0; j <= Ny+1; j++) {
    //             p_prime[i][j][0] = p_prime[i][j][Nz];
    //         }
    //     }
    //     break;
    // default:;
    // }

    // /* Up. */
    // switch (solver->bc[5].type) {
    // case BC_VELOCITY_INLET:
    // case BC_STATIONARY_WALL:
    // case BC_FREE_SLIP_WALL:
    //     for (int i = 0; i <= Nx+1; i++) {
    //         for (int j = 0; j <= Ny+1; j++) {
    //             p_prime[i][j][Nz+1] = p_prime[i][j][Nz];
    //         }
    //     }
    //     break;
    // case BC_PRESSURE_OUTLET:
    // case BC_VELOCITY_PERIODIC:
    //     for (int i = 0; i <= Nx+1; i++) {
    //         for (int j = 0; j <= Ny+1; j++) {
    //             p_prime[i][j][Nz+1] = -p_prime[i][j][Nz];
    //         }
    //     }
    //     break;
    // case BC_ALL_PERIODIC:
    //     for (int i = 0; i <= Nx+1; i++) {
    //         for (int j = 0; j <= Ny+1; j++) {
    //             p_prime[i][j][Nz+1] = p_prime[i][j][1];
    //         }
    //     }
    //     break;
    // default:;
    // }

    // /* Exchange p_prime between the adjacent processes. */
    // if (solver->rank != solver->num_process-1) {
    //     /* Send to next process. */
    //     MPI_Send(p_prime[Nx], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->rank+1, 0, MPI_COMM_WORLD);
    // }
    // if (solver->rank != 0) {
    //     /* Receive from previous process. */
    //     MPI_Recv(p_prime[0], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //     /* Send to previous process. */
    //     MPI_Send(p_prime[1], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->rank-1, 0, MPI_COMM_WORLD);
    // }
    // if (solver->rank != solver->num_process-1) {
    //     /* Receive from next process. */
    //     MPI_Recv(p_prime[Nx+1], (Ny+2)*(Nz+2), MPI_DOUBLE, solver->rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // }
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
    FOR_INNER_CELL (i, j, k) {
        if (flag[i][j][k] == FLAG_FLUID || flag[i][j][k] == FLAG_GHOST) {
            p_next[i][j][k] = p[i][j][k] + p_prime[i][j][k];
        }
        else if (flag[i][j][k] == FLAG_SOLID) {
            p_next[i][j][k] = NAN;
        }
    }

    /* Calculate u_next. */
    FOR_INNER_CELL (i, j, k) {
        if (flag[i][j][k] == FLAG_FLUID) {
            u1_next[i][j][k] = u1_star[i][j][k] - dt * (p_prime[i+1][j][k] - p_prime[i-1][j][k]) / (xc[i+1] - xc[i-1]);
            u2_next[i][j][k] = u2_star[i][j][k] - dt * (p_prime[i][j+1][k] - p_prime[i][j-1][k]) / (yc[j+1] - yc[j-1]);
            u3_next[i][j][k] = u3_star[i][j][k] - dt * (p_prime[i][j][k+1] - p_prime[i][j][k-1]) / (zc[k+1] - zc[k-1]);
        }
        else if (flag[i][j][k] == FLAG_SOLID) {
            u1_next[i][j][k] = u2_next[i][j][k] = u3_next[i][j][k] = NAN;
        }
    }

    /* Calculate U_next. */
    FOR_ALL_XSTAG (i, j, k) {
        if (flag[i][j][k] == FLAG_FLUID && flag[i+1][j][k] == FLAG_FLUID) {
            U1_next[i][j][k] = U1_star[i][j][k] - dt * (p_prime[i+1][j][k] - p_prime[i][j][k]) / (xc[i+1] - xc[i]);
        }
        else if (flag[i][j][k] == FLAG_SOLID || flag[i+1][j][k] == FLAG_SOLID) {
            U1_next[i][j][k] = NAN;
        }
    }
    FOR_ALL_YSTAG (i, j, k) {
        if (flag[i][j][k] == FLAG_FLUID && flag[i][j+1][k] == FLAG_FLUID) {
            U2_next[i][j][k] = U2_star[i][j][k] - dt * (p_prime[i][j+1][k] - p_prime[i][j][k]) / (yc[j+1] - yc[j]);
        }
        else if (flag[i][j][k] == FLAG_SOLID || flag[i][j+1][k] == FLAG_SOLID) {
            U2_next[i][j][k] = NAN;
        }
    }
    FOR_ALL_ZSTAG (i, j, k) {
        if (flag[i][j][k] == FLAG_FLUID && flag[i][j][k+1] == FLAG_FLUID) {
            U3_next[i][j][k] = U3_star[i][j][k] - dt * (p_prime[i][j][k+1] - p_prime[i][j][k]) / (zc[k+1] - zc[k]);
        }
        else if (flag[i][j][k] == FLAG_SOLID || flag[i][j][k+1] == FLAG_SOLID) {
            U3_next[i][j][k] = NAN;
        }
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
    update_outer(solver);

    /* Exchange u1, u2, u3, and p between the adjacent processes. */
    adj_exchange(solver);

    /* Set boundary condition on obstacle surface. */
    update_ghost(solver);
}

static void interp_stag_vel(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    const double *const dx = solver->dx;
    const double *const dy = solver->dy;
    const double *const dz = solver->dz;

    double *const u1 = solver->u1;
    double *const u2 = solver->u2;
    double *const u3 = solver->u3;

    double *const U1 = solver->U1;
    double *const U2 = solver->U2;
    double *const U3 = solver->U3;

    for (int i = 0; i <= Nx; i++) {
        for (int j = -2; j < Ny+2; j++) {
            for (int k = -2; k < Nz+2; k++) {
                xse(U1, i, j, k) = (c3e(u1, i-1, j, k)*c1e(dx, i) + c3e(u1, i, j, k)*c1e(dx, i-1)) / (c1e(dx, i-1) + c1e(dx, i));
            }
        }
    }
    for (int i = -2; i < Nx+2; i++) {
        for (int j = 0; j <= Ny; j++) {
            for (int k = -2; k < Nz+2; k++) {
                yse(U2, i, j, k) = (c3e(u2, i, j-1, k)*c1e(dy, j) + c3e(u2, i, j, k)*c1e(dy, j-1)) / (c1e(dy, j-1) + c1e(dy, j));
            }
        }
    }
    for (int i = -2; i < Nx+2; i++) {
        for (int j = -2; j < Ny+2; j++) {
            for (int k = 0; k <= Nz; k++) {
                zse(U3, i, j, k) = (c3e(u3, i, j, k-1)*c1e(dz, k) + c3e(u3, i, j, k)*c1e(dz, k-1)) / (c1e(dz, k-1) + c1e(dz, k));
            }
        }
    }
}

static void autosave(IBMSolver *solver) {
    char filename[100];

    snprintf(filename, 100, "%s-%05d", solver->autosave_filename, solver->iter);

    IBMSolver_export_result(solver, filename);
}

static void update_outer(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    const double *const xc = solver->xc;
    const double *const yc = solver->yc;
    const double *const zc = solver->zc;

    double *const u1 = solver->u1;
    double *const u2 = solver->u2;
    double *const u3 = solver->u3;
    double *const p = solver->p;

    const double xmin = solver->xmin, xmax = solver->xmax;
    const double ymin = solver->ymin, ymax = solver->ymax;
    const double zmin = solver->zmin, zmax = solver->zmax;

    double a, b;
    int cnt;

    /* Set velocity boundary conditions. */

    /* West. */
    if (solver->ri == 0) {
        switch (solver->bc[3].type) {
        case BC_VELOCITY_COMPONENT:
            a = c1e(solver->dx, -1) / 2;
            b = c1e(solver->dx, 0) / 2;
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(u1, -1, j, k) = (a+b)/b*bc_val_u1(solver, DIR_WEST, xmin, c1e(yc, j), c1e(zc, k)) - a/b*c3e(u1, 0, j, k);
                    c3e(u2, -1, j, k) = (a+b)/b*bc_val_u2(solver, DIR_WEST, xmin, c1e(yc, j), c1e(zc, k)) - a/b*c3e(u2, 0, j, k);
                    c3e(u3, -1, j, k) = (a+b)/b*bc_val_u3(solver, DIR_WEST, xmin, c1e(yc, j), c1e(zc, k)) - a/b*c3e(u3, 0, j, k);
                }
            }
            a = c1e(solver->dx, -2) / 2 + c1e(solver->dx, -1);
            b = c1e(solver->dx, 0) + c1e(solver->dx, 1) / 2;
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(u1, -2, j, k) = (a+b)/b*bc_val_u1(solver, DIR_WEST, xmin, c1e(yc, j), c1e(zc, k)) - a/b*c3e(u1, 1, j, k);
                    c3e(u2, -2, j, k) = (a+b)/b*bc_val_u2(solver, DIR_WEST, xmin, c1e(yc, j), c1e(zc, k)) - a/b*c3e(u2, 1, j, k);
                    c3e(u3, -2, j, k) = (a+b)/b*bc_val_u3(solver, DIR_WEST, xmin, c1e(yc, j), c1e(zc, k)) - a/b*c3e(u3, 1, j, k);
                }
            }
            break;
        case BC_PRESSURE:
            a = c1e(solver->xc, 0) - c1e(solver->xc, -1);
            b = c1e(solver->xc, 1) - c1e(solver->xc, 0);
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(u1, -1, j, k) = (a+b)/b*c3e(u1, 0, j, k) - a/b*c3e(u1, 1, j, k);
                    c3e(u2, -1, j, k) = (a+b)/b*c3e(u2, 0, j, k) - a/b*c3e(u2, 1, j, k);
                    c3e(u3, -1, j, k) = (a+b)/b*c3e(u3, 0, j, k) - a/b*c3e(u3, 1, j, k);
                }
            }
            a = c1e(solver->xc, -1) - c1e(solver->xc, -2);
            b = c1e(solver->xc, 0) - c1e(solver->xc, -1);
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(u1, -2, j, k) = (a+b)/b*c3e(u1, -1, j, k) - a/b*c3e(u1, 0, j, k);
                    c3e(u2, -2, j, k) = (a+b)/b*c3e(u2, -1, j, k) - a/b*c3e(u2, 0, j, k);
                    c3e(u3, -2, j, k) = (a+b)/b*c3e(u3, -1, j, k) - a/b*c3e(u3, 0, j, k);
                }
            }
            break;
        case BC_FREE_SLIP_WALL:
            a = c1e(solver->dx, -1) / 2;
            b = c1e(solver->dx, 0) / 2;
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(u1, -1, j, k) = -a/b*c3e(u1, 0, j, k);
                    c3e(u2, -1, j, k) = c3e(u2, 0, j, k);
                    c3e(u3, -1, j, k) = c3e(u3, 0, j, k);
                }
            }
            a = c1e(solver->dx, -2) / 2 + c1e(solver->dx, -1);
            b = c1e(solver->dx, 0) + c1e(solver->dx, 1) / 2;
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(u1, -2, j, k) = -a/b*c3e(u1, 1, j, k);
                    c3e(u2, -2, j, k) = c3e(u2, 1, j, k);
                    c3e(u3, -2, j, k) = c3e(u3, 1, j, k);
                }
            }
            break;
        case BC_ALL_PERIODIC:
        case BC_VELOCITY_PERIODIC:
            if (solver->Px == 1) {
                for (int i = -2; i <= -1; i++) {
                    for (int j = -2; j < Ny+2; j++) {
                        for (int k = -2; k < Nz+2; k++) {
                            c3e(u1, i, j, k) = c3e(u1, i+Nx, j, k);
                            c3e(u2, i, j, k) = c3e(u2, i+Nx, j, k);
                            c3e(u3, i, j, k) = c3e(u3, i+Nx, j, k);
                        }
                    }
                }
            }
            else {
                cnt = 0;
                for (int i = 0; i <= 1; i++) {
                    for (int j = -2; j < Ny+2; j++) {
                        for (int k = -2; k < Nz+2; k++) {
                            solver->x_exchg[cnt++] = c3e(u1, i, j, k);
                            solver->x_exchg[cnt++] = c3e(u2, i, j, k);
                            solver->x_exchg[cnt++] = c3e(u3, i, j, k);
                        }
                    }
                }
                MPI_Send(solver->x_exchg, 6*(Ny+4)*(Nz+4), MPI_DOUBLE, solver->rank + (solver->Px-1)*solver->Py*solver->Pz, 0, MPI_COMM_WORLD);
                MPI_Recv(solver->x_exchg, 6*(Ny+4)*(Nz+4), MPI_DOUBLE, solver->rank + (solver->Px-1)*solver->Py*solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cnt = 0;
                for (int i = -2; i <= -1; i++) {
                    for (int j = -2; j < Ny+2; j++) {
                        for (int k = -2; k < Nz+2; k++) {
                            c3e(u1, i, j, k) = solver->x_exchg[cnt++];
                            c3e(u2, i, j, k) = solver->x_exchg[cnt++];
                            c3e(u3, i, j, k) = solver->x_exchg[cnt++];
                        }
                    }
                }
            }
            break;
        default:;
        }
    }

    /* East. */
    if (solver->ri == solver->Px-1) {
        switch (solver->bc[1].type) {
        case BC_VELOCITY_COMPONENT:
            a = c1e(solver->dx, Nx-1) / 2;
            b = c1e(solver->dx, Nx) / 2;
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(u1, Nx, j, k) = (a+b)/a*bc_val_u1(solver, DIR_EAST, xmax, c1e(yc, j), c1e(zc, k)) - b/a*c3e(u1, Nx-1, j, k);
                    c3e(u2, Nx, j, k) = (a+b)/a*bc_val_u2(solver, DIR_EAST, xmax, c1e(yc, j), c1e(zc, k)) - b/a*c3e(u2, Nx-1, j, k);
                    c3e(u3, Nx, j, k) = (a+b)/a*bc_val_u3(solver, DIR_EAST, xmax, c1e(yc, j), c1e(zc, k)) - b/a*c3e(u3, Nx-1, j, k);
                }
            }
            a = c1e(solver->dx, Nx-2) / 2 + c1e(solver->dx, Nx-1);
            b = c1e(solver->dx, Nx) + c1e(solver->dx, Nx+1);
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(u1, Nx+1, j, k) = (a+b)/a*bc_val_u1(solver, DIR_EAST, xmax, c1e(yc, j), c1e(zc, k)) - b/a*c3e(u1, Nx-2, j, k);
                    c3e(u2, Nx+1, j, k) = (a+b)/a*bc_val_u2(solver, DIR_EAST, xmax, c1e(yc, j), c1e(zc, k)) - b/a*c3e(u2, Nx-2, j, k);
                    c3e(u3, Nx+1, j, k) = (a+b)/a*bc_val_u3(solver, DIR_EAST, xmax, c1e(yc, j), c1e(zc, k)) - b/a*c3e(u3, Nx-2, j, k);
                }
            }
            break;
        case BC_PRESSURE:
            a = c1e(solver->xc, Nx-1) - c1e(solver->xc, Nx-2);
            b = c1e(solver->xc, Nx) - c1e(solver->xc, Nx-1);
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(u1, Nx, j, k) = (a+b)/a*c3e(u1, Nx-1, j, k) - b/a*c3e(u1, Nx-2, j, k);
                    c3e(u2, Nx, j, k) = (a+b)/a*c3e(u2, Nx-1, j, k) - b/a*c3e(u2, Nx-2, j, k);
                    c3e(u3, Nx, j, k) = (a+b)/a*c3e(u3, Nx-1, j, k) - b/a*c3e(u3, Nx-2, j, k);
                }
            }
            a = c1e(solver->xc, Nx) - c1e(solver->xc, Nx-1);
            b = c1e(solver->xc, Nx+1) - c1e(solver->xc, Nx);
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(u1, Nx+1, j, k) = (a+b)/a*c3e(u1, Nx, j, k) - b/a*c3e(u1, Nx-1, j, k);
                    c3e(u2, Nx+1, j, k) = (a+b)/a*c3e(u2, Nx, j, k) - b/a*c3e(u2, Nx-1, j, k);
                    c3e(u3, Nx+1, j, k) = (a+b)/a*c3e(u3, Nx, j, k) - b/a*c3e(u3, Nx-1, j, k);
                }
            }
            break;
        case BC_FREE_SLIP_WALL:
            a = c1e(solver->dx, Nx-1) / 2;
            b = c1e(solver->dx, Nx) / 2;
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(u1, Nx, j, k) = -b/a*c3e(u1, Nx-1, j, k);
                    c3e(u2, Nx, j, k) = c3e(u2, Nx-1, j, k);
                    c3e(u3, Nx, j, k) = c3e(u3, Nx-1, j, k);
                }
            }
            a = c1e(solver->dx, Nx-2) / 2 + c1e(solver->dx, Nx-1);
            b = c1e(solver->dx, Nx) + c1e(solver->dx, Nx+1) / 2;
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(u1, Nx+1, j, k) = -b/a*c3e(u1, Nx-2, j, k);
                    c3e(u2, Nx+1, j, k) = c3e(u2, Nx-2, j, k);
                    c3e(u3, Nx+1, j, k) = c3e(u3, Nx-2, j, k);
                }
            }
            break;
        case BC_ALL_PERIODIC:
        case BC_VELOCITY_PERIODIC:
            if (solver->Px == 1) {
                for (int i = Nx; i <= Nx+1; i++) {
                    for (int j = -2; j < Ny+2; j++) {
                        for (int k = -2; k < Nz+2; k++) {
                            c3e(u1, i, j, k) = c3e(u1, i-Nx, j, k);
                            c3e(u2, i, j, k) = c3e(u2, i-Nx, j, k);
                            c3e(u3, i, j, k) = c3e(u3, i-Nx, j, k);
                        }
                    }
                }
            }
            else {
                MPI_Recv(solver->x_exchg, 6*(Ny+4)*(Nz+4), MPI_DOUBLE, solver->rank - (solver->Px-1)*solver->Py*solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cnt = 0;
                for (int i = Nx; i <= Nx+1; i++) {
                    for (int j = -2; j < Ny+2; j++) {
                        for (int k = -2; k < Nz+2; k++) {
                            c3e(u1, i, j, k) = solver->x_exchg[cnt++];
                            c3e(u2, i, j, k) = solver->x_exchg[cnt++];
                            c3e(u3, i, j, k) = solver->x_exchg[cnt++];
                        }
                    }
                }
                cnt = 0;
                for (int i = Nx-2; i <= Nx-1; i++) {
                    for (int j = -2; j < Ny+2; j++) {
                        for (int k = -2; k < Nz+2; k++) {
                            solver->x_exchg[cnt++] = c3e(u1, i, j, k);
                            solver->x_exchg[cnt++] = c3e(u2, i, j, k);
                            solver->x_exchg[cnt++] = c3e(u3, i, j, k);
                        }
                    }
                }
                MPI_Send(solver->x_exchg, 6*(Ny+4)*(Nz+4), MPI_DOUBLE, solver->rank - (solver->Px-1)*solver->Py*solver->Pz, 0, MPI_COMM_WORLD);
            }
            break;
        default:;
        }
    }

    /* South. */
    if (solver->rj == 0) {
        switch (solver->bc[2].type) {
        case BC_VELOCITY_COMPONENT:
            a = c1e(solver->dy, -1) / 2;
            b = c1e(solver->dy, 0) / 2;
            for (int i = -2; i < Nx+2; i++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(u1, i, -1, k) = (a+b)/b*bc_val_u1(solver, DIR_SOUTH, c1e(xc, i), ymin, c1e(zc, k)) - a/b*c3e(u1, i, 0, k);
                    c3e(u2, i, -1, k) = (a+b)/b*bc_val_u2(solver, DIR_SOUTH, c1e(xc, i), ymin, c1e(zc, k)) - a/b*c3e(u2, i, 0, k);
                    c3e(u3, i, -1, k) = (a+b)/b*bc_val_u3(solver, DIR_SOUTH, c1e(xc, i), ymin, c1e(zc, k)) - a/b*c3e(u3, i, 0, k);
                }
            }
            a = c1e(solver->dy, -2) / 2 + c1e(solver->dy, -1);
            b = c1e(solver->dy, 0) + c1e(solver->dy, 1) / 2;
            for (int i = -2; i < Nx+2; i++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(u1, i, -2, k) = (a+b)/b*bc_val_u1(solver, DIR_SOUTH, c1e(xc, i), ymin, c1e(zc, k)) - a/b*c3e(u1, i, 1, k);
                    c3e(u2, i, -2, k) = (a+b)/b*bc_val_u2(solver, DIR_SOUTH, c1e(xc, i), ymin, c1e(zc, k)) - a/b*c3e(u2, i, 1, k);
                    c3e(u3, i, -2, k) = (a+b)/b*bc_val_u3(solver, DIR_SOUTH, c1e(xc, i), ymin, c1e(zc, k)) - a/b*c3e(u3, i, 1, k);
                }
            }
            break;
        case BC_PRESSURE:
            a = c1e(solver->yc, 0) - c1e(solver->yc, -1);
            b = c1e(solver->yc, 1) - c1e(solver->yc, 0);
            for (int i = -2; i < Nx+2; i++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(u1, i, -1, k) = (a+b)/b*c3e(u1, i, 0, k) - a/b*c3e(u1, i, 1, k);
                    c3e(u2, i, -1, k) = (a+b)/b*c3e(u2, i, 0, k) - a/b*c3e(u2, i, 1, k);
                    c3e(u3, i, -1, k) = (a+b)/b*c3e(u3, i, 0, k) - a/b*c3e(u3, i, 1, k);
                }
            }
            a = c1e(solver->yc, -1) - c1e(solver->yc, -2);
            b = c1e(solver->yc, 0) - c1e(solver->yc, -1);
            for (int i = -2; i < Nx+2; i++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(u1, i, -2, k) = (a+b)/b*c3e(u1, i, -1, k) - a/b*c3e(u1, i, 0, k);
                    c3e(u2, i, -2, k) = (a+b)/b*c3e(u2, i, -1, k) - a/b*c3e(u2, i, 0, k);
                    c3e(u3, i, -2, k) = (a+b)/b*c3e(u3, i, -1, k) - a/b*c3e(u3, i, 0, k);
                }
            }
            break;
        case BC_FREE_SLIP_WALL:
            a = c1e(solver->dy, -1) / 2;
            b = c1e(solver->dy, 0) / 2;
            for (int i = -2; i < Nx+2; i++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(u1, i, -1, k) = -a/b*c3e(u1, i, 0, k);
                    c3e(u2, i, -1, k) = c3e(u2, i, 0, k);
                    c3e(u3, i, -1, k) = c3e(u3, i, 0, k);
                }
            }
            a = c1e(solver->dy, -2) / 2 + c1e(solver->dy, -1);
            b = c1e(solver->dy, 0) + c1e(solver->dy, 1) / 2;
            for (int i = -2; i < Nx+2; i++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(u1, i, -2, k) = -a/b*c3e(u1, i, 1, k);
                    c3e(u2, i, -2, k) = c3e(u2, i, 1, k);
                    c3e(u3, i, -2, k) = c3e(u3, i, 1, k);
                }
            }
            break;
        case BC_ALL_PERIODIC:
        case BC_VELOCITY_PERIODIC:
            if (solver->Py == 1) {
                for (int i = -2; i < Nx+2; i++) {
                    for (int j = -2; j <= -1; j++) {
                        for (int k = -2; k < Nz+2; k++) {
                            c3e(u1, i, j, k) = c3e(u1, i, j+Ny, k);
                            c3e(u2, i, j, k) = c3e(u2, i, j+Ny, k);
                            c3e(u3, i, j, k) = c3e(u3, i, j+Ny, k);
                        }
                    }
                }
            }
            else {
                cnt = 0;
                for (int i = -2; i < Nx+2; i++) {
                    for (int j = 0; j <= 1; j++) {
                        for (int k = -2; k < Nz+2; k++) {
                            solver->y_exchg[cnt++] = c3e(u1, i, j, k);
                            solver->y_exchg[cnt++] = c3e(u2, i, j, k);
                            solver->y_exchg[cnt++] = c3e(u3, i, j, k);
                        }
                    }
                }
                MPI_Send(solver->y_exchg, 6*(Nx+4)*(Nz+4), MPI_DOUBLE, solver->rank + (solver->Py-1)*solver->Pz, 0, MPI_COMM_WORLD);
                MPI_Recv(solver->y_exchg, 6*(Nx+4)*(Nz+4), MPI_DOUBLE, solver->rank + (solver->Py-1)*solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cnt = 0;
                for (int i = -2; i < Nx+2; i++) {
                    for (int j = -2; j <= -1; j++) {
                        for (int k = -2; k < Nz+2; k++) {
                            c3e(u1, i, j, k) = solver->y_exchg[cnt++];
                            c3e(u2, i, j, k) = solver->y_exchg[cnt++];
                            c3e(u3, i, j, k) = solver->y_exchg[cnt++];
                        }
                    }
                }
            }
            break;
        default:;
        }
    }

    /* North. */
    if (solver->rj == solver->Py-1) {
        switch (solver->bc[0].type) {
        case BC_VELOCITY_COMPONENT:
            a = c1e(solver->dy, Ny-1) / 2;
            b = c1e(solver->dy, Ny) / 2;
            for (int i = -2; i < Nx+2; i++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(u1, i, Ny, k) = (a+b)/a*bc_val_u1(solver, DIR_NORTH, c1e(xc, i), ymax, c1e(zc, k)) - b/a*c3e(u1, i, Ny-1, k);
                    c3e(u2, i, Ny, k) = (a+b)/a*bc_val_u2(solver, DIR_NORTH, c1e(xc, i), ymax, c1e(zc, k)) - b/a*c3e(u2, i, Ny-1, k);
                    c3e(u3, i, Ny, k) = (a+b)/a*bc_val_u3(solver, DIR_NORTH, c1e(xc, i), ymax, c1e(zc, k)) - b/a*c3e(u3, i, Ny-1, k);
                }
            }
            a = c1e(solver->dy, Ny-2) / 2 + c1e(solver->dy, Ny-1);
            b = c1e(solver->dy, Ny) + c1e(solver->dy, Ny+1);
            for (int i = -2; i < Nx+2; i++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(u1, i, Ny+1, k) = (a+b)/a*bc_val_u1(solver, DIR_NORTH, c1e(xc, i), ymax, c1e(zc, k)) - b/a*c3e(u1, i, Ny-2, k);
                    c3e(u2, i, Ny+1, k) = (a+b)/a*bc_val_u2(solver, DIR_NORTH, c1e(xc, i), ymax, c1e(zc, k)) - b/a*c3e(u2, i, Ny-2, k);
                    c3e(u3, i, Ny+1, k) = (a+b)/a*bc_val_u3(solver, DIR_NORTH, c1e(xc, i), ymax, c1e(zc, k)) - b/a*c3e(u3, i, Ny-2, k);
                }
            }
            break;
        case BC_PRESSURE:
            a = c1e(solver->yc, Ny-1) - c1e(solver->yc, Ny-2);
            b = c1e(solver->yc, Ny) - c1e(solver->yc, Ny-1);
            for (int i = -2; i < Nx+2; i++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(u1, i, Ny, k) = (a+b)/a*c3e(u1, i, Ny-1, k) - b/a*c3e(u1, i, Ny-2, k);
                    c3e(u2, i, Ny, k) = (a+b)/a*c3e(u2, i, Ny-1, k) - b/a*c3e(u2, i, Ny-2, k);
                    c3e(u3, i, Ny, k) = (a+b)/a*c3e(u3, i, Ny-1, k) - b/a*c3e(u3, i, Ny-2, k);
                }
            }
            a = c1e(solver->yc, Ny) - c1e(solver->yc, Ny-1);
            b = c1e(solver->yc, Ny+1) - c1e(solver->yc, Ny);
            for (int i = -2; i < Nx+2; i++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(u1, i, Ny+1, k) = (a+b)/a*c3e(u1, i, Ny, k) - b/a*c3e(u1, i, Ny-1, k);
                    c3e(u2, i, Ny+1, k) = (a+b)/a*c3e(u2, i, Ny, k) - b/a*c3e(u2, i, Ny-1, k);
                    c3e(u3, i, Ny+1, k) = (a+b)/a*c3e(u3, i, Ny, k) - b/a*c3e(u3, i, Ny-1, k);
                }
            }
            break;
        case BC_FREE_SLIP_WALL:
            a = c1e(solver->dy, Ny-1) / 2;
            b = c1e(solver->dy, Ny) / 2;
            for (int i = -2; i < Nx+2; i++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(u1, i, Ny, k) = -b/a*c3e(u1, i, Ny-1, k);
                    c3e(u2, i, Ny, k) = c3e(u2, i, Ny-1, k);
                    c3e(u3, i, Ny, k) = c3e(u3, i, Ny-1, k);
                }
            }
            a = c1e(solver->dy, Ny-2) / 2 + c1e(solver->dy, Ny-1);
            b = c1e(solver->dy, Ny) + c1e(solver->dy, Ny+1) / 2;
            for (int i = -2; i < Nx+2; i++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(u1, i, Ny+1, k) = -b/a*c3e(u1, i, Ny-2, k);
                    c3e(u2, i, Ny+1, k) = c3e(u2, i, Ny-2, k);
                    c3e(u3, i, Ny+1, k) = c3e(u3, i, Ny-2, k);
                }
            }
            break;
        case BC_ALL_PERIODIC:
        case BC_VELOCITY_PERIODIC:
            if (solver->num_process == 1) {
                for (int i = -2; i < Nx+2; i++) {
                    for (int j = Ny; j <= Ny+1; j++) {
                        for (int k = -2; k < Nz+2; k++) {
                            c3e(u1, i, j, k) = c3e(u1, i, j-Ny, k);
                            c3e(u2, i, j, k) = c3e(u2, i, j-Ny, k);
                            c3e(u3, i, j, k) = c3e(u3, i, j-Ny, k);
                        }
                    }
                }
            }
            else {
                MPI_Recv(solver->x_exchg, 6*(Nx+4)*(Nz+4), MPI_DOUBLE, solver->rank - (solver->Py-1)*solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cnt = 0;
                for (int i = -2; i < Nx+2; i++) {
                    for (int j = Ny; j <= Ny+1; j++) {
                        for (int k = -2; k < Nz+2; k++) {
                            c3e(u1, i, j, k) = solver->y_exchg[cnt++];
                            c3e(u2, i, j, k) = solver->y_exchg[cnt++];
                            c3e(u3, i, j, k) = solver->y_exchg[cnt++];
                        }
                    }
                }
                cnt = 0;
                for (int i = -2; i < Nx+2; i++) {
                    for (int j = Ny-2; j <= Ny-1; j++) {
                        for (int k = -2; k < Nz+2; k++) {
                            solver->y_exchg[cnt++] = c3e(u1, i, j, k);
                            solver->y_exchg[cnt++] = c3e(u2, i, j, k);
                            solver->y_exchg[cnt++] = c3e(u3, i, j, k);
                        }
                    }
                }
                MPI_Send(solver->x_exchg, 6*(Nx+4)*(Nz+4), MPI_DOUBLE, solver->rank - (solver->Py-1)*solver->Pz, 0, MPI_COMM_WORLD);
            }
            break;
        default:;
        }
    }

    /* Down. */
    if (solver->rk == 0) {
        switch (solver->bc[4].type) {
        case BC_VELOCITY_COMPONENT:
            a = c1e(solver->dz, -1) / 2;
            b = c1e(solver->dz, 0) / 2;
            for (int i = -2; i < Nx+2; i++) {
                for (int j = -2; j < Ny+2; j++) {
                    c3e(u1, i, j, -1) = (a+b)/b*bc_val_u1(solver, DIR_DOWN, c1e(xc, i), c1e(yc, j), zmin) - a/b*c3e(u1, i, j, 0);
                    c3e(u2, i, j, -1) = (a+b)/b*bc_val_u2(solver, DIR_DOWN, c1e(xc, i), c1e(yc, j), zmin) - a/b*c3e(u2, i, j, 0);
                    c3e(u3, i, j, -1) = (a+b)/b*bc_val_u3(solver, DIR_DOWN, c1e(xc, i), c1e(yc, j), zmin) - a/b*c3e(u3, i, j, 0);
                }
            }
            a = c1e(solver->dz, -2) / 2 + c1e(solver->dz, -1);
            b = c1e(solver->dz, 0) + c1e(solver->dz, 1) / 2;
            for (int i = -2; i < Nx+2; i++) {
                for (int j = -2; j < Ny+2; j++) {
                    c3e(u1, i, j, -2) = (a+b)/b*bc_val_u1(solver, DIR_DOWN, c1e(xc, i), c1e(yc, j), zmin) - a/b*c3e(u1, i, j, 1);
                    c3e(u2, i, j, -2) = (a+b)/b*bc_val_u2(solver, DIR_DOWN, c1e(xc, i), c1e(yc, j), zmin) - a/b*c3e(u2, i, j, 1);
                    c3e(u3, i, j, -2) = (a+b)/b*bc_val_u3(solver, DIR_DOWN, c1e(xc, i), c1e(yc, j), zmin) - a/b*c3e(u3, i, j, 1);
                }
            }
            break;
        case BC_PRESSURE:
            a = c1e(solver->zc, 0) - c1e(solver->zc, -1);
            b = c1e(solver->zc, 1) - c1e(solver->zc, 0);
            for (int i = -2; i < Nx+2; i++) {
                for (int j = -2; j < Ny+2; j++) {
                    c3e(u1, i, j, -1) = (a+b)/b*c3e(u1, i, j, 0) - a/b*c3e(u1, i, j, 1);
                    c3e(u2, i, j, -1) = (a+b)/b*c3e(u2, i, j, 0) - a/b*c3e(u2, i, j, 1);
                    c3e(u3, i, j, -1) = (a+b)/b*c3e(u3, i, j, 0) - a/b*c3e(u3, i, j, 1);
                }
            }
            a = c1e(solver->zc, -1) - c1e(solver->zc, -2);
            b = c1e(solver->zc, 0) - c1e(solver->zc, -1);
            for (int i = -2; i < Nx+2; i++) {
                for (int j = -2; j < Ny+2; j++) {
                    c3e(u1, i, j, -2) = (a+b)/b*c3e(u1, i, j, -1) - a/b*c3e(u1, i, j, 0);
                    c3e(u2, i, j, -2) = (a+b)/b*c3e(u2, i, j, -1) - a/b*c3e(u2, i, j, 0);
                    c3e(u3, i, j, -2) = (a+b)/b*c3e(u3, i, j, -1) - a/b*c3e(u3, i, j, 0);
                }
            }
            break;
        case BC_FREE_SLIP_WALL:
            a = c1e(solver->dz, -1) / 2;
            b = c1e(solver->dz, 0) / 2;
            for (int i = -2; i < Nx+2; i++) {
                for (int j = -2; j < Ny+2; j++) {
                    c3e(u1, i, j, -1) = -a/b*c3e(u1, i, j, 0);
                    c3e(u2, i, j, -1) = c3e(u2, i, j, 0);
                    c3e(u3, i, j, -1) = c3e(u3, i, j, 0);
                }
            }
            a = c1e(solver->dz, -2) / 2 + c1e(solver->dz, -1);
            b = c1e(solver->dz, 0) + c1e(solver->dz, 1) / 2;
            for (int i = -2; i < Nx+2; i++) {
                for (int j = -2; j < Ny+2; j++) {
                    c3e(u1, i, j, -2) = -a/b*c3e(u1, i, j, 1);
                    c3e(u2, i, j, -2) = c3e(u2, i, j, 1);
                    c3e(u3, i, j, -2) = c3e(u3, i, j, 1);
                }
            }
            break;
        case BC_ALL_PERIODIC:
        case BC_VELOCITY_PERIODIC:
            if (solver->Pz == 1) {
                for (int i = -2; i < Nx+2; i++) {
                    for (int j = -2; j < Ny+2; j++) {
                        for (int k = -2; k <= -1; k++) {
                            c3e(u1, i, j, k) = c3e(u1, i, j, k+Nz);
                            c3e(u2, i, j, k) = c3e(u2, i, j, k+Nz);
                            c3e(u3, i, j, k) = c3e(u3, i, j, k+Nz);
                        }
                    }
                }
            }
            else {
                cnt = 0;
                for (int i = -2; i < Nx+2; i++) {
                    for (int j = -2; j < Ny+2; j++) {
                        for (int k = 0; k <= 1; k++) {
                            solver->z_exchg[cnt++] = c3e(u1, i, j, k);
                            solver->z_exchg[cnt++] = c3e(u2, i, j, k);
                            solver->z_exchg[cnt++] = c3e(u3, i, j, k);
                        }
                    }
                }
                MPI_Send(solver->z_exchg, 6*(Nx+4)*(Ny+4), MPI_DOUBLE, solver->rank + (solver->Pz-1), 0, MPI_COMM_WORLD);
                MPI_Recv(solver->z_exchg, 6*(Nx+4)*(Ny+4), MPI_DOUBLE, solver->rank + (solver->Pz-1), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cnt = 0;
                for (int i = -2; i < Nx+2; i++) {
                    for (int j = -2; j < Ny+2; j++) {
                        for (int k = -2; k <= -1; k++) {
                            c3e(u1, i, j, k) = solver->z_exchg[cnt++];
                            c3e(u2, i, j, k) = solver->z_exchg[cnt++];
                            c3e(u3, i, j, k) = solver->z_exchg[cnt++];
                        }
                    }
                }
            }
            break;
        default:;
        }
    }

    /* Up. */
    if (solver->rk == solver->Pz-1) {
        switch (solver->bc[5].type) {
        case BC_VELOCITY_COMPONENT:
            a = c1e(solver->dz, Nz-1) / 2;
            b = c1e(solver->dz, Nz) / 2;
            for (int i = -2; i < Nx+2; i++) {
                for (int j = -2; j < Ny+2; j++) {
                    c3e(u1, i, j, Nz) = (a+b)/a*bc_val_u1(solver, DIR_UP, c1e(xc, i), c1e(yc, j), zmax) - b/a*c3e(u1, i, j, Nz-1);
                    c3e(u2, i, j, Nz) = (a+b)/a*bc_val_u2(solver, DIR_UP, c1e(xc, i), c1e(yc, j), zmax) - b/a*c3e(u2, i, j, Nz-1);
                    c3e(u3, i, j, Nz) = (a+b)/a*bc_val_u3(solver, DIR_UP, c1e(xc, i), c1e(yc, j), zmax) - b/a*c3e(u3, i, j, Nz-1);
                }
            }
            a = c1e(solver->dz, Nz-2) / 2 + c1e(solver->dz, Nz-1);
            b = c1e(solver->dz, Nz) + c1e(solver->dz, Nz+1);
            for (int i = -2; i < Nx+2; i++) {
                for (int j = -2; j < Ny+2; j++) {
                    c3e(u1, i, j, Nz+1) = (a+b)/a*bc_val_u1(solver, DIR_UP, c1e(xc, i), c1e(yc, j), zmax) - b/a*c3e(u1, i, j, Nz-2);
                    c3e(u2, i, j, Nz+1) = (a+b)/a*bc_val_u2(solver, DIR_UP, c1e(xc, i), c1e(yc, j), zmax) - b/a*c3e(u2, i, j, Nz-2);
                    c3e(u3, i, j, Nz+1) = (a+b)/a*bc_val_u3(solver, DIR_UP, c1e(xc, i), c1e(yc, j), zmax) - b/a*c3e(u3, i, j, Nz-2);
                }
            }
            break;
        case BC_PRESSURE:
            a = c1e(solver->zc, Nz-1) - c1e(solver->zc, Nz-2);
            b = c1e(solver->zc, Nz) - c1e(solver->zc, Nz-1);
            for (int i = -2; i < Nx+2; i++) {
                for (int j = -2; j < Ny+2; j++) {
                    c3e(u1, i, j, Nz) = (a+b)/a*c3e(u1, i, j, Nz-1) - b/a*c3e(u1, i, j, Nz-2);
                    c3e(u2, i, j, Nz) = (a+b)/a*c3e(u2, i, j, Nz-1) - b/a*c3e(u2, i, j, Nz-2);
                    c3e(u3, i, j, Nz) = (a+b)/a*c3e(u3, i, j, Nz-1) - b/a*c3e(u3, i, j, Nz-2);
                }
            }
            a = c1e(solver->zc, Nz) - c1e(solver->zc, Nz-1);
            b = c1e(solver->zc, Nz+1) - c1e(solver->zc, Nz);
            for (int i = -2; i < Nx+2; i++) {
                for (int j = -2; j < Ny+2; j++) {
                    c3e(u1, i, j, Nz+1) = (a+b)/a*c3e(u1, i, j, Nz) - b/a*c3e(u1, i, j, Nz-1);
                    c3e(u2, i, j, Nz+1) = (a+b)/a*c3e(u2, i, j, Nz) - b/a*c3e(u2, i, j, Nz-1);
                    c3e(u3, i, j, Nz+1) = (a+b)/a*c3e(u3, i, j, Nz) - b/a*c3e(u3, i, j, Nz-1);
                }
            }
            break;
        case BC_FREE_SLIP_WALL:
            a = c1e(solver->dz, Nz-1) / 2;
            b = c1e(solver->dz, Nz) / 2;
            for (int i = -2; i < Nx+2; i++) {
                for (int j = -2; j < Ny+2; j++) {
                    c3e(u1, i, j, Nz) = -b/a*c3e(u1, i, j, Nz-1);
                    c3e(u2, i, j, Nz) = c3e(u2, i, j, Nz-1);
                    c3e(u3, i, j, Nz) = c3e(u3, i, j, Nz-1);
                }
            }
            a = c1e(solver->dz, Nz-2) / 2 + c1e(solver->dz, Nz-1);
            b = c1e(solver->dz, Nz) + c1e(solver->dz, Nz+1) / 2;
            for (int i = -2; i < Nx+2; i++) {
                for (int j = -2; j < Ny+2; j++) {
                    c3e(u1, i, j, Nz+1) = -b/a*c3e(u1, i, j, Nz-2);
                    c3e(u2, i, j, Nz+1) = c3e(u2, i, j, Nz-2);
                    c3e(u3, i, j, Nz+1) = c3e(u3, i, j, Nz-2);
                }
            }
            break;
        case BC_ALL_PERIODIC:
        case BC_VELOCITY_PERIODIC:
            if (solver->num_process == 1) {
                for (int i = -2; i < Nx+2; i++) {
                    for (int j = -2; j < Ny+2; j++) {
                        for (int k = Nz; k <= Nz+1; k++) {
                            c3e(u1, i, j, k) = c3e(u1, i, j, k-Nz);
                            c3e(u2, i, j, k) = c3e(u2, i, j, k-Nz);
                            c3e(u3, i, j, k) = c3e(u3, i, j, k-Nz);
                        }
                    }
                }
            }
            else {
                MPI_Recv(solver->z_exchg, 6*(Nx+4)*(Ny+4), MPI_DOUBLE, solver->rank - (solver->Pz-1), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cnt = 0;
                for (int i = -2; i < Nx+2; i++) {
                    for (int j = -2; j < Ny+2; j++) {
                        for (int k = Nz; k <= Nz+1; k++) {
                            c3e(u1, i, j, k) = solver->z_exchg[cnt++];
                            c3e(u2, i, j, k) = solver->z_exchg[cnt++];
                            c3e(u3, i, j, k) = solver->z_exchg[cnt++];
                        }
                    }
                }
                cnt = 0;
                for (int i = -2; i < Nx+2; i++) {
                    for (int j = -2; j < Ny+2; j++) {
                        for (int k = Nz-2; k <= Nz-1; k++) {
                            solver->z_exchg[cnt++] = c3e(u1, i, j, k);
                            solver->z_exchg[cnt++] = c3e(u2, i, j, k);
                            solver->z_exchg[cnt++] = c3e(u3, i, j, k);
                        }
                    }
                }
                MPI_Send(solver->x_exchg, 6*(Nx+4)*(Ny+4), MPI_DOUBLE, solver->rank - (solver->Pz-1), 0, MPI_COMM_WORLD);
            }
            break;
        default:;
        }
    }

    /* Set pressure boundary conditions. */

    /* West. */
    if (solver->ri == 0) {
        switch (solver->bc[3].type) {
        case BC_VELOCITY_COMPONENT:
        case BC_STATIONARY_WALL:
        case BC_FREE_SLIP_WALL:
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(p, -1, j, k) = c3e(p, 0, j, k);
                    c3e(p, -2, j, k) = c3e(p, 1, j, k);
                }
            }
            break;
        case BC_PRESSURE:
        case BC_VELOCITY_PERIODIC:
            a = c1e(solver->dx, -1) / 2;
            b = c1e(solver->dx, 0) / 2;
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(p, -1, j, k) = (a+b)/b*bc_val_p(solver, DIR_WEST, xmin, c1e(yc, j), c1e(zc, k)) - a/b*c3e(p, 0, j, k);
                }
            }
            a = c1e(solver->dx, -2) / 2 + c1e(solver->dx, -1);
            b = c1e(solver->dx, 0) + c1e(solver->dx, 1) / 2;
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(p, -2, j, k) = (a+b)/b*bc_val_p(solver, DIR_WEST, xmin, c1e(yc, j), c1e(zc, k)) - a/b*c3e(p, 1, j, k);
                }
            }
            break;
        case BC_ALL_PERIODIC:
            if (solver->Px == 1) {
                for (int i = -2; i <= -1; i++) {
                    for (int j = -2; j < Ny+2; j++) {
                        for (int k = -2; k < Nz+2; k++) {
                            c3e(p, i, j, k) = c3e(p, i+Nx, j, k);
                        }
                    }
                }
            }
            else {
                cnt = 0;
                for (int i = 0; i <= 1; i++) {
                    for (int j = -2; j < Ny+2; j++) {
                        for (int k = -2; k < Nz+2; k++) {
                            solver->x_exchg[cnt++] = c3e(p, i, j, k);
                        }
                    }
                }
                MPI_Send(solver->x_exchg, 2*(Ny+4)*(Nz+4), MPI_DOUBLE, solver->rank + (solver->Px-1)*solver->Py*solver->Pz, 0, MPI_COMM_WORLD);
                MPI_Recv(solver->x_exchg, 2*(Ny+4)*(Nz+4), MPI_DOUBLE, solver->rank + (solver->Px-1)*solver->Py*solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cnt = 0;
                for (int i = -2; i <= -1; i++) {
                    for (int j = -2; j < Ny+2; j++) {
                        for (int k = -2; k < Nz+2; k++) {
                            c3e(p, i, j, k) = solver->x_exchg[cnt++];
                        }
                    }
                }
            }
            break;
        default:;
        }
    }

    /* East. */
    if (solver->ri == solver->Px-1) {
        switch (solver->bc[1].type) {
        case BC_VELOCITY_COMPONENT:
        case BC_STATIONARY_WALL:
        case BC_FREE_SLIP_WALL:
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(p, Nx, j, k) = c3e(p, Nx-1, j, k);
                    c3e(p, Nx+1, j, k) = c3e(p, Nx-2, j, k);
                }
            }
            break;
        case BC_PRESSURE:
        case BC_VELOCITY_PERIODIC:
            a = c1e(solver->dx, Nx-1) / 2;
            b = c1e(solver->dx, Nx) / 2;
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(p, Nx, j, k) = (a+b)/a*bc_val_p(solver, DIR_EAST, xmax, c1e(yc, j), c1e(zc, k)) - b/a*c3e(p, Nx-1, j, k);
                }
            }
            a = c1e(solver->dx, Nx-2) / 2 + c1e(solver->dx, Nx-1);
            b = c1e(solver->dx, Nx) + c1e(solver->dx, Nx+1) / 2;
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(p, Nx+1, j, k) = (a+b)/a*bc_val_p(solver, DIR_EAST, xmax, c1e(yc, j), c1e(zc, k)) - b/a*c3e(p, Nx-2, j, k);
                }
            }
            break;
        case BC_ALL_PERIODIC:
            if (solver->Px == 1) {
                for (int i = Nx; i <= Nx+1; i++) {
                    for (int j = -2; j < Ny+2; j++) {
                        for (int k = -2; k < Nz+2; k++) {
                            c3e(p, i, j, k) = c3e(p, i-Nx, j, k);
                        }
                    }
                }
            }
            else {
                MPI_Recv(solver->x_exchg, 2*(Ny+4)*(Nz+4), MPI_DOUBLE, solver->rank - (solver->Px-1)*solver->Py*solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cnt = 0;
                for (int i = Nx; i <= Nx+1; i++) {
                    for (int j = -2; j < Ny+2; j++) {
                        for (int k = -2; k < Nz+2; k++) {
                            c3e(p, i, j, k) = solver->x_exchg[cnt++];
                        }
                    }
                }
                cnt = 0;
                for (int i = Nx-2; i <= Nx-1; i++) {
                    for (int j = -2; j < Ny+2; j++) {
                        for (int k = -2; k < Nz+2; k++) {
                            solver->x_exchg[cnt++] = c3e(p, i, j, k);
                        }
                    }
                }
                MPI_Send(solver->x_exchg, 2*(Ny+4)*(Nz+4), MPI_DOUBLE, solver->rank - (solver->Px-1)*solver->Py*solver->Pz, 0, MPI_COMM_WORLD);
            }
            break;
        default:;
        }
    }

    /* South. */
    if (solver->rj == 0) {
        switch (solver->bc[2].type) {
        case BC_VELOCITY_COMPONENT:
        case BC_STATIONARY_WALL:
        case BC_FREE_SLIP_WALL:
            for (int i = -2; i < Nx+2; i++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(p, i, -1, k) = c3e(p, i, 0, k);
                    c3e(p, i, -2, k) = c3e(p, i, 1, k);
                }
            }
            break;
        case BC_PRESSURE:
        case BC_VELOCITY_PERIODIC:
            a = c1e(solver->dy, -1) / 2;
            b = c1e(solver->dy, 0) / 2;
            for (int i = -2; i < Nx+2; i++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(p, i, -1, k) = (a+b)/b*bc_val_p(solver, DIR_SOUTH, c1e(xc, i), ymin, c1e(zc, k)) - a/b*c3e(p, i, 0, k);
                }
            }
            a = c1e(solver->dy, -2) / 2 + c1e(solver->dy, -1);
            b = c1e(solver->dy, 0) + c1e(solver->dy, 1) / 2;
            for (int i = -2; i < Nx+2; i++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(p, i, -2, k) = (a+b)/b*bc_val_p(solver, DIR_SOUTH, c1e(xc, i), ymin, c1e(zc, k)) - a/b*c3e(p, i, 1, k);
                }
            }
            break;
        case BC_ALL_PERIODIC:
            if (solver->Py == 1) {
                for (int i = -2; i < Ny+2; i++) {
                    for (int j = -2; j <= -1; j++) {
                        for (int k = -2; k < Nz+2; k++) {
                            c3e(p, i, j, k) = c3e(p, i, j+Ny, k);
                        }
                    }
                }
            }
            else {
                cnt = 0;
                for (int i = 0; i <= 1; i++) {
                    for (int j = -2; j < Ny+2; j++) {
                        for (int k = -2; k < Nz+2; k++) {
                            solver->x_exchg[cnt++] = c3e(p, i, j, k);
                        }
                    }
                }
                MPI_Send(solver->y_exchg, 2*(Nx+4)*(Nz+4), MPI_DOUBLE, solver->rank + (solver->Py-1)*solver->Pz, 0, MPI_COMM_WORLD);
                MPI_Recv(solver->y_exchg, 2*(Nx+4)*(Nz+4), MPI_DOUBLE, solver->rank + (solver->Py-1)*solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cnt = 0;
                for (int i = -2; i <= -1; i++) {
                    for (int j = -2; j < Ny+2; j++) {
                        for (int k = -2; k < Nz+2; k++) {
                            c3e(p, i, j, k) = solver->y_exchg[cnt++];
                        }
                    }
                }
            }
            break;
        default:;
        }
    }

    /* North. */
    if (solver->rj == solver->Py-1) {
        switch (solver->bc[0].type) {
        case BC_VELOCITY_COMPONENT:
        case BC_STATIONARY_WALL:
        case BC_FREE_SLIP_WALL:
            for (int i = -2; i < Nx+2; i++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(p, i, Ny, k) = c3e(p, i, Ny-1, k);
                    c3e(p, i, Ny+1, k) = c3e(p, i, Ny-2, k);
                }
            }
            break;
        case BC_PRESSURE:
        case BC_VELOCITY_PERIODIC:
            a = c1e(solver->dy, Ny-1) / 2;
            b = c1e(solver->dy, Ny) / 2;
            for (int i = -2; i < Nx+2; i++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(p, i, Ny, k) = (a+b)/a*bc_val_p(solver, DIR_NORTH, c1e(xc, i), ymax, c1e(zc, k)) - b/a*c3e(p, i, Ny-1, k);
                }
            }
            a = c1e(solver->dy, Ny-2) / 2 + c1e(solver->dy, Ny-1);
            b = c1e(solver->dy, Ny) + c1e(solver->dy, Ny+1) / 2;
            for (int i = -2; i < Nx+2; i++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(p, i, Ny+1, k) = (a+b)/a*bc_val_p(solver, DIR_NORTH, c1e(xc, i), ymax, c1e(zc, k)) - b/a*c3e(p, i, Ny-2, k);
                }
            }
            break;
        case BC_ALL_PERIODIC:
            if (solver->Py == 1) {
                for (int i = -2; i < Nx+2; i++) {
                    for (int j = Ny; j <= Ny+1; j++) {
                        for (int k = -2; k < Nz+2; k++) {
                            c3e(p, i, j, k) = c3e(p, i, j-Ny, k);
                        }
                    }
                }
            }
            else {
                MPI_Recv(solver->y_exchg, 2*(Nx+4)*(Nz+4), MPI_DOUBLE, solver->rank - (solver->Py-1)*solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cnt = 0;
                for (int i = -2; i < Nx+2; i++) {
                    for (int j = Ny; j <= Ny+1; j++) {
                        for (int k = -2; k < Nz+2; k++) {
                            c3e(p, i, j, k) = solver->y_exchg[cnt++];
                        }
                    }
                }
                cnt = 0;
                for (int i = -2; i < Nx+2; i++) {
                    for (int j = Ny-2; j <= Ny-1; j++) {
                        for (int k = -2; k < Nz+2; k++) {
                            solver->y_exchg[cnt++] = c3e(p, i, j, k);
                        }
                    }
                }
                MPI_Send(solver->y_exchg, 2*(Nx+4)*(Nz+4), MPI_DOUBLE, solver->rank - (solver->Py-1)*solver->Pz, 0, MPI_COMM_WORLD);
            }
            break;
        default:;
        }
    }

    /* Down. */
    if (solver->rk == 0) {
        switch (solver->bc[4].type) {
        case BC_VELOCITY_COMPONENT:
        case BC_STATIONARY_WALL:
        case BC_FREE_SLIP_WALL:
            for (int i = -2; i < Nx+2; i++) {
                for (int j = -2; j < Ny+2; j++) {
                    c3e(p, i, j, -1) = c3e(p, i, j, 0);
                    c3e(p, i, j, -2) = c3e(p, i, j, 1);
                }
            }
            break;
        case BC_PRESSURE:
        case BC_VELOCITY_PERIODIC:
            a = c1e(solver->dx, -1) / 2;
            b = c1e(solver->dx, 0) / 2;
            for (int i = -2; i < Nx+2; i++) {
                for (int j = -2; j < Ny+2; j++) {
                    c3e(p, i, j, -1) = (a+b)/b*bc_val_p(solver, DIR_DOWN, c1e(xc, i), c1e(yc, j), zmin) - a/b*c3e(p, i, j, 0);
                }
            }
            a = c1e(solver->dx, -2) / 2 + c1e(solver->dx, -1);
            b = c1e(solver->dx, 0) + c1e(solver->dx, 1) / 2;
            for (int i = -2; i < Nx+2; i++) {
                for (int j = -2; j < Ny+2; j++) {
                    c3e(p, i, j, -2) = (a+b)/b*bc_val_p(solver, DIR_DOWN, c1e(xc, i), c1e(yc, j), zmin) - a/b*c3e(p, i, j, 1);
                }
            }
            break;
        case BC_ALL_PERIODIC:
            if (solver->Pz == 1) {
                for (int i = -2; i < Nx+2; i++) {
                    for (int j = -2; j < Ny+2; j++) {
                        for (int k = -2; k <= -1; k++) {
                            c3e(p, i, j, k) = c3e(p, i, j, k+Nz);
                        }
                    }
                }
            }
            else {
                cnt = 0;
                for (int i = -2; i < Nx+2; i++) {
                    for (int j = -2; j < Ny+2; j++) {
                        for (int k = 0; k <= 1; k++) {
                            solver->z_exchg[cnt++] = c3e(p, i, j, k);
                        }
                    }
                }
                MPI_Send(solver->z_exchg, 2*(Nx+4)*(Ny+4), MPI_DOUBLE, solver->rank + (solver->Pz-1), 0, MPI_COMM_WORLD);
                MPI_Recv(solver->z_exchg, 2*(Nx+4)*(Ny+4), MPI_DOUBLE, solver->rank + (solver->Pz-1), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cnt = 0;
                for (int i = -2; i < Nx+2; i++) {
                    for (int j = -2; j < Ny+2; j++) {
                        for (int k = -2; k <= -1; k++) {
                            c3e(p, i, j, k) = solver->z_exchg[cnt++];
                        }
                    }
                }
            }
            break;
        default:;
        }
    }

    /* Up. */
    if (solver->rk == solver->Pz-1) {
        switch (solver->bc[5].type) {
        case BC_VELOCITY_COMPONENT:
        case BC_STATIONARY_WALL:
        case BC_FREE_SLIP_WALL:
            for (int i = -2; i < Nx+2; i++) {
                for (int j = -2; j < Ny+2; j++) {
                    c3e(p, i, j, Nz) = c3e(p, i, j, Nz-1);
                    c3e(p, i, j, Nz+1) = c3e(p, i, j, Nz-2);
                }
            }
            break;
        case BC_PRESSURE:
        case BC_VELOCITY_PERIODIC:
            a = c1e(solver->dz, Nz-1) / 2;
            b = c1e(solver->dz, Nz) / 2;
            for (int i = -2; i < Nx+2; i++) {
                for (int j = -2; j < Ny+2; j++) {
                    c3e(p, i, j, Nz) = (a+b)/a*bc_val_p(solver, DIR_UP, c1e(xc, i), c1e(yc, j), zmax) - b/a*c3e(p, i, j, Nz-1);
                }
            }
            a = c1e(solver->dz, Nz-2) / 2 + c1e(solver->dz, Nz-1);
            b = c1e(solver->dz, Nz) + c1e(solver->dz, Nz+1) / 2;
            for (int i = -2; i < Nx+2; i++) {
                for (int j = -2; j < Ny+2; j++) {
                    c3e(p, i, j, Nz+1) = (a+b)/a*bc_val_p(solver, DIR_UP, c1e(xc, i), c1e(yc, j), zmax) - b/a*c3e(p, i, j, Nz-2);
                }
            }
            break;
        case BC_ALL_PERIODIC:
            if (solver->Pz == 1) {
                for (int i = -2; i < Nx+2; i++) {
                    for (int j = -2; j < Ny+2; j++) {
                        for (int k = Nz; k <= Nz+1; k++) {
                                c3e(p, i, j, k) = c3e(p, i, j, k-Nz);
                            }
                        }
                    }
                }
            else {
                MPI_Recv(solver->z_exchg, 2*(Nx+4)*(Ny+4), MPI_DOUBLE, solver->rank - (solver->Pz-1), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cnt = 0;
                for (int i = -2; i < Nx+2; i++) {
                    for (int j = -2; j < Ny+2; j++) {
                        for (int k = Nz; k <= Nz+1; k++) {
                            c3e(p, i, j, k) = solver->z_exchg[cnt++];
                        }
                    }
                }
                cnt = 0;
                for (int i = -2; i < Nx+2; i++) {
                    for (int j = -2; j < Ny+2; j++) {
                        for (int k = Nz-2; k <= Nz-1; k++) {
                            solver->z_exchg[cnt++] = c3e(p, i, j, k);
                        }
                    }
                }
                MPI_Send(solver->x_exchg, 2*(Nx+4)*(Ny+4), MPI_DOUBLE, solver->rank - (solver->Pz-1), 0, MPI_COMM_WORLD);
            }
            break;
        default:;
        }
    }
}

static void adj_exchange(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    int cnt = 0;

    /* X. */
    if (solver->ri != solver->Px-1) {
        /* Send to next process. */
        cnt = 0;
        for (int i = Nx-2; i <= Nx-1; i++) {
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    solver->x_exchg[cnt++] = c3e(solver->u1, i, j, k);
                    solver->x_exchg[cnt++] = c3e(solver->u2, i, j, k);
                    solver->x_exchg[cnt++] = c3e(solver->u3, i, j, k);
                    solver->x_exchg[cnt++] = c3e(solver->p, i, j, k);
                }
            }
        }
        MPI_Send(solver->x_exchg, 8*(Ny+4)*(Nz+4), MPI_DOUBLE, solver->rank + solver->Py*solver->Pz, 0, MPI_COMM_WORLD);
    }
    if (solver->ri != 0) {
        /* Receive from previous process. */
        MPI_Recv(solver->x_exchg, 8*(Ny+4)*(Nz+4), MPI_DOUBLE, solver->rank - solver->Py*solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        for (int i = -2; i <= -1; i++) {
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(solver->u1, i, j, k) = solver->x_exchg[cnt++];
                    c3e(solver->u2, i, j, k) = solver->x_exchg[cnt++];
                    c3e(solver->u3, i, j, k) = solver->x_exchg[cnt++];
                    c3e(solver->p, i, j, k) = solver->x_exchg[cnt++];
                }
            }
        }
        /* Send to previous process. */
        cnt = 0;
        for (int i = 0; i <= 1; i++) {
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    solver->x_exchg[cnt++] = c3e(solver->u1, i, j, k);
                    solver->x_exchg[cnt++] = c3e(solver->u2, i, j, k);
                    solver->x_exchg[cnt++] = c3e(solver->u3, i, j, k);
                    solver->x_exchg[cnt++] = c3e(solver->p, i, j, k);
                }
            }
        }
        MPI_Send(solver->x_exchg, 8*(Ny+4)*(Nz+4), MPI_DOUBLE, solver->rank - solver->Py*solver->Pz, 0, MPI_COMM_WORLD);
    }
    if (solver->ri != solver->Px-1) {
        /* Receive from next process. */
        MPI_Recv(solver->x_exchg, 8*(Ny+4)*(Nz+4), MPI_DOUBLE, solver->rank + solver->Py*solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        for (int i = Nx; i <= Nx+1; i++) {
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(solver->u1, i, j, k) = solver->x_exchg[cnt++];
                    c3e(solver->u2, i, j, k) = solver->x_exchg[cnt++];
                    c3e(solver->u3, i, j, k) = solver->x_exchg[cnt++];
                    c3e(solver->p, i, j, k) = solver->x_exchg[cnt++];
                }
            }
        }
    }

    /* Y. */
    if (solver->rj != solver->Py-1) {
        /* Send to next process. */
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int j = Ny-2; j <= Ny-1; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    solver->y_exchg[cnt++] = c3e(solver->u1, i, j, k);
                    solver->y_exchg[cnt++] = c3e(solver->u2, i, j, k);
                    solver->y_exchg[cnt++] = c3e(solver->u3, i, j, k);
                    solver->y_exchg[cnt++] = c3e(solver->p, i, j, k);
                }
            }
        }
        MPI_Send(solver->y_exchg, 8*(Nx+4)*(Nz+4), MPI_DOUBLE, solver->rank + solver->Pz, 0, MPI_COMM_WORLD);
    }
    if (solver->rj != 0) {
        /* Receive from previous process. */
        MPI_Recv(solver->y_exchg, 8*(Nx+4)*(Nz+4), MPI_DOUBLE, solver->rank - solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int j = -2; j <= -1; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(solver->u1, i, j, k) = solver->y_exchg[cnt++];
                    c3e(solver->u2, i, j, k) = solver->y_exchg[cnt++];
                    c3e(solver->u3, i, j, k) = solver->y_exchg[cnt++];
                    c3e(solver->p, i, j, k) = solver->y_exchg[cnt++];
                }
            }
        }
        /* Send to previous process. */
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int j = 0; j <= 1; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    solver->y_exchg[cnt++] = c3e(solver->u1, i, j, k);
                    solver->y_exchg[cnt++] = c3e(solver->u2, i, j, k);
                    solver->y_exchg[cnt++] = c3e(solver->u3, i, j, k);
                    solver->y_exchg[cnt++] = c3e(solver->p, i, j, k);
                }
            }
        }
        MPI_Send(solver->y_exchg, 8*(Nx+4)*(Nz+4), MPI_DOUBLE, solver->rank - solver->Pz, 0, MPI_COMM_WORLD);
    }
    if (solver->rj != solver->Py-1) {
        /* Receive from next process. */
        MPI_Recv(solver->y_exchg, 8*(Nx+4)*(Nz+4), MPI_DOUBLE, solver->rank + solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int j = Ny; j <= Ny+1; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(solver->u1, i, j, k) = solver->y_exchg[cnt++];
                    c3e(solver->u2, i, j, k) = solver->y_exchg[cnt++];
                    c3e(solver->u3, i, j, k) = solver->y_exchg[cnt++];
                    c3e(solver->p, i, j, k) = solver->y_exchg[cnt++];
                }
            }
        }
    }

    /* Z. */
    if (solver->rk != solver->Pz-1) {
        /* Send to next process. */
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int j = -2; j < Ny+2; j++) {
                for (int k = Nz-2; k <= Nz-1; k++) {
                    solver->z_exchg[cnt++] = c3e(solver->u1, i, j, k);
                    solver->z_exchg[cnt++] = c3e(solver->u2, i, j, k);
                    solver->z_exchg[cnt++] = c3e(solver->u3, i, j, k);
                    solver->z_exchg[cnt++] = c3e(solver->p, i, j, k);
                }
            }
        }
        MPI_Send(solver->z_exchg, 8*(Nx+4)*(Ny+4), MPI_DOUBLE, solver->rank + 1, 0, MPI_COMM_WORLD);
    }
    if (solver->rk != 0) {
        /* Receive from previous process. */
        MPI_Recv(solver->z_exchg, 8*(Nx+4)*(Ny+4), MPI_DOUBLE, solver->rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k <= -1; k++) {
                    c3e(solver->u1, i, j, k) = solver->z_exchg[cnt++];
                    c3e(solver->u2, i, j, k) = solver->z_exchg[cnt++];
                    c3e(solver->u3, i, j, k) = solver->z_exchg[cnt++];
                    c3e(solver->p, i, j, k) = solver->z_exchg[cnt++];
                }
            }
        }
        /* Send to previous process. */
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int j = -2; j < Ny+2; j++) {
                for (int k = 0; k <= 1; k++) {
                    solver->z_exchg[cnt++] = c3e(solver->u1, i, j, k);
                    solver->z_exchg[cnt++] = c3e(solver->u2, i, j, k);
                    solver->z_exchg[cnt++] = c3e(solver->u3, i, j, k);
                    solver->z_exchg[cnt++] = c3e(solver->p, i, j, k);
                }
            }
        }
        MPI_Send(solver->z_exchg, 8*(Nx+4)*(Ny+4), MPI_DOUBLE, solver->rank - 1, 0, MPI_COMM_WORLD);
    }
    if (solver->rk != solver->Pz-1) {
        /* Receive from next process. */
        MPI_Recv(solver->z_exchg, 8*(Nx+4)*(Ny+4), MPI_DOUBLE, solver->rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int j = -2; j < Ny+2; j++) {
                for (int k = Nz; k <= Nz+1; k++) {
                    c3e(solver->u1, i, j, k) = solver->z_exchg[cnt++];
                    c3e(solver->u2, i, j, k) = solver->z_exchg[cnt++];
                    c3e(solver->u3, i, j, k) = solver->z_exchg[cnt++];
                    c3e(solver->p, i, j, k) = solver->z_exchg[cnt++];
                }
            }
        }
    }
}

static void update_ghost(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    const double *const dx = solver->dx;
    const double *const dy = solver->dy;
    const double *const dz = solver->dz;

    const int *const flag = solver->flag;

    double *const u1 = solver->u1;
    double *const u2 = solver->u2;
    double *const u3 = solver->u3;
    double *const p = solver->p;

    double *const U1 = solver->U1;
    double *const U2 = solver->U2;
    double *const U3 = solver->U3;

    int interp_idx[8][3];
    double interp_coeff[8];
    double coeffsum, coeff_lhs_u, coeff_lhs_p;

    double sum_u1, sum_u2, sum_u3, sum_p;

    FOR_INNER_CELL (i, j, k) {
        if (c3e(flag, i, j, k) == FLAG_GHOST) {
            IBMSolver_ghost_interp(solver, i, j, k, interp_idx, interp_coeff);

            coeffsum = 0;
            coeff_lhs_u = coeff_lhs_p = 1;
            sum_u1 = sum_u2 = sum_u3 = sum_p = 0;

            /* If a solid cell is used for interpolation, ignore it. */
            for (int l = 0; l < 8; l++) {
                if (c3e(solver->flag, interp_idx[l][0], interp_idx[l][1], interp_idx[l][2]) == FLAG_SOLID) {
                    interp_coeff[l] = 0;
                }
                else {
                    coeffsum += interp_coeff[l];
                }
            }

            for (int l = 0; l < 8; l++) {
                interp_coeff[l] /= coeffsum;
                if (interp_idx[l][0] == i && interp_idx[l][1] == j && interp_idx[l][2] == k) {
                    coeff_lhs_u += interp_coeff[l];
                    coeff_lhs_p -= interp_coeff[l];
                    interp_coeff[l] = 0;
                }
                sum_u1 += interp_coeff[l] * c3e(u1, interp_idx[l][0], interp_idx[l][1], interp_idx[l][2]);
                sum_u2 += interp_coeff[l] * c3e(u2, interp_idx[l][0], interp_idx[l][1], interp_idx[l][2]);
                sum_u3 += interp_coeff[l] * c3e(u3, interp_idx[l][0], interp_idx[l][1], interp_idx[l][2]);
                sum_p += interp_coeff[l] * c3e(p, interp_idx[l][0], interp_idx[l][1], interp_idx[l][2]);
            }

            c3e(u1, i, j, k) = -sum_u1 / coeff_lhs_u;
            c3e(u2, i, j, k) = -sum_u2 / coeff_lhs_u;
            c3e(u3, i, j, k) = -sum_u3 / coeff_lhs_u;
            c3e(p, i, j, k) = sum_p / coeff_lhs_p;
        }
    }

    FOR_ALL_XSTAG (i, j, k) {
        if (
            (c3e(flag, i-1, j, k) == FLAG_FLUID && c3e(flag, i, j, k) == FLAG_GHOST)
            || (c3e(flag, i-1, j, k) == FLAG_GHOST && c3e(flag, i, j, k) == FLAG_FLUID)
        ) {
            xse(U1, i, j, k) = (c3e(u1, i-1, j, k)*c1e(dx, i) + c3e(u1, i, j, k)*c1e(dx, i-1)) / (c1e(dx, i-1) + c1e(dx, i));
        }
    }
    FOR_ALL_YSTAG (i, j, k) {
        if (
            (c3e(flag, i, j-1, k) == FLAG_FLUID && c3e(flag, i, j, k) == FLAG_GHOST)
            || (c3e(flag, i, j-1, k) == FLAG_GHOST && c3e(flag, i, j, k) == FLAG_FLUID)
        ) {
            yse(U2, i, j, k) = (c3e(u2, i, j-1, k)*c1e(dy, j) + c3e(u2, i, j, k)*c1e(dy, j-1)) / (c1e(dy, j-1) + c1e(dy, j));
        }
    }
    FOR_ALL_ZSTAG (i, j, k) {
        if (
            (c3e(flag, i, j, k-1) == FLAG_FLUID && c3e(flag, i, j, k) == FLAG_GHOST)
            || (c3e(flag, i, j, k-1) == FLAG_GHOST && c3e(flag, i, j, k) == FLAG_FLUID)
        ) {
            zse(U3, i, j, k) = (c3e(u3, i, j, k-1)*c1e(dz, k) + c3e(u3, i, j, k)*c1e(dz, k-1)) / (c1e(dz, k-1) + c1e(dz, k));
        }
    }
}

static double bc_val_u1(
    IBMSolver *solver,
    IBMSolverDirection dir,
    double x, double y, double z
) {
    int idx = dir_to_idx(dir);
    return solver->bc[idx].val_type == BC_CONST
        ? solver->bc[idx].const_u1
        : solver->bc[idx].func_u1(solver->time, x, y, z);
}

static double bc_val_u2(
    IBMSolver *solver,
    IBMSolverDirection dir,
    double x, double y, double z
) {
    int idx = dir_to_idx(dir);
    return solver->bc[idx].val_type == BC_CONST
        ? solver->bc[idx].const_u2
        : solver->bc[idx].func_u2(solver->time, x, y, z);
}

static double bc_val_u3(
    IBMSolver *solver,
    IBMSolverDirection dir,
    double x, double y, double z
) {
    int idx = dir_to_idx(dir);
    return solver->bc[idx].val_type == BC_CONST
        ? solver->bc[idx].const_u3
        : solver->bc[idx].func_u3(solver->time, x, y, z);
}

static double bc_val_p(
    IBMSolver *solver,
    IBMSolverDirection dir,
    double x, double y, double z
) {
    int idx = dir_to_idx(dir);
    return solver->bc[idx].val_type == BC_CONST
        ? solver->bc[idx].const_p
        : solver->bc[idx].func_p(solver->time, x, y, z);
}
