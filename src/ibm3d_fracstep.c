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

static double ext_force(IBMSolver *, int, double, double, double, double);
static double bc_val_u(IBMSolver *, int, IBMSolverDirection, double, double, double, double);
static double bc_val_p(IBMSolver *, IBMSolverDirection, double, double, double, double);

static void exchange_var(IBMSolver *, double *);

/**
 * @brief Iterate \p solver for \p num_time_steps steps using fractional step
 *        method.
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
        calc_N(solver);
        calc_u_star(solver, &final_norm_u1, &final_norm_u2, &final_norm_u3);
        calc_u_tilde(solver);
        calc_U_star(solver);
        calc_p_prime(solver, &final_norm_p);
        update_next(solver);

        i++;
        solver->iter++;
        solver->time = start_time + solver->dt * i;

        update_outer(solver);
        adj_exchange(solver);
        update_ghost(solver);

        /* Print iteration results. */
        if (verbose && solver->rank == 0) {
            clock_gettime(CLOCK_REALTIME, &t_end);
            elapsed_time = (t_end.tv_sec*1000+t_end.tv_nsec/1000000)
                - (t_start.tv_sec*1000+t_start.tv_nsec/1000000);

            if (solver->iter % 10 == 1 || i == 1) {
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

    const double *u[4] = {
        [1] = solver->u1, [2] = solver->u2, [3] = solver->u3
    };

    const double *const U1 = solver->U1;
    const double *const U2 = solver->U2;
    const double *const U3 = solver->U3;

    double *N[4] = {
        [1] = solver->N1, [2] = solver->N2, [3] = solver->N3
    };

    /* Velocities on cell faces. */
    double uw, ue, us, un, ud, uu;

    FOR_INNER_CELL (i, j, k) {
        if (c3e(solver->flag, i, j, k) == FLAG_FLUID) {
            for (int l = 1; l <= 3; l++) {
                uw = (c3e(u[l], i-1, j, k)*c1e(dx, i) + c3e(u[l], i, j, k)*c1e(dx, i-1))
                    / (c1e(dx, i-1) + c1e(dx, i));
                ue = (c3e(u[l], i, j, k)*c1e(dx, i+1) + c3e(u[l], i+1, j, k)*c1e(dx, i))
                    / (c1e(dx, i) + c1e(dx, i+1));
                us = (c3e(u[l], i, j-1, k)*c1e(dy, j) + c3e(u[l], i, j, k)*c1e(dy, j-1))
                    / (c1e(dy, j-1) + c1e(dy, j));
                un = (c3e(u[l], i, j, k)*c1e(dy, j+1) + c3e(u[l], i, j+1, k)*c1e(dy, j))
                    / (c1e(dy, j) + c1e(dy, j+1));
                ud = (c3e(u[l], i, j, k-1)*c1e(dz, k) + c3e(u[l], i, j, k)*c1e(dz, k-1))
                    / (c1e(dz, k-1) + c1e(dz, k));
                uu = (c3e(u[l], i, j, k)*c1e(dz, k+1) + c3e(u[l], i, j, k+1)*c1e(dz, k))
                    / (c1e(dz, k) + c1e(dz, k+1));

                /* Ni = d(U1ui)/dx + d(U2ui)/dy + d(U3ui)/dz */
                c3e(N[l], i, j, k)
                    = (xse(U1, i+1, j, k)*ue - xse(U1, i, j, k)*uw) / c1e(dx, i)
                    + (yse(U2, i, j+1, k)*un - yse(U2, i, j, k)*us) / c1e(dy, j)
                    + (zse(U3, i, j, k+1)*uu - zse(U3, i, j, k)*ud) / c1e(dz, k);
            }
        }
        else {
            for (int l = 1; l <= 3; l++) {
                c3e(N[l], i, j, k) = NAN;
            }
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

    const double *const u[4] = {
        [1] = solver->u1, [2] = solver->u2, [3] = solver->u3
    };
    double *const u_star[4] = {
        [1] = solver->u1_star, [2] = solver->u2_star, [3] = solver->u3_star
    };

    const double *const p = solver->p;

    const double *const N[4] = {
        [1] = solver->N1, [2] = solver->N2, [3] = solver->N3
    };
    const double *const N_prev[4] = {
        [1] = solver->N1_prev, [2] = solver->N2_prev, [3] = solver->N3_prev
    };

    const double xmin = solver->xmin, xmax = solver->xmax;
    const double ymin = solver->ymin, ymax = solver->ymax;
    const double zmin = solver->zmin, zmax = solver->zmax;

    const HYPRE_ParCSRMatrix parcsr_A_u[4] = {
        [1] = solver->parcsr_A_u1, [2] = solver->parcsr_A_u2, [3] = solver->parcsr_A_u3
    };
    double *const final_norm_u[4] = {
        [1] = final_norm_u1, [2] = final_norm_u2, [3] = final_norm_u3
    };

    int ifirst, ilast, jfirst, jlast, kfirst, klast;
    int idx;
    double dpdxl, div_tau;

    int hypre_ierr = 0;

    ifirst = solver->ri == 0 ? -2 : 0;
    ilast = solver->ri != solver->Px-1 ? Nx : Nx+2;
    jfirst = solver->rj == 0 ? -2 : 0;
    jlast = solver->rj != solver->Py-1 ? Ny : Ny+2;
    kfirst = solver->rk == 0 ? -2 : 0;
    klast = solver->rk != solver->Pz-1 ? Nz : Nz+2;

    IBMSolver_calc_tau_r(solver);

    for (int l = 1; l <= 3; l++) {
        memcpy(solver->vector_values, solver->vector_zeros, sizeof(double)*(solver->idx_last-solver->idx_first));

        FOR_INNER_CELL (i, j, k) {
            if (c3e(solver->flag, i, j, k) == FLAG_FLUID) {
                idx = c3e(solver->cell_idx, i, j, k) - solver->idx_first;
                switch (l) {
                case 1:
                    dpdxl = (c3e(p, i+1, j, k) - c3e(p, i-1, j, k)) / (c1e(xc, i+1) - c1e(xc, i-1));
                    break;
                case 2:
                    dpdxl = (c3e(p, i, j+1, k) - c3e(p, i, j-1, k)) / (c1e(yc, j+1) - c1e(yc, j-1));
                    break;
                case 3:
                    dpdxl = (c3e(p, i, j, k+1) - c3e(p, i, j, k-1)) / (c1e(zc, k+1) - c1e(zc, k-1));
                    break;
                }
                div_tau
                    = (c3e(solver->tau_r[l][1], i+1, j, k) - c3e(solver->tau_r[l][1], i-1, j, k))
                        / (c1e(xc, i+1) - c1e(xc, i-1))
                    + (c3e(solver->tau_r[l][2], i, j+1, k) - c3e(solver->tau_r[l][2], i, j-1, k))
                        / (c1e(yc, j+1) - c1e(yc, j-1))
                    + (c3e(solver->tau_r[l][3], i, j, k+1) - c3e(solver->tau_r[l][3], i, j, k-1))
                        / (c1e(zc, k+1) - c1e(zc, k-1));

                solver->vector_values[idx]
                    = -dt/2 * (3*c3e(N[l], i, j, k) - c3e(N_prev[l], i, j, k))
                    - dt * dpdxl
                    + (1-c1e(kx_W, i)-c1e(kx_E, i)-c1e(ky_S, j)-c1e(ky_N, j)-c1e(kz_D, k)-c1e(kz_U, k))*c3e(u[l], i, j, k)
                    + c1e(kx_W, i)*c3e(u[l], i-1, j, k) + c1e(kx_E, i)*c3e(u[l], i+1, j, k)
                    + c1e(ky_S, j)*c3e(u[l], i, j-1, k) + c1e(ky_N, j)*c3e(u[l], i, j+1, k)
                    + c1e(kz_D, k)*c3e(u[l], i, j, k-1) + c1e(kz_U, k)*c3e(u[l], i, j, k+1)
                    + dt * ext_force(solver, l, solver->time, c1e(xc, i), c1e(yc, j), c1e(zc, k))
                    - dt * div_tau;
            }
        }

        if (solver->ri == 0) {
            switch (solver->bc[3].type) {
            case BC_VELOCITY_COMPONENT:
                for (int j = 0; j < Ny; j++) {
                    for (int k = 0; k < Nz; k++) {
                        solver->vector_values[c3e(solver->cell_idx, -1, j, k) - solver->idx_first]
                            = solver->vector_values[c3e(solver->cell_idx, -2, j, k) - solver->idx_first]
                            = bc_val_u(solver, l, DIR_WEST, solver->time, xmin, c1e(yc, j), c1e(zc, k));
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
                            = bc_val_u(solver, l, DIR_EAST, solver->time, xmax, c1e(yc, j), c1e(zc, k));
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
                            = bc_val_u(solver, l, DIR_SOUTH, solver->time, c1e(xc, i), ymin, c1e(zc, k));
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
                            = bc_val_u(solver, l, DIR_NORTH, solver->time, c1e(xc, i), ymax, c1e(zc, k));
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
                            = bc_val_u(solver, l, DIR_DOWN, solver->time, c1e(xc, i), c1e(yc, j), zmin);
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
                            = bc_val_u(solver, l, DIR_UP, solver->time, c1e(xc, i), c1e(yc, j), zmax);
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

        HYPRE_ParCSRBiCGSTABSetup(solver->linear_solver, parcsr_A_u[l], solver->par_b, solver->par_x);
        hypre_ierr = HYPRE_ParCSRBiCGSTABSolve(solver->linear_solver, parcsr_A_u[l], solver->par_b, solver->par_x);
        if (HYPRE_CheckError(hypre_ierr, HYPRE_ERROR_GENERIC)) {
            fprintf(stderr, "error: floating pointer error raised in u%d_star\n", l);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        if (HYPRE_CheckError(hypre_ierr, HYPRE_ERROR_CONV)) {
            fprintf(stderr, "warning: u%d_star did not converge\n", l);
        }

        HYPRE_IJVectorGetValues(solver->x, solver->idx_last-solver->idx_first, solver->vector_rows, solver->vector_res);
        for (int i = ifirst; i < ilast; i++) {
            for (int j = jfirst; j < jlast; j++) {
                for (int k = kfirst; k < klast; k++) {
                    c3e(u_star[l], i, j, k) = solver->vector_res[c3e(solver->cell_idx, i, j, k)-solver->idx_first];
                }
            }
        }
        HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm(solver->linear_solver, final_norm_u[l]);
    }

    /* Exchange u_star between adjacent processes. */
    exchange_var(solver, solver->u1_star);
    exchange_var(solver, solver->u2_star);
    exchange_var(solver, solver->u3_star);
}

static inline void calc_u_tilde(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;
    const double dt = solver->dt;

    const double *const xc = solver->xc;
    const double *const yc = solver->yc;
    const double *const zc = solver->zc;

    const double *const p = solver->p;

    for (int i = -1; i < Nx+1; i++) {
        for (int j = -1; j < Ny+1; j++) {
            for (int k = -1; k < Nz+1; k++) {
                c3e(solver->u1_tilde, i, j, k) = c3e(solver->u1_star, i, j, k)
                    + dt * (c3e(p, i+1, j, k) - c3e(p, i-1, j, k)) / (c1e(xc, i+1) - c1e(xc, i-1));
                c3e(solver->u2_tilde, i, j, k) = c3e(solver->u2_star, i, j, k)
                    + dt * (c3e(p, i, j+1, k) - c3e(p, i, j-1, k)) / (c1e(yc, j+1) - c1e(yc, j-1));
                c3e(solver->u3_tilde, i, j, k) = c3e(solver->u3_star, i, j, k)
                    + dt * (c3e(p, i, j, k+1) - c3e(p, i, j, k-1)) / (c1e(zc, k+1) - c1e(zc, k-1));
            }
        }
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

    const int *const flag = solver->flag;

    const double *const u1_star = solver->u1_star;
    const double *const u2_star = solver->u2_star;
    const double *const u3_star = solver->u3_star;

    const double *const u1_tilde = solver->u1_tilde;
    const double *const u2_tilde = solver->u2_tilde;
    const double *const u3_tilde = solver->u3_tilde;

    double *const U1_star = solver->U1_star;
    double *const U2_star = solver->U2_star;
    double *const U3_star = solver->U3_star;

    const double *const p = solver->p;

    FOR_ALL_XSTAG (i, j, k) {
        xse(U1_star, i, j, k) = (c3e(u1_tilde, i-1, j, k)*c1e(dx, i) + c3e(u1_tilde, i, j, k)*c1e(dx, i-1))
            / (c1e(dx, i-1) + c1e(dx, i))
            - dt * (c3e(p, i, j, k) - c3e(p, i-1, j, k)) / (c1e(xc, i) - c1e(xc, i-1));
    }
    FOR_ALL_YSTAG (i, j, k) {
        yse(U2_star, i, j, k) = (c3e(u2_tilde, i, j-1, k)*c1e(dy, j) + c3e(u2_tilde, i, j, k)*c1e(dy, j-1))
            / (c1e(dy, j-1) + c1e(dy, j))
            - dt * (c3e(p, i, j, k) - c3e(p, i, j-1, k)) / (c1e(yc, j) - c1e(yc, j-1));
    }
    FOR_ALL_ZSTAG (i, j, k) {
        zse(U3_star, i, j, k) = (c3e(u3_tilde, i, j, k-1)*c1e(dz, k) + c3e(u3_tilde, i, j, k)*c1e(dz, k-1))
            / (c1e(dz, k-1) + c1e(dz, k))
            - dt * (c3e(p, i, j, k) - c3e(p, i, j, k-1)) / (c1e(zc, k) - c1e(zc, k-1));
    }

    /* U_star between a fluid cell and a ghost cell. */
    FOR_ALL_XSTAG (i, j, k) {
        if (
            (c3e(flag, i-1, j, k) == FLAG_FLUID && c3e(flag, i, j, k) == FLAG_GHOST)
            || (c3e(flag, i-1, j, k) == FLAG_GHOST && c3e(flag, i, j, k) == FLAG_FLUID)
        ) {
            xse(U1_star, i, j, k) = (c3e(u1_star, i-1, j, k)*c1e(dx, i) + c3e(u1_star, i, j, k)*c1e(dx, i-1))
                / (c1e(dx, i-1) + c1e(dx, i));
        }
    }
    FOR_ALL_YSTAG (i, j, k) {
        if (
            (c3e(flag, i, j-1, k) == FLAG_FLUID && c3e(flag, i, j, k) == FLAG_GHOST)
            || (c3e(flag, i, j-1, k) == FLAG_GHOST && c3e(flag, i, j, k) == FLAG_FLUID)
        ) {
            yse(U2_star, i, j, k) = (c3e(u2_star, i, j-1, k)*c1e(dy, j) + c3e(u2_star, i, j, k)*c1e(dy, j-1))
                / (c1e(dy, j-1) + c1e(dy, j));
        }
    }
    FOR_ALL_ZSTAG (i, j, k) {
        if (
            (c3e(flag, i, j, k-1) == FLAG_FLUID && c3e(flag, i, j, k) == FLAG_GHOST)
            || (c3e(flag, i, j, k-1) == FLAG_GHOST && c3e(flag, i, j, k)== FLAG_FLUID)
        ) {
            zse(U3_star, i, j, k) = (c3e(u3_star, i, j, k-1)*c1e(dz, k) + c3e(u3_star, i, j, k)*c1e(dz, k-1))
                / (c1e(dz, k-1) + c1e(dz, k));
        }
    }
}

static inline void calc_p_prime(IBMSolver *solver, double *final_norm_p) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;
    const double Re = solver->Re;

    const double *const dx = solver->dx;
    const double *const dy = solver->dy;
    const double *const dz = solver->dz;

    const double *const xc = solver->xc;
    const double *const yc = solver->yc;
    const double *const zc = solver->zc;

    const int *const flag = solver->flag;

    const double *const U1_star = solver->U1_star;
    const double *const U2_star = solver->U2_star;
    const double *const U3_star = solver->U3_star;

    double *const p_prime = solver->p_prime;
    double *const p_coeffsum = solver->p_coeffsum;

    const double xmin = solver->xmin, xmax = solver->xmax;
    const double ymin = solver->ymin, ymax = solver->ymax;
    const double zmin = solver->zmin, zmax = solver->zmax;

    int ifirst, ilast, jfirst, jlast, kfirst, klast;
    int idx;

    bool is_p_singular;

    int hypre_ierr = 0;

    ifirst = solver->ri == 0 ? -2 : 0;
    ilast = solver->ri != solver->Px-1 ? Nx : Nx+2;
    jfirst = solver->rj == 0 ? -2 : 0;
    jlast = solver->rj != solver->Py-1 ? Ny : Ny+2;
    kfirst = solver->rk == 0 ? -2 : 0;
    klast = solver->rk != solver->Pz-1 ? Nz : Nz+2;

    is_p_singular = true;
    for (int i = 0; i < 6; i++) {
        if (solver->bc[i].type == BC_PRESSURE || solver->bc[i].type == BC_VELOCITY_PERIODIC) {
            is_p_singular = false;
        }
    }

    memcpy(solver->vector_values, solver->vector_zeros, sizeof(double)*(solver->idx_last-solver->idx_first));

    FOR_INNER_CELL (i, j, k) {
        idx = c3e(solver->cell_idx, i, j, k) - solver->idx_first;
        if (solver->rank == 0 && is_p_singular && i == 0 && j == 0 && k == 0) {
            solver->vector_values[idx] = 0;
        }
        else if (c3e(flag, i, j, k) == FLAG_FLUID) {
            solver->vector_values[idx] = -1/(2*Re) * (
                    (xse(U1_star, i+1, j, k) - xse(U1_star, i, j, k)) / c1e(dx, i)
                    + (yse(U2_star, i, j+1, k) - yse(U2_star, i, j, k)) / c1e(dy, j)
                    + (zse(U3_star, i, j, k+1) - zse(U3_star, i, j, k)) / c1e(dz, k)
                );
            solver->vector_values[idx] /= c3e(p_coeffsum, i, j, k);
        }
    }

    if (solver->ri == 0) {
        switch (solver->bc[3].type) {
        case BC_PRESSURE:
        case BC_VELOCITY_PERIODIC:
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    solver->vector_values[c3e(solver->cell_idx, -1, j, k) - solver->idx_first]
                        = solver->vector_values[c3e(solver->cell_idx, -2, j, k) - solver->idx_first]
                        = bc_val_p(solver, DIR_WEST, solver->time + solver->dt, xmin, c1e(yc, j), c1e(zc, k))
                        - bc_val_p(solver, DIR_WEST, solver->time, xmin, c1e(yc, j), c1e(zc, k));
                }
            }
            break;
        default:;
        }
    }
    if (solver->ri == solver->Px-1) {
        switch (solver->bc[1].type) {
        case BC_PRESSURE:
        case BC_VELOCITY_PERIODIC:
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    solver->vector_values[c3e(solver->cell_idx, Nx, j, k) - solver->idx_first]
                        = solver->vector_values[c3e(solver->cell_idx, Nx+1, j, k) - solver->idx_first]
                        = bc_val_p(solver, DIR_EAST, solver->time + solver->dt, xmax, c1e(yc, j), c1e(zc, k))
                        - bc_val_p(solver, DIR_EAST, solver->time, xmax, c1e(yc, j), c1e(zc, k));
                }
            }
            break;
        default:;
        }
    }
    if (solver->rj == 0) {
        switch (solver->bc[2].type) {
        case BC_PRESSURE:
        case BC_VELOCITY_PERIODIC:
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    solver->vector_values[c3e(solver->cell_idx, i, -1, k) - solver->idx_first]
                        = solver->vector_values[c3e(solver->cell_idx, i, -2, k) - solver->idx_first]
                        = bc_val_p(solver, DIR_SOUTH, solver->time + solver->dt, c1e(xc, i), ymin, c1e(zc, k))
                        - bc_val_p(solver, DIR_SOUTH, solver->time, c1e(xc, i), ymin, c1e(zc, k));
                }
            }
        default:;
        }
    }
    if (solver->rj == solver->Py-1) {
        switch (solver->bc[0].type) {
        case BC_PRESSURE:
        case BC_VELOCITY_PERIODIC:
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    solver->vector_values[c3e(solver->cell_idx, i, Ny, k) - solver->idx_first]
                        = solver->vector_values[c3e(solver->cell_idx, i, Ny+1, k) - solver->idx_first]
                        = bc_val_p(solver, DIR_NORTH, solver->time + solver->dt, c1e(xc, i), ymax, c1e(zc, k))
                        - bc_val_p(solver, DIR_NORTH, solver->time, c1e(xc, i), ymax, c1e(zc, k));
                }
            }
        default:;
        }
    }
    if (solver->rk == 0) {
        switch (solver->bc[4].type) {
        case BC_PRESSURE:
        case BC_VELOCITY_PERIODIC:
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    solver->vector_values[c3e(solver->cell_idx, i, j, -1) - solver->idx_first]
                        = solver->vector_values[c3e(solver->cell_idx, i, j, -2) - solver->idx_first]
                        = bc_val_p(solver, DIR_DOWN, solver->time + solver->dt, c1e(xc, i), c1e(yc, j), zmin)
                        - bc_val_p(solver, DIR_DOWN, solver->time, c1e(xc, i), c1e(yc, j), zmin);
                }
            }
            break;
        default:;
        }
    }
    if (solver->rk == solver->Pz-1) {
        switch (solver->bc[5].type) {
        case BC_PRESSURE:
        case BC_VELOCITY_PERIODIC:
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    solver->vector_values[c3e(solver->cell_idx, i, j, Nz) - solver->idx_first]
                        = solver->vector_values[c3e(solver->cell_idx, i, j, Nz+1) - solver->idx_first]
                        = bc_val_p(solver, DIR_UP, solver->time + solver->dt, c1e(xc, i), c1e(yc, j), zmax)
                        - bc_val_p(solver, DIR_UP, solver->time, c1e(xc, i), c1e(yc, j), zmax);
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
    if (HYPRE_CheckError(hypre_ierr, HYPRE_ERROR_GENERIC)) {
        fprintf(stderr, "error: floating pointer error raised in p_prime\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    if (HYPRE_CheckError(hypre_ierr, HYPRE_ERROR_CONV)) {
        fprintf(stderr, "warning: p_prime did not converge\n");
    }

    HYPRE_IJVectorGetValues(solver->x, solver->idx_last-solver->idx_first, solver->vector_rows, solver->vector_res);
    for (int i = ifirst; i < ilast; i++) {
        for (int j = jfirst; j < jlast; j++) {
            for (int k = kfirst; k < klast; k++) {
                c3e(p_prime, i, j, k) = solver->vector_res[c3e(solver->cell_idx, i, j, k)-solver->idx_first];
            }
        }
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

    /* Exchange p_prime between adjacent processes. */
    exchange_var(solver, solver->p_prime);
}

static inline void update_next(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;
    const double dt = solver->dt;

    const double *const xc = solver->xc;
    const double *const yc = solver->yc;
    const double *const zc = solver->zc;

    const int *const flag = solver->flag;

    const double *const u1_star = solver->u1_star;
    const double *const u2_star = solver->u2_star;
    const double *const u3_star = solver->u3_star;

    const double *const U1_star = solver->U1_star;
    const double *const U2_star = solver->U2_star;
    const double *const U3_star = solver->U3_star;

    const double *const p = solver->p;
    const double *const p_prime = solver->p_prime;

    double *const u1_next = solver->u1_next;
    double *const u2_next = solver->u2_next;
    double *const u3_next = solver->u3_next;
    double *const U1_next = solver->U1_next;
    double *const U2_next = solver->U2_next;
    double *const U3_next = solver->U3_next;
    double *const p_next = solver->p_next;

    /* Calculate p_next. */
    FOR_INNER_CELL (i, j, k) {
        if (c3e(flag, i, j, k) == FLAG_FLUID || c3e(flag, i, j, k) == FLAG_GHOST) {
            c3e(p_next, i, j, k) = c3e(p, i, j, k) + c3e(p_prime, i, j, k);
        }
        else {
            c3e(p_next, i, j, k) = NAN;
        }
    }

    /* Calculate u_next. */
    FOR_INNER_CELL (i, j, k) {
        if (c3e(flag, i, j, k) == FLAG_FLUID) {
            c3e(u1_next, i, j, k) = c3e(u1_star, i, j, k)
                - dt * (c3e(p_prime, i+1, j, k) - c3e(p_prime, i-1, j, k)) / (c1e(xc, i+1) - c1e(xc, i-1));
            c3e(u2_next, i, j, k) = c3e(u2_star, i, j, k)
                - dt * (c3e(p_prime, i, j+1, k) - c3e(p_prime, i, j-1, k)) / (c1e(yc, j+1) - c1e(yc, j-1));
            c3e(u3_next, i, j, k) = c3e(u3_star, i, j, k)
                - dt * (c3e(p_prime, i, j, k+1) - c3e(p_prime, i, j, k-1)) / (c1e(zc, k+1) - c1e(zc, k-1));
        }
        else if (c3e(flag, i, j, k) == FLAG_SOLID) {
            c3e(u1_next, i, j, k) = c3e(u2_next, i, j, k) = c3e(u3_next, i, j, k) = NAN;
        }
    }

    /* Calculate U_next. */
    FOR_ALL_XSTAG (i, j, k) {
        if (c3e(flag, i-1, j, k) != FLAG_SOLID && c3e(flag, i, j, k) != FLAG_SOLID) {
            xse(U1_next, i, j, k) = xse(U1_star, i, j, k)
                - dt * (c3e(p_prime, i, j, k) - c3e(p_prime, i-1, j, k)) / (c1e(xc, i) - c1e(xc, i-1));
        }
        else {
            xse(U1_next, i, j, k) = NAN;
        }
    }
    FOR_ALL_YSTAG (i, j, k) {
        if (c3e(flag, i, j-1, k) != FLAG_SOLID && c3e(flag, i, j, k) != FLAG_SOLID) {
            yse(U2_next, i, j, k) = yse(U2_star, i, j, k)
                - dt * (c3e(p_prime, i, j, k) - c3e(p_prime, i, j-1, k)) / (c1e(yc, j) - c1e(yc, j-1));
        }
        else {
            yse(U2_next, i, j, k) = NAN;
        }
    }
    FOR_ALL_ZSTAG (i, j, k) {
        if (c3e(flag, i, j, k-1) != FLAG_SOLID && c3e(flag, i, j, k) != FLAG_SOLID) {
            zse(U3_next, i, j, k) = zse(U3_star, i, j, k)
                - dt * (c3e(p_prime, i, j, k) - c3e(p_prime, i, j, k-1)) / (c1e(zc, k) - c1e(zc, k-1));
        }
        else {
            zse(U3_next, i, j, k) = NAN;
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

    int ifirst, ilast, jfirst, jlast;

    double a, b;
    int cnt;

    ifirst = solver->ri == 0 ? -2 : 0;
    ilast = solver->ri == solver->Px-1 ? solver->Nx+2 : solver->Nx;
    jfirst = solver->rj == 0 ? -2 : 0;
    jlast = solver->rj == solver->Py-1 ? solver->Ny+2 : solver->Ny;

    /* Set velocity boundary conditions. */

    /* West. */
    if (solver->ri == 0) {
        switch (solver->bc[3].type) {
        case BC_VELOCITY_COMPONENT:
            a = c1e(solver->dx, -1) / 2;
            b = c1e(solver->dx, 0) / 2;
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(u1, -1, j, k) = (a+b)/b*bc_val_u(solver, 1, DIR_WEST, solver->time, xmin, c1e(yc, j), c1e(zc, k)) - a/b*c3e(u1, 0, j, k);
                    c3e(u2, -1, j, k) = (a+b)/b*bc_val_u(solver, 2, DIR_WEST, solver->time, xmin, c1e(yc, j), c1e(zc, k)) - a/b*c3e(u2, 0, j, k);
                    c3e(u3, -1, j, k) = (a+b)/b*bc_val_u(solver, 3, DIR_WEST, solver->time, xmin, c1e(yc, j), c1e(zc, k)) - a/b*c3e(u3, 0, j, k);
                }
            }
            a = c1e(solver->dx, -2) / 2 + c1e(solver->dx, -1);
            b = c1e(solver->dx, 0) + c1e(solver->dx, 1) / 2;
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(u1, -2, j, k) = (a+b)/b*bc_val_u(solver, 1, DIR_WEST, solver->time, xmin, c1e(yc, j), c1e(zc, k)) - a/b*c3e(u1, 1, j, k);
                    c3e(u2, -2, j, k) = (a+b)/b*bc_val_u(solver, 2, DIR_WEST, solver->time, xmin, c1e(yc, j), c1e(zc, k)) - a/b*c3e(u2, 1, j, k);
                    c3e(u3, -2, j, k) = (a+b)/b*bc_val_u(solver, 3, DIR_WEST, solver->time, xmin, c1e(yc, j), c1e(zc, k)) - a/b*c3e(u3, 1, j, k);
                }
            }
            break;
        case BC_PRESSURE:
            a = c1e(solver->xc, 0) - c1e(solver->xc, -1);
            b = c1e(solver->xc, 1) - c1e(solver->xc, 0);
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(u1, -1, j, k) = (a+b)/b*c3e(u1, 0, j, k) - a/b*c3e(u1, 1, j, k);
                    c3e(u2, -1, j, k) = (a+b)/b*c3e(u2, 0, j, k) - a/b*c3e(u2, 1, j, k);
                    c3e(u3, -1, j, k) = (a+b)/b*c3e(u3, 0, j, k) - a/b*c3e(u3, 1, j, k);
                }
            }
            a = c1e(solver->xc, -1) - c1e(solver->xc, -2);
            b = c1e(solver->xc, 0) - c1e(solver->xc, -1);
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(u1, -2, j, k) = (a+b)/b*c3e(u1, -1, j, k) - a/b*c3e(u1, 0, j, k);
                    c3e(u2, -2, j, k) = (a+b)/b*c3e(u2, -1, j, k) - a/b*c3e(u2, 0, j, k);
                    c3e(u3, -2, j, k) = (a+b)/b*c3e(u3, -1, j, k) - a/b*c3e(u3, 0, j, k);
                }
            }
            break;
        case BC_FREE_SLIP_WALL:
            a = c1e(solver->dx, -1) / 2;
            b = c1e(solver->dx, 0) / 2;
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(u1, -1, j, k) = -a/b*c3e(u1, 0, j, k);
                    c3e(u2, -1, j, k) = c3e(u2, 0, j, k);
                    c3e(u3, -1, j, k) = c3e(u3, 0, j, k);
                }
            }
            a = c1e(solver->dx, -2) / 2 + c1e(solver->dx, -1);
            b = c1e(solver->dx, 0) + c1e(solver->dx, 1) / 2;
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
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
                    for (int j = 0; j < Ny; j++) {
                        for (int k = 0; k < Nz; k++) {
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
                    for (int j = 0; j < Ny; j++) {
                        for (int k = 0; k < Nz; k++) {
                            solver->x_exchg[cnt++] = c3e(u1, i, j, k);
                            solver->x_exchg[cnt++] = c3e(u2, i, j, k);
                            solver->x_exchg[cnt++] = c3e(u3, i, j, k);
                        }
                    }
                }
                MPI_Send(solver->x_exchg, 6*Ny*Nz, MPI_DOUBLE, solver->rank + (solver->Px-1)*solver->Py*solver->Pz, 0, MPI_COMM_WORLD);
                MPI_Recv(solver->x_exchg, 6*Ny*Nz, MPI_DOUBLE, solver->rank + (solver->Px-1)*solver->Py*solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cnt = 0;
                for (int i = -2; i <= -1; i++) {
                    for (int j = 0; j < Ny; j++) {
                        for (int k = 0; k < Nz; k++) {
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
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(u1, Nx, j, k) = (a+b)/a*bc_val_u(solver, 1, DIR_EAST, solver->time, xmax, c1e(yc, j), c1e(zc, k)) - b/a*c3e(u1, Nx-1, j, k);
                    c3e(u2, Nx, j, k) = (a+b)/a*bc_val_u(solver, 2, DIR_EAST, solver->time, xmax, c1e(yc, j), c1e(zc, k)) - b/a*c3e(u2, Nx-1, j, k);
                    c3e(u3, Nx, j, k) = (a+b)/a*bc_val_u(solver, 3, DIR_EAST, solver->time, xmax, c1e(yc, j), c1e(zc, k)) - b/a*c3e(u3, Nx-1, j, k);
                }
            }
            a = c1e(solver->dx, Nx-2) / 2 + c1e(solver->dx, Nx-1);
            b = c1e(solver->dx, Nx) + c1e(solver->dx, Nx+1) / 2;
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(u1, Nx+1, j, k) = (a+b)/a*bc_val_u(solver, 1, DIR_EAST, solver->time, xmax, c1e(yc, j), c1e(zc, k)) - b/a*c3e(u1, Nx-2, j, k);
                    c3e(u2, Nx+1, j, k) = (a+b)/a*bc_val_u(solver, 2, DIR_EAST, solver->time, xmax, c1e(yc, j), c1e(zc, k)) - b/a*c3e(u2, Nx-2, j, k);
                    c3e(u3, Nx+1, j, k) = (a+b)/a*bc_val_u(solver, 3, DIR_EAST, solver->time, xmax, c1e(yc, j), c1e(zc, k)) - b/a*c3e(u3, Nx-2, j, k);
                }
            }
            break;
        case BC_PRESSURE:
            a = c1e(solver->xc, Nx-1) - c1e(solver->xc, Nx-2);
            b = c1e(solver->xc, Nx) - c1e(solver->xc, Nx-1);
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(u1, Nx, j, k) = (a+b)/a*c3e(u1, Nx-1, j, k) - b/a*c3e(u1, Nx-2, j, k);
                    c3e(u2, Nx, j, k) = (a+b)/a*c3e(u2, Nx-1, j, k) - b/a*c3e(u2, Nx-2, j, k);
                    c3e(u3, Nx, j, k) = (a+b)/a*c3e(u3, Nx-1, j, k) - b/a*c3e(u3, Nx-2, j, k);
                }
            }
            a = c1e(solver->xc, Nx) - c1e(solver->xc, Nx-1);
            b = c1e(solver->xc, Nx+1) - c1e(solver->xc, Nx);
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(u1, Nx+1, j, k) = (a+b)/a*c3e(u1, Nx, j, k) - b/a*c3e(u1, Nx-1, j, k);
                    c3e(u2, Nx+1, j, k) = (a+b)/a*c3e(u2, Nx, j, k) - b/a*c3e(u2, Nx-1, j, k);
                    c3e(u3, Nx+1, j, k) = (a+b)/a*c3e(u3, Nx, j, k) - b/a*c3e(u3, Nx-1, j, k);
                }
            }
            break;
        case BC_FREE_SLIP_WALL:
            a = c1e(solver->dx, Nx-1) / 2;
            b = c1e(solver->dx, Nx) / 2;
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
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
                    for (int j = 0; j < Ny; j++) {
                        for (int k = 0; k < Nz; k++) {
                            c3e(u1, i, j, k) = c3e(u1, i-Nx, j, k);
                            c3e(u2, i, j, k) = c3e(u2, i-Nx, j, k);
                            c3e(u3, i, j, k) = c3e(u3, i-Nx, j, k);
                        }
                    }
                }
            }
            else {
                MPI_Recv(solver->x_exchg, 6*Ny*Nz, MPI_DOUBLE, solver->rank - (solver->Px-1)*solver->Py*solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cnt = 0;
                for (int i = Nx; i <= Nx+1; i++) {
                    for (int j = 0; j < Ny; j++) {
                        for (int k = 0; k < Nz; k++) {
                            c3e(u1, i, j, k) = solver->x_exchg[cnt++];
                            c3e(u2, i, j, k) = solver->x_exchg[cnt++];
                            c3e(u3, i, j, k) = solver->x_exchg[cnt++];
                        }
                    }
                }
                cnt = 0;
                for (int i = Nx-2; i <= Nx-1; i++) {
                    for (int j = 0; j < Ny; j++) {
                        for (int k = 0; k < Nz; k++) {
                            solver->x_exchg[cnt++] = c3e(u1, i, j, k);
                            solver->x_exchg[cnt++] = c3e(u2, i, j, k);
                            solver->x_exchg[cnt++] = c3e(u3, i, j, k);
                        }
                    }
                }
                MPI_Send(solver->x_exchg, 6*Ny*Nz, MPI_DOUBLE, solver->rank - (solver->Px-1)*solver->Py*solver->Pz, 0, MPI_COMM_WORLD);
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
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(u1, i, -1, k) = (a+b)/b*bc_val_u(solver, 1, DIR_SOUTH, solver->time, c1e(xc, i), ymin, c1e(zc, k)) - a/b*c3e(u1, i, 0, k);
                    c3e(u2, i, -1, k) = (a+b)/b*bc_val_u(solver, 2, DIR_SOUTH, solver->time, c1e(xc, i), ymin, c1e(zc, k)) - a/b*c3e(u2, i, 0, k);
                    c3e(u3, i, -1, k) = (a+b)/b*bc_val_u(solver, 3, DIR_SOUTH, solver->time, c1e(xc, i), ymin, c1e(zc, k)) - a/b*c3e(u3, i, 0, k);
                }
            }
            a = c1e(solver->dy, -2) / 2 + c1e(solver->dy, -1);
            b = c1e(solver->dy, 0) + c1e(solver->dy, 1) / 2;
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(u1, i, -2, k) = (a+b)/b*bc_val_u(solver, 1, DIR_SOUTH, solver->time, c1e(xc, i), ymin, c1e(zc, k)) - a/b*c3e(u1, i, 1, k);
                    c3e(u2, i, -2, k) = (a+b)/b*bc_val_u(solver, 2, DIR_SOUTH, solver->time, c1e(xc, i), ymin, c1e(zc, k)) - a/b*c3e(u2, i, 1, k);
                    c3e(u3, i, -2, k) = (a+b)/b*bc_val_u(solver, 3, DIR_SOUTH, solver->time, c1e(xc, i), ymin, c1e(zc, k)) - a/b*c3e(u3, i, 1, k);
                }
            }
            break;
        case BC_PRESSURE:
            a = c1e(solver->yc, 0) - c1e(solver->yc, -1);
            b = c1e(solver->yc, 1) - c1e(solver->yc, 0);
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(u1, i, -1, k) = (a+b)/b*c3e(u1, i, 0, k) - a/b*c3e(u1, i, 1, k);
                    c3e(u2, i, -1, k) = (a+b)/b*c3e(u2, i, 0, k) - a/b*c3e(u2, i, 1, k);
                    c3e(u3, i, -1, k) = (a+b)/b*c3e(u3, i, 0, k) - a/b*c3e(u3, i, 1, k);
                }
            }
            a = c1e(solver->yc, -1) - c1e(solver->yc, -2);
            b = c1e(solver->yc, 0) - c1e(solver->yc, -1);
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(u1, i, -2, k) = (a+b)/b*c3e(u1, i, -1, k) - a/b*c3e(u1, i, 0, k);
                    c3e(u2, i, -2, k) = (a+b)/b*c3e(u2, i, -1, k) - a/b*c3e(u2, i, 0, k);
                    c3e(u3, i, -2, k) = (a+b)/b*c3e(u3, i, -1, k) - a/b*c3e(u3, i, 0, k);
                }
            }
            break;
        case BC_FREE_SLIP_WALL:
            a = c1e(solver->dy, -1) / 2;
            b = c1e(solver->dy, 0) / 2;
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(u1, i, -1, k) = c3e(u1, i, 0, k);
                    c3e(u2, i, -1, k) = -a/b*c3e(u2, i, 0, k);
                    c3e(u3, i, -1, k) = c3e(u3, i, 0, k);
                }
            }
            a = c1e(solver->dy, -2) / 2 + c1e(solver->dy, -1);
            b = c1e(solver->dy, 0) + c1e(solver->dy, 1) / 2;
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(u1, i, -2, k) = c3e(u1, i, 1, k);
                    c3e(u2, i, -2, k) = -a/b*c3e(u2, i, 1, k);
                    c3e(u3, i, -2, k) = c3e(u3, i, 1, k);
                }
            }
            break;
        case BC_ALL_PERIODIC:
        case BC_VELOCITY_PERIODIC:
            if (solver->Py == 1) {
                for (int i = ifirst; i < ilast; i++) {
                    for (int j = -2; j <= -1; j++) {
                        for (int k = 0; k < Nz; k++) {
                            c3e(u1, i, j, k) = c3e(u1, i, j+Ny, k);
                            c3e(u2, i, j, k) = c3e(u2, i, j+Ny, k);
                            c3e(u3, i, j, k) = c3e(u3, i, j+Ny, k);
                        }
                    }
                }
            }
            else {
                cnt = 0;
                for (int i = ifirst; i < ilast; i++) {
                    for (int j = 0; j <= 1; j++) {
                        for (int k = 0; k < Nz; k++) {
                            solver->y_exchg[cnt++] = c3e(u1, i, j, k);
                            solver->y_exchg[cnt++] = c3e(u2, i, j, k);
                            solver->y_exchg[cnt++] = c3e(u3, i, j, k);
                        }
                    }
                }
                MPI_Send(solver->y_exchg, 6*(ilast-ifirst)*Nz, MPI_DOUBLE, solver->rank + (solver->Py-1)*solver->Pz, 0, MPI_COMM_WORLD);
                MPI_Recv(solver->y_exchg, 6*(ilast-ifirst)*Nz, MPI_DOUBLE, solver->rank + (solver->Py-1)*solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cnt = 0;
                for (int i = ifirst; i < ilast; i++) {
                    for (int j = -2; j <= -1; j++) {
                        for (int k = 0; k < Nz; k++) {
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
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(u1, i, Ny, k) = (a+b)/a*bc_val_u(solver, 1, DIR_NORTH, solver->time, c1e(xc, i), ymax, c1e(zc, k)) - b/a*c3e(u1, i, Ny-1, k);
                    c3e(u2, i, Ny, k) = (a+b)/a*bc_val_u(solver, 2, DIR_NORTH, solver->time, c1e(xc, i), ymax, c1e(zc, k)) - b/a*c3e(u2, i, Ny-1, k);
                    c3e(u3, i, Ny, k) = (a+b)/a*bc_val_u(solver, 3, DIR_NORTH, solver->time, c1e(xc, i), ymax, c1e(zc, k)) - b/a*c3e(u3, i, Ny-1, k);
                }
            }
            a = c1e(solver->dy, Ny-2) / 2 + c1e(solver->dy, Ny-1);
            b = c1e(solver->dy, Ny) + c1e(solver->dy, Ny+1) / 2;
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(u1, i, Ny+1, k) = (a+b)/a*bc_val_u(solver, 1, DIR_NORTH, solver->time, c1e(xc, i), ymax, c1e(zc, k)) - b/a*c3e(u1, i, Ny-2, k);
                    c3e(u2, i, Ny+1, k) = (a+b)/a*bc_val_u(solver, 2, DIR_NORTH, solver->time, c1e(xc, i), ymax, c1e(zc, k)) - b/a*c3e(u2, i, Ny-2, k);
                    c3e(u3, i, Ny+1, k) = (a+b)/a*bc_val_u(solver, 3, DIR_NORTH, solver->time, c1e(xc, i), ymax, c1e(zc, k)) - b/a*c3e(u3, i, Ny-2, k);
                }
            }
            break;
        case BC_PRESSURE:
            a = c1e(solver->yc, Ny-1) - c1e(solver->yc, Ny-2);
            b = c1e(solver->yc, Ny) - c1e(solver->yc, Ny-1);
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(u1, i, Ny, k) = (a+b)/a*c3e(u1, i, Ny-1, k) - b/a*c3e(u1, i, Ny-2, k);
                    c3e(u2, i, Ny, k) = (a+b)/a*c3e(u2, i, Ny-1, k) - b/a*c3e(u2, i, Ny-2, k);
                    c3e(u3, i, Ny, k) = (a+b)/a*c3e(u3, i, Ny-1, k) - b/a*c3e(u3, i, Ny-2, k);
                }
            }
            a = c1e(solver->yc, Ny) - c1e(solver->yc, Ny-1);
            b = c1e(solver->yc, Ny+1) - c1e(solver->yc, Ny);
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(u1, i, Ny+1, k) = (a+b)/a*c3e(u1, i, Ny, k) - b/a*c3e(u1, i, Ny-1, k);
                    c3e(u2, i, Ny+1, k) = (a+b)/a*c3e(u2, i, Ny, k) - b/a*c3e(u2, i, Ny-1, k);
                    c3e(u3, i, Ny+1, k) = (a+b)/a*c3e(u3, i, Ny, k) - b/a*c3e(u3, i, Ny-1, k);
                }
            }
            break;
        case BC_FREE_SLIP_WALL:
            a = c1e(solver->dy, Ny-1) / 2;
            b = c1e(solver->dy, Ny) / 2;
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(u1, i, Ny, k) = c3e(u1, i, Ny-1, k);
                    c3e(u2, i, Ny, k) = -b/a*c3e(u2, i, Ny-1, k);
                    c3e(u3, i, Ny, k) = c3e(u3, i, Ny-1, k);
                }
            }
            a = c1e(solver->dy, Ny-2) / 2 + c1e(solver->dy, Ny-1);
            b = c1e(solver->dy, Ny) + c1e(solver->dy, Ny+1) / 2;
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(u1, i, Ny+1, k) = c3e(u1, i, Ny-2, k);
                    c3e(u2, i, Ny+1, k) = -b/a*c3e(u2, i, Ny-2, k);
                    c3e(u3, i, Ny+1, k) = c3e(u3, i, Ny-2, k);
                }
            }
            break;
        case BC_ALL_PERIODIC:
        case BC_VELOCITY_PERIODIC:
            if (solver->Py == 1) {
                for (int i = ifirst; i < ilast; i++) {
                    for (int j = Ny; j <= Ny+1; j++) {
                        for (int k = 0; k < Nz; k++) {
                            c3e(u1, i, j, k) = c3e(u1, i, j-Ny, k);
                            c3e(u2, i, j, k) = c3e(u2, i, j-Ny, k);
                            c3e(u3, i, j, k) = c3e(u3, i, j-Ny, k);
                        }
                    }
                }
            }
            else {
                MPI_Recv(solver->y_exchg, 6*(ilast-ifirst)*Nz, MPI_DOUBLE, solver->rank - (solver->Py-1)*solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cnt = 0;
                for (int i = ifirst; i < ilast; i++) {
                    for (int j = Ny; j <= Ny+1; j++) {
                        for (int k = 0; k < Nz; k++) {
                            c3e(u1, i, j, k) = solver->y_exchg[cnt++];
                            c3e(u2, i, j, k) = solver->y_exchg[cnt++];
                            c3e(u3, i, j, k) = solver->y_exchg[cnt++];
                        }
                    }
                }
                cnt = 0;
                for (int i = ifirst; i < ilast; i++) {
                    for (int j = Ny-2; j <= Ny-1; j++) {
                        for (int k = 0; k < Nz; k++) {
                            solver->y_exchg[cnt++] = c3e(u1, i, j, k);
                            solver->y_exchg[cnt++] = c3e(u2, i, j, k);
                            solver->y_exchg[cnt++] = c3e(u3, i, j, k);
                        }
                    }
                }
                MPI_Send(solver->y_exchg, 6*(ilast-ifirst)*Nz, MPI_DOUBLE, solver->rank - (solver->Py-1)*solver->Pz, 0, MPI_COMM_WORLD);
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
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    c3e(u1, i, j, -1) = (a+b)/b*bc_val_u(solver, 1, DIR_DOWN, solver->time, c1e(xc, i), c1e(yc, j), zmin) - a/b*c3e(u1, i, j, 0);
                    c3e(u2, i, j, -1) = (a+b)/b*bc_val_u(solver, 2, DIR_DOWN, solver->time, c1e(xc, i), c1e(yc, j), zmin) - a/b*c3e(u2, i, j, 0);
                    c3e(u3, i, j, -1) = (a+b)/b*bc_val_u(solver, 3, DIR_DOWN, solver->time, c1e(xc, i), c1e(yc, j), zmin) - a/b*c3e(u3, i, j, 0);
                }
            }
            a = c1e(solver->dz, -2) / 2 + c1e(solver->dz, -1);
            b = c1e(solver->dz, 0) + c1e(solver->dz, 1) / 2;
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    c3e(u1, i, j, -2) = (a+b)/b*bc_val_u(solver, 1, DIR_DOWN, solver->time, c1e(xc, i), c1e(yc, j), zmin) - a/b*c3e(u1, i, j, 1);
                    c3e(u2, i, j, -2) = (a+b)/b*bc_val_u(solver, 2, DIR_DOWN, solver->time, c1e(xc, i), c1e(yc, j), zmin) - a/b*c3e(u2, i, j, 1);
                    c3e(u3, i, j, -2) = (a+b)/b*bc_val_u(solver, 3, DIR_DOWN, solver->time, c1e(xc, i), c1e(yc, j), zmin) - a/b*c3e(u3, i, j, 1);
                }
            }
            break;
        case BC_PRESSURE:
            a = c1e(solver->zc, 0) - c1e(solver->zc, -1);
            b = c1e(solver->zc, 1) - c1e(solver->zc, 0);
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    c3e(u1, i, j, -1) = (a+b)/b*c3e(u1, i, j, 0) - a/b*c3e(u1, i, j, 1);
                    c3e(u2, i, j, -1) = (a+b)/b*c3e(u2, i, j, 0) - a/b*c3e(u2, i, j, 1);
                    c3e(u3, i, j, -1) = (a+b)/b*c3e(u3, i, j, 0) - a/b*c3e(u3, i, j, 1);
                }
            }
            a = c1e(solver->zc, -1) - c1e(solver->zc, -2);
            b = c1e(solver->zc, 0) - c1e(solver->zc, -1);
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    c3e(u1, i, j, -2) = (a+b)/b*c3e(u1, i, j, -1) - a/b*c3e(u1, i, j, 0);
                    c3e(u2, i, j, -2) = (a+b)/b*c3e(u2, i, j, -1) - a/b*c3e(u2, i, j, 0);
                    c3e(u3, i, j, -2) = (a+b)/b*c3e(u3, i, j, -1) - a/b*c3e(u3, i, j, 0);
                }
            }
            break;
        case BC_FREE_SLIP_WALL:
            a = c1e(solver->dz, -1) / 2;
            b = c1e(solver->dz, 0) / 2;
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    c3e(u1, i, j, -1) = c3e(u1, i, j, 0);
                    c3e(u2, i, j, -1) = c3e(u2, i, j, 0);
                    c3e(u3, i, j, -1) = -a/b*c3e(u3, i, j, 0);
                }
            }
            a = c1e(solver->dz, -2) / 2 + c1e(solver->dz, -1);
            b = c1e(solver->dz, 0) + c1e(solver->dz, 1) / 2;
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    c3e(u1, i, j, -2) = c3e(u1, i, j, 1);
                    c3e(u2, i, j, -2) = c3e(u2, i, j, 1);
                    c3e(u3, i, j, -2) = -a/b*c3e(u3, i, j, 1);
                }
            }
            break;
        case BC_ALL_PERIODIC:
        case BC_VELOCITY_PERIODIC:
            if (solver->Pz == 1) {
                for (int i = ifirst; i < ilast; i++) {
                    for (int j = jfirst; j < jlast; j++) {
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
                for (int i = ifirst; i < ilast; i++) {
                    for (int j = jfirst; j < jlast; j++) {
                        for (int k = 0; k <= 1; k++) {
                            solver->z_exchg[cnt++] = c3e(u1, i, j, k);
                            solver->z_exchg[cnt++] = c3e(u2, i, j, k);
                            solver->z_exchg[cnt++] = c3e(u3, i, j, k);
                        }
                    }
                }
                MPI_Send(solver->z_exchg, 6*(ilast-ifirst)*(jlast-jfirst), MPI_DOUBLE, solver->rank + (solver->Pz-1), 0, MPI_COMM_WORLD);
                MPI_Recv(solver->z_exchg, 6*(ilast-ifirst)*(jlast-jfirst), MPI_DOUBLE, solver->rank + (solver->Pz-1), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cnt = 0;
                for (int i = ifirst; i < ilast; i++) {
                    for (int j = jfirst; j < jlast; j++) {
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
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    c3e(u1, i, j, Nz) = (a+b)/a*bc_val_u(solver, 1, DIR_UP, solver->time, c1e(xc, i), c1e(yc, j), zmax) - b/a*c3e(u1, i, j, Nz-1);
                    c3e(u2, i, j, Nz) = (a+b)/a*bc_val_u(solver, 2, DIR_UP, solver->time, c1e(xc, i), c1e(yc, j), zmax) - b/a*c3e(u2, i, j, Nz-1);
                    c3e(u3, i, j, Nz) = (a+b)/a*bc_val_u(solver, 3, DIR_UP, solver->time, c1e(xc, i), c1e(yc, j), zmax) - b/a*c3e(u3, i, j, Nz-1);
                }
            }
            a = c1e(solver->dz, Nz-2) / 2 + c1e(solver->dz, Nz-1);
            b = c1e(solver->dz, Nz) + c1e(solver->dz, Nz+1) / 2;
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    c3e(u1, i, j, Nz+1) = (a+b)/a*bc_val_u(solver, 1, DIR_UP, solver->time, c1e(xc, i), c1e(yc, j), zmax) - b/a*c3e(u1, i, j, Nz-2);
                    c3e(u2, i, j, Nz+1) = (a+b)/a*bc_val_u(solver, 2, DIR_UP, solver->time, c1e(xc, i), c1e(yc, j), zmax) - b/a*c3e(u2, i, j, Nz-2);
                    c3e(u3, i, j, Nz+1) = (a+b)/a*bc_val_u(solver, 3, DIR_UP, solver->time, c1e(xc, i), c1e(yc, j), zmax) - b/a*c3e(u3, i, j, Nz-2);
                }
            }
            break;
        case BC_PRESSURE:
            a = c1e(solver->zc, Nz-1) - c1e(solver->zc, Nz-2);
            b = c1e(solver->zc, Nz) - c1e(solver->zc, Nz-1);
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    c3e(u1, i, j, Nz) = (a+b)/a*c3e(u1, i, j, Nz-1) - b/a*c3e(u1, i, j, Nz-2);
                    c3e(u2, i, j, Nz) = (a+b)/a*c3e(u2, i, j, Nz-1) - b/a*c3e(u2, i, j, Nz-2);
                    c3e(u3, i, j, Nz) = (a+b)/a*c3e(u3, i, j, Nz-1) - b/a*c3e(u3, i, j, Nz-2);
                }
            }
            a = c1e(solver->zc, Nz) - c1e(solver->zc, Nz-1);
            b = c1e(solver->zc, Nz+1) - c1e(solver->zc, Nz);
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    c3e(u1, i, j, Nz+1) = (a+b)/a*c3e(u1, i, j, Nz) - b/a*c3e(u1, i, j, Nz-1);
                    c3e(u2, i, j, Nz+1) = (a+b)/a*c3e(u2, i, j, Nz) - b/a*c3e(u2, i, j, Nz-1);
                    c3e(u3, i, j, Nz+1) = (a+b)/a*c3e(u3, i, j, Nz) - b/a*c3e(u3, i, j, Nz-1);
                }
            }
            break;
        case BC_FREE_SLIP_WALL:
            a = c1e(solver->dz, Nz-1) / 2;
            b = c1e(solver->dz, Nz) / 2;
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    c3e(u1, i, j, Nz) = c3e(u1, i, j, Nz-1);
                    c3e(u2, i, j, Nz) = c3e(u2, i, j, Nz-1);
                    c3e(u3, i, j, Nz) = -b/a*c3e(u3, i, j, Nz-1);
                }
            }
            a = c1e(solver->dz, Nz-2) / 2 + c1e(solver->dz, Nz-1);
            b = c1e(solver->dz, Nz) + c1e(solver->dz, Nz+1) / 2;
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    c3e(u1, i, j, Nz+1) = c3e(u1, i, j, Nz-2);
                    c3e(u2, i, j, Nz+1) = c3e(u2, i, j, Nz-2);
                    c3e(u3, i, j, Nz+1) = -b/a*c3e(u3, i, j, Nz-2);
                }
            }
            break;
        case BC_ALL_PERIODIC:
        case BC_VELOCITY_PERIODIC:
            if (solver->Pz == 1) {
                for (int i = ifirst; i < ilast; i++) {
                    for (int j = jfirst; j < jlast; j++) {
                        for (int k = Nz; k <= Nz+1; k++) {
                            c3e(u1, i, j, k) = c3e(u1, i, j, k-Nz);
                            c3e(u2, i, j, k) = c3e(u2, i, j, k-Nz);
                            c3e(u3, i, j, k) = c3e(u3, i, j, k-Nz);
                        }
                    }
                }
            }
            else {
                MPI_Recv(solver->z_exchg, 6*(ilast-ifirst)*(jlast-jfirst), MPI_DOUBLE, solver->rank - (solver->Pz-1), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cnt = 0;
                for (int i = ifirst; i < ilast; i++) {
                    for (int j = jfirst; j < jlast; j++) {
                        for (int k = Nz; k <= Nz+1; k++) {
                            c3e(u1, i, j, k) = solver->z_exchg[cnt++];
                            c3e(u2, i, j, k) = solver->z_exchg[cnt++];
                            c3e(u3, i, j, k) = solver->z_exchg[cnt++];
                        }
                    }
                }
                cnt = 0;
                for (int i = ifirst; i < ilast; i++) {
                    for (int j = jfirst; j < jlast; j++) {
                        for (int k = Nz-2; k <= Nz-1; k++) {
                            solver->z_exchg[cnt++] = c3e(u1, i, j, k);
                            solver->z_exchg[cnt++] = c3e(u2, i, j, k);
                            solver->z_exchg[cnt++] = c3e(u3, i, j, k);
                        }
                    }
                }
                MPI_Send(solver->z_exchg, 6*(ilast-ifirst)*(jlast-jfirst), MPI_DOUBLE, solver->rank - (solver->Pz-1), 0, MPI_COMM_WORLD);
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
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(p, -1, j, k) = c3e(p, 0, j, k);
                    c3e(p, -2, j, k) = c3e(p, 1, j, k);
                }
            }
            break;
        case BC_PRESSURE:
        case BC_VELOCITY_PERIODIC:
            a = c1e(solver->dx, -1) / 2;
            b = c1e(solver->dx, 0) / 2;
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(p, -1, j, k) = (a+b)/b*bc_val_p(solver, DIR_WEST, solver->time, xmin, c1e(yc, j), c1e(zc, k)) - a/b*c3e(p, 0, j, k);
                }
            }
            a = c1e(solver->dx, -2) / 2 + c1e(solver->dx, -1);
            b = c1e(solver->dx, 0) + c1e(solver->dx, 1) / 2;
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(p, -2, j, k) = (a+b)/b*bc_val_p(solver, DIR_WEST, solver->time, xmin, c1e(yc, j), c1e(zc, k)) - a/b*c3e(p, 1, j, k);
                }
            }
            break;
        case BC_ALL_PERIODIC:
            if (solver->Px == 1) {
                for (int i = -2; i <= -1; i++) {
                    for (int j = 0; j < Ny; j++) {
                        for (int k = 0; k < Nz; k++) {
                            c3e(p, i, j, k) = c3e(p, i+Nx, j, k);
                        }
                    }
                }
            }
            else {
                cnt = 0;
                for (int i = 0; i <= 1; i++) {
                    for (int j = 0; j < Ny; j++) {
                        for (int k = 0; k < Nz; k++) {
                            solver->x_exchg[cnt++] = c3e(p, i, j, k);
                        }
                    }
                }
                MPI_Send(solver->x_exchg, 2*Ny*Nz, MPI_DOUBLE, solver->rank + (solver->Px-1)*solver->Py*solver->Pz, 0, MPI_COMM_WORLD);
                MPI_Recv(solver->x_exchg, 2*Ny*Nz, MPI_DOUBLE, solver->rank + (solver->Px-1)*solver->Py*solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cnt = 0;
                for (int i = -2; i <= -1; i++) {
                    for (int j = 0; j < Ny; j++) {
                        for (int k = 0; k < Nz; k++) {
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
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(p, Nx, j, k) = c3e(p, Nx-1, j, k);
                    c3e(p, Nx+1, j, k) = c3e(p, Nx-2, j, k);
                }
            }
            break;
        case BC_PRESSURE:
        case BC_VELOCITY_PERIODIC:
            a = c1e(solver->dx, Nx-1) / 2;
            b = c1e(solver->dx, Nx) / 2;
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(p, Nx, j, k) = (a+b)/a*bc_val_p(solver, DIR_EAST, solver->time, xmax, c1e(yc, j), c1e(zc, k)) - b/a*c3e(p, Nx-1, j, k);
                }
            }
            a = c1e(solver->dx, Nx-2) / 2 + c1e(solver->dx, Nx-1);
            b = c1e(solver->dx, Nx) + c1e(solver->dx, Nx+1) / 2;
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(p, Nx+1, j, k) = (a+b)/a*bc_val_p(solver, DIR_EAST, solver->time, xmax, c1e(yc, j), c1e(zc, k)) - b/a*c3e(p, Nx-2, j, k);
                }
            }
            break;
        case BC_ALL_PERIODIC:
            if (solver->Px == 1) {
                for (int i = Nx; i <= Nx+1; i++) {
                    for (int j = 0; j < Ny; j++) {
                        for (int k = 0; k < Nz; k++) {
                            c3e(p, i, j, k) = c3e(p, i-Nx, j, k);
                        }
                    }
                }
            }
            else {
                MPI_Recv(solver->x_exchg, 2*Ny*Nz, MPI_DOUBLE, solver->rank - (solver->Px-1)*solver->Py*solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cnt = 0;
                for (int i = Nx; i <= Nx+1; i++) {
                    for (int j = 0; j < Ny; j++) {
                        for (int k = 0; k < Nz; k++) {
                            c3e(p, i, j, k) = solver->x_exchg[cnt++];
                        }
                    }
                }
                cnt = 0;
                for (int i = Nx-2; i <= Nx-1; i++) {
                    for (int j = 0; j < Ny; j++) {
                        for (int k = 0; k < Nz; k++) {
                            solver->x_exchg[cnt++] = c3e(p, i, j, k);
                        }
                    }
                }
                MPI_Send(solver->x_exchg, 2*Ny*Nz, MPI_DOUBLE, solver->rank - (solver->Px-1)*solver->Py*solver->Pz, 0, MPI_COMM_WORLD);
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
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(p, i, -1, k) = c3e(p, i, 0, k);
                    c3e(p, i, -2, k) = c3e(p, i, 1, k);
                }
            }
            break;
        case BC_PRESSURE:
        case BC_VELOCITY_PERIODIC:
            a = c1e(solver->dy, -1) / 2;
            b = c1e(solver->dy, 0) / 2;
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(p, i, -1, k) = (a+b)/b*bc_val_p(solver, DIR_SOUTH, solver->time, c1e(xc, i), ymin, c1e(zc, k)) - a/b*c3e(p, i, 0, k);
                }
            }
            a = c1e(solver->dy, -2) / 2 + c1e(solver->dy, -1);
            b = c1e(solver->dy, 0) + c1e(solver->dy, 1) / 2;
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(p, i, -2, k) = (a+b)/b*bc_val_p(solver, DIR_SOUTH, solver->time, c1e(xc, i), ymin, c1e(zc, k)) - a/b*c3e(p, i, 1, k);
                }
            }
            break;
        case BC_ALL_PERIODIC:
            if (solver->Py == 1) {
                for (int i = ifirst; i < ilast; i++) {
                    for (int j = -2; j <= -1; j++) {
                        for (int k = 0; k < Nz; k++) {
                            c3e(p, i, j, k) = c3e(p, i, j+Ny, k);
                        }
                    }
                }
            }
            else {
                cnt = 0;
                for (int i = ifirst; i < ilast; i++) {
                    for (int j = 0; j <= 1; j++) {
                        for (int k = 0; k < Nz; k++) {
                            solver->y_exchg[cnt++] = c3e(p, i, j, k);
                        }
                    }
                }
                MPI_Send(solver->y_exchg, 2*(ilast-ifirst)*Nz, MPI_DOUBLE, solver->rank + (solver->Py-1)*solver->Pz, 0, MPI_COMM_WORLD);
                MPI_Recv(solver->y_exchg, 2*(ilast-ifirst)*Nz, MPI_DOUBLE, solver->rank + (solver->Py-1)*solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cnt = 0;
                for (int i = ifirst; i < ilast; i++) {
                    for (int j = -2; j <= -1; j++) {
                        for (int k = 0; k < Nz; k++) {
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
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(p, i, Ny, k) = c3e(p, i, Ny-1, k);
                    c3e(p, i, Ny+1, k) = c3e(p, i, Ny-2, k);
                }
            }
            break;
        case BC_PRESSURE:
        case BC_VELOCITY_PERIODIC:
            a = c1e(solver->dy, Ny-1) / 2;
            b = c1e(solver->dy, Ny) / 2;
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(p, i, Ny, k) = (a+b)/a*bc_val_p(solver, DIR_NORTH, solver->time, c1e(xc, i), ymax, c1e(zc, k)) - b/a*c3e(p, i, Ny-1, k);
                }
            }
            a = c1e(solver->dy, Ny-2) / 2 + c1e(solver->dy, Ny-1);
            b = c1e(solver->dy, Ny) + c1e(solver->dy, Ny+1) / 2;
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    c3e(p, i, Ny+1, k) = (a+b)/a*bc_val_p(solver, DIR_NORTH, solver->time, c1e(xc, i), ymax, c1e(zc, k)) - b/a*c3e(p, i, Ny-2, k);
                }
            }
            break;
        case BC_ALL_PERIODIC:
            if (solver->Py == 1) {
                for (int i = ifirst; i < ilast; i++) {
                    for (int j = Ny; j <= Ny+1; j++) {
                        for (int k = 0; k < Nz; k++) {
                            c3e(p, i, j, k) = c3e(p, i, j-Ny, k);
                        }
                    }
                }
            }
            else {
                MPI_Recv(solver->y_exchg, 2*(ilast-ifirst)*Nz, MPI_DOUBLE, solver->rank - (solver->Py-1)*solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cnt = 0;
                for (int i = ifirst; i < ilast; i++) {
                    for (int j = Ny; j <= Ny+1; j++) {
                        for (int k = 0; k < Nz; k++) {
                            c3e(p, i, j, k) = solver->y_exchg[cnt++];
                        }
                    }
                }
                cnt = 0;
                for (int i = ifirst; i < ilast; i++) {
                    for (int j = Ny-2; j <= Ny-1; j++) {
                        for (int k = 0; k < Nz; k++) {
                            solver->y_exchg[cnt++] = c3e(p, i, j, k);
                        }
                    }
                }
                MPI_Send(solver->y_exchg, 2*(ilast-ifirst)*Nz, MPI_DOUBLE, solver->rank - (solver->Py-1)*solver->Pz, 0, MPI_COMM_WORLD);
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
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    c3e(p, i, j, -1) = c3e(p, i, j, 0);
                    c3e(p, i, j, -2) = c3e(p, i, j, 1);
                }
            }
            break;
        case BC_PRESSURE:
        case BC_VELOCITY_PERIODIC:
            a = c1e(solver->dx, -1) / 2;
            b = c1e(solver->dx, 0) / 2;
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    c3e(p, i, j, -1) = (a+b)/b*bc_val_p(solver, DIR_DOWN, solver->time, c1e(xc, i), c1e(yc, j), zmin) - a/b*c3e(p, i, j, 0);
                }
            }
            a = c1e(solver->dx, -2) / 2 + c1e(solver->dx, -1);
            b = c1e(solver->dx, 0) + c1e(solver->dx, 1) / 2;
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    c3e(p, i, j, -2) = (a+b)/b*bc_val_p(solver, DIR_DOWN, solver->time, c1e(xc, i), c1e(yc, j), zmin) - a/b*c3e(p, i, j, 1);
                }
            }
            break;
        case BC_ALL_PERIODIC:
            if (solver->Pz == 1) {
                for (int i = ifirst; i < ilast; i++) {
                    for (int j = jfirst; j < jlast; j++) {
                        for (int k = -2; k <= -1; k++) {
                            c3e(p, i, j, k) = c3e(p, i, j, k+Nz);
                        }
                    }
                }
            }
            else {
                cnt = 0;
                for (int i = ifirst; i < ilast; i++) {
                    for (int j = jfirst; j < jlast; j++) {
                        for (int k = 0; k <= 1; k++) {
                            solver->z_exchg[cnt++] = c3e(p, i, j, k);
                        }
                    }
                }
                MPI_Send(solver->z_exchg, 2*(ilast-ifirst)*(jlast-jfirst), MPI_DOUBLE, solver->rank + (solver->Pz-1), 0, MPI_COMM_WORLD);
                MPI_Recv(solver->z_exchg, 2*(ilast-ifirst)*(jlast-jfirst), MPI_DOUBLE, solver->rank + (solver->Pz-1), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cnt = 0;
                for (int i = ifirst; i < ilast; i++) {
                    for (int j = jfirst; j < jlast; j++) {
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
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    c3e(p, i, j, Nz) = c3e(p, i, j, Nz-1);
                    c3e(p, i, j, Nz+1) = c3e(p, i, j, Nz-2);
                }
            }
            break;
        case BC_PRESSURE:
        case BC_VELOCITY_PERIODIC:
            a = c1e(solver->dz, Nz-1) / 2;
            b = c1e(solver->dz, Nz) / 2;
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    c3e(p, i, j, Nz) = (a+b)/a*bc_val_p(solver, DIR_UP, solver->time, c1e(xc, i), c1e(yc, j), zmax) - b/a*c3e(p, i, j, Nz-1);
                }
            }
            a = c1e(solver->dz, Nz-2) / 2 + c1e(solver->dz, Nz-1);
            b = c1e(solver->dz, Nz) + c1e(solver->dz, Nz+1) / 2;
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    c3e(p, i, j, Nz+1) = (a+b)/a*bc_val_p(solver, DIR_UP, solver->time, c1e(xc, i), c1e(yc, j), zmax) - b/a*c3e(p, i, j, Nz-2);
                }
            }
            break;
        case BC_ALL_PERIODIC:
            if (solver->Pz == 1) {
                for (int i = ifirst; i < ilast; i++) {
                    for (int j = jfirst; j < jlast; j++) {
                        for (int k = Nz; k <= Nz+1; k++) {
                                c3e(p, i, j, k) = c3e(p, i, j, k-Nz);
                            }
                        }
                    }
                }
            else {
                MPI_Recv(solver->z_exchg, 2*(ilast-ifirst)*(jlast-jfirst), MPI_DOUBLE, solver->rank - (solver->Pz-1), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cnt = 0;
                for (int i = ifirst; i < ilast; i++) {
                    for (int j = jfirst; j < jlast; j++) {
                        for (int k = Nz; k <= Nz+1; k++) {
                            c3e(p, i, j, k) = solver->z_exchg[cnt++];
                        }
                    }
                }
                cnt = 0;
                for (int i = ifirst; i < ilast; i++) {
                    for (int j = jfirst; j < jlast; j++) {
                        for (int k = Nz-2; k <= Nz-1; k++) {
                            solver->z_exchg[cnt++] = c3e(p, i, j, k);
                        }
                    }
                }
                MPI_Send(solver->z_exchg, 2*(ilast-ifirst)*(jlast-jfirst), MPI_DOUBLE, solver->rank - (solver->Pz-1), 0, MPI_COMM_WORLD);
            }
            break;
        default:;
        }
    }
}

static void adj_exchange(IBMSolver *solver) {
    exchange_var(solver, solver->u1);
    exchange_var(solver, solver->u2);
    exchange_var(solver, solver->u3);
    exchange_var(solver, solver->p);
}

static void update_ghost(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    const int *const flag = solver->flag;

    double *const u1 = solver->u1;
    double *const u2 = solver->u2;
    double *const u3 = solver->u3;
    double *const p = solver->p;

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
                }
                else if (c3e(solver->flag, interp_idx[l][0], interp_idx[l][1], interp_idx[l][2]) != FLAG_SOLID) {
                    sum_u1 += interp_coeff[l] * c3e(u1, interp_idx[l][0], interp_idx[l][1], interp_idx[l][2]);
                    sum_u2 += interp_coeff[l] * c3e(u2, interp_idx[l][0], interp_idx[l][1], interp_idx[l][2]);
                    sum_u3 += interp_coeff[l] * c3e(u3, interp_idx[l][0], interp_idx[l][1], interp_idx[l][2]);
                    sum_p += interp_coeff[l] * c3e(p, interp_idx[l][0], interp_idx[l][1], interp_idx[l][2]);
                }
            }

            c3e(u1, i, j, k) = -sum_u1 / coeff_lhs_u;
            c3e(u2, i, j, k) = -sum_u2 / coeff_lhs_u;
            c3e(u3, i, j, k) = -sum_u3 / coeff_lhs_u;
            if (coeff_lhs_p != 0) {
                c3e(p, i, j, k) = sum_p / coeff_lhs_p;
            }
        }
        else if (c3e(flag, i, j, k) == FLAG_SOLID) {
            c3e(u1, i, j, k) = c3e(u2, i, j, k) = c3e(u3, i, j, k) = c3e(p, i, j, k) = NAN;
        }
    }
}

static double ext_force(
    IBMSolver *solver,
    int type,
    double t, double x, double y, double z
) {
    switch (type) {
    case 1:
        return solver->ext_force.val_type == VAL_CONST
            ? solver->ext_force.const_f1
            : solver->ext_force.func_f1(t, x, y, z);
    case 2:
        return solver->ext_force.val_type == VAL_CONST
            ? solver->ext_force.const_f2
            : solver->ext_force.func_f2(t, x, y, z);
    case 3:
        return solver->ext_force.val_type == VAL_CONST
            ? solver->ext_force.const_f3
            : solver->ext_force.func_f3(t, x, y, z);
    default:;
    }
    return NAN;
}

static double bc_val_u(
    IBMSolver *solver,
    int type,
    IBMSolverDirection dir,
    double t, double x, double y, double z
) {
    int idx = dir_to_idx(dir);
    switch (type) {
    case 1:
        return solver->bc[idx].val_type == VAL_CONST
            ? solver->bc[idx].const_u1
            : solver->bc[idx].func_u1(t, x, y, z);
    case 2:
        return solver->bc[idx].val_type == VAL_CONST
            ? solver->bc[idx].const_u2
            : solver->bc[idx].func_u2(t, x, y, z);
    case 3:
        return solver->bc[idx].val_type == VAL_CONST
            ? solver->bc[idx].const_u3
            : solver->bc[idx].func_u3(t, x, y, z);
    default:;
    }
    return NAN;
}

static double bc_val_p(
    IBMSolver *solver,
    IBMSolverDirection dir,
    double t, double x, double y, double z
) {
    int idx = dir_to_idx(dir);
    return solver->bc[idx].val_type == VAL_CONST
        ? solver->bc[idx].const_p
        : solver->bc[idx].func_p(t, x, y, z);
}

static void exchange_var(IBMSolver *solver, double *var) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    int cnt;

    /* X. */
    if (solver->ri != solver->Px-1) {
        /* Send to next process. */
        cnt = 0;
        for (int i = Nx-2; i <= Nx-1; i++) {
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    solver->x_exchg[cnt++] = c3e(var, i, j, k);
                }
            }
        }
        MPI_Send(solver->x_exchg, 2*(Ny+4)*(Nz+4), MPI_DOUBLE, solver->rank + solver->Py*solver->Pz, 0, MPI_COMM_WORLD);
    }
    if (solver->ri != 0) {
        /* Receive from previous process. */
        MPI_Recv(solver->x_exchg, 2*(Ny+4)*(Nz+4), MPI_DOUBLE, solver->rank - solver->Py*solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        for (int i = -2; i <= -1; i++) {
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(var, i, j, k) = solver->x_exchg[cnt++];
                }
            }
        }
        /* Send to previous process. */
        cnt = 0;
        for (int i = 0; i <= 1; i++) {
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    solver->x_exchg[cnt++] = c3e(var, i, j, k);
                }
            }
        }
        MPI_Send(solver->x_exchg, 2*(Ny+4)*(Nz+4), MPI_DOUBLE, solver->rank - solver->Py*solver->Pz, 0, MPI_COMM_WORLD);
    }
    if (solver->ri != solver->Px-1) {
        /* Receive from next process. */
        MPI_Recv(solver->x_exchg, 2*(Ny+4)*(Nz+4), MPI_DOUBLE, solver->rank + solver->Py*solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        for (int i = Nx; i <= Nx+1; i++) {
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(var, i, j, k) = solver->x_exchg[cnt++];
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
                    solver->y_exchg[cnt++] = c3e(var, i, j, k);
                }
            }
        }
        MPI_Send(solver->y_exchg, 2*(Nx+4)*(Nz+4), MPI_DOUBLE, solver->rank + solver->Pz, 0, MPI_COMM_WORLD);
    }
    if (solver->rj != 0) {
        /* Receive from previous process. */
        MPI_Recv(solver->y_exchg, 2*(Nx+4)*(Nz+4), MPI_DOUBLE, solver->rank - solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int j = -2; j <= -1; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(var, i, j, k) = solver->y_exchg[cnt++];
                }
            }
        }
        /* Send to previous process. */
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int j = 0; j <= 1; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    solver->y_exchg[cnt++] = c3e(var, i, j, k);
                }
            }
        }
        MPI_Send(solver->y_exchg, 2*(Nx+4)*(Nz+4), MPI_DOUBLE, solver->rank - solver->Pz, 0, MPI_COMM_WORLD);
    }
    if (solver->rj != solver->Py-1) {
        /* Receive from next process. */
        MPI_Recv(solver->y_exchg, 2*(Nx+4)*(Nz+4), MPI_DOUBLE, solver->rank + solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int j = Ny; j <= Ny+1; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(var, i, j, k) = solver->y_exchg[cnt++];
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
                    solver->z_exchg[cnt++] = c3e(var, i, j, k);
                }
            }
        }
        MPI_Send(solver->z_exchg, 2*(Nx+4)*(Ny+4), MPI_DOUBLE, solver->rank + 1, 0, MPI_COMM_WORLD);
    }
    if (solver->rk != 0) {
        /* Receive from previous process. */
        MPI_Recv(solver->z_exchg, 2*(Nx+4)*(Ny+4), MPI_DOUBLE, solver->rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k <= -1; k++) {
                    c3e(var, i, j, k) = solver->z_exchg[cnt++];
                }
            }
        }
        /* Send to previous process. */
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int j = -2; j < Ny+2; j++) {
                for (int k = 0; k <= 1; k++) {
                    solver->z_exchg[cnt++] = c3e(var, i, j, k);
                }
            }
        }
        MPI_Send(solver->z_exchg, 2*(Nx+4)*(Ny+4), MPI_DOUBLE, solver->rank - 1, 0, MPI_COMM_WORLD);
    }
    if (solver->rk != solver->Pz-1) {
        /* Receive from next process. */
        MPI_Recv(solver->z_exchg, 2*(Nx+4)*(Ny+4), MPI_DOUBLE, solver->rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int j = -2; j < Ny+2; j++) {
                for (int k = Nz; k <= Nz+1; k++) {
                    c3e(var, i, j, k) = solver->z_exchg[cnt++];
                }
            }
        }
    }
}
