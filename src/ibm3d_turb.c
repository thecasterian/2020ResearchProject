#include "ibm3d_turb.h"

#include "utils.h"
#include <math.h>

static void IBMSolver_calc_S(IBMSolver *);

void IBMSolver_calc_tau_r(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    double delta, S_mag, l;

    switch (solver->turb_model.type) {
    case TURBMODEL_NONE:
        return;
    case TURBMODEL_SMAGORINSKY:
        IBMSolver_calc_S(solver);
        FOR_INNER_CELL (i, j, k) {
            if (c3e(solver->flag, i, j, k) == FLAG_FLUID) {
                delta = 2 * cbrt(c1e(solver->dx, i) * c1e(solver->dy, j) * c1e(solver->dz, k));

                S_mag = 0;
                for (int ii = 1; ii <= 3; ii++) {
                    for (int jj = 1; jj <= 3; jj++) {
                        S_mag += 2*sq(c3e(solver->S[ii][jj], i, j, k));
                    }
                }
                S_mag = sqrt(S_mag);

                l = min(solver->turb_model.Cs * delta, 0.41 * c3e(solver->lvset, i, j, k));

                for (int ii = 1; ii <= 3; ii++) {
                    for (int jj = ii; jj <= 3; jj++) {
                        c3e(solver->tau_r[ii][jj], i, j, k)
                            = -2 * sq(l) * S_mag * c3e(solver->S[ii][jj], i, j, k);
                    }
                }
            }
        }
        break;
    default:;
    }
}

static void IBMSolver_calc_S(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    const double *xc = solver->xc;
    const double *yc = solver->yc;
    const double *zc = solver->zc;

    const double *u1 = solver->u1;
    const double *u2 = solver->u2;
    const double *u3 = solver->u3;

    FOR_INNER_CELL (i, j, k) {
        if (c3e(solver->flag, i, j, k) == FLAG_FLUID) {
            c3e(solver->S[1][1], i, j, k)
                = (c3e(u1, i+1, j, k) - c3e(u1, i-1, j, k)) / (c1e(xc, i+1) - c1e(xc, i-1));
            c3e(solver->S[1][2], i, j, k) = 0.5 * (
                (c3e(u1, i, j+1, k) - c3e(u1, i, j-1, k)) / (c1e(yc, j+1) - c1e(yc, j-1))
                + (c3e(u2, i+1, j, k) - c3e(u2, i-1, j, k)) / (c1e(xc, i+1) - c1e(xc, i-1))
            );
            c3e(solver->S[1][3], i, j, k) = 0.5 * (
                (c3e(u1, i, j, k+1) - c3e(u1, i, j, k-1)) / (c1e(zc, k+1) - c1e(zc, k-1))
                + (c3e(u3, i+1, j, k) - c3e(u3, i-1, j, k)) / (c1e(xc, i+1) - c1e(xc, i-1))
            );
            c3e(solver->S[2][2], i, j, k)
                = (c3e(u2, i, j+1, k) - c3e(u2, i, j-1, k)) / (c1e(yc, j+1) - c1e(yc, j-1));
            c3e(solver->S[2][3], i, j, k) = 0.5 * (
                (c3e(u2, i, j, k+1) - c3e(u2, i, j, k-1)) / (c1e(zc, k+1) - c1e(zc, k-1))
                + (c3e(u3, i, j+1, k) - c3e(u3, i, j-1, k)) / (c1e(yc, j+1) - c1e(yc, j-1))
            );
            c3e(solver->S[3][3], i, j, k)
                = (c3e(u3, i, j, k+1) - c3e(u3, i, j, k-1)) / (c1e(zc, k+1) - c1e(zc, k-1));
        }
    }
}
